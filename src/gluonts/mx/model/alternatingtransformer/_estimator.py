# -----------------------------------------------------------------------------#
# Estimator for Alternating-Attention Transformer on Hierarchical Data          #
# -----------------------------------------------------------------------------#

from typing import List, Optional

import numpy as np
from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    RemoveFields,
    SetField,
    SelectFields,
)
from gluonts.transform.sampler import (
    ExpectedNumInstanceSampler,
    ValidationSplitSampler,
    TestSplitSampler,
)
from gluonts.time_feature import (
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.trainer import Trainer
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.mx.batchify import batchify
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.mx.model.predictor import RepresentableBlockPredictor

from .alternating_encoder import AlternatingTransformerEncoder
from .mlp_decoder import HierMLPDecoder
from ._network import (
    AltHierTrainingNetwork,
    AltHierPredictionNetwork,
)

# -----------------------------------------------------------------------------#


class AltTransformerHierarchicalEstimator(GluonEstimator):
    """
    Alternating-attention Transformer with optional OLS or bottom-up
    reconciliation.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        S: np.ndarray,  # (M × B) summation matrix
        cardinality: List[int],
        *,
        embedding_dimension: int = 20,
        context_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
        dropout_rate: float = 0.1,
        distr_output: DistributionOutput = StudentTOutput(),
        model_dim: int = 32,
        inner_ff_dim_scale: int = 4,
        num_heads: int = 8,
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List] = None,
        reconciliation_method: str = "ols",  # "ols" | "bottom_up"
        num_parallel_samples: int = 100,
        coherent_train_samples: bool = True,
        batch_size: int = 32,
    ) -> None:
        super().__init__(trainer=trainer, batch_size=batch_size)

        assert reconciliation_method in {"ols", "bottom_up"}

        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length
        self.S = S.astype("float32")
        self.bottom_count = S.shape[1]
        self.cardinality = cardinality

        self.lags_seq = lags_seq or get_lags_for_frequency(freq)
        self.time_features = (
            time_features or time_features_from_frequency_str(freq)
        )
        self.history_length = self.context_length + max(self.lags_seq)

        # encoder config
        self.config = dict(
            model_dim=model_dim,
            dropout_rate=dropout_rate,
            inner_ff_dim_scale=inner_ff_dim_scale,
            num_heads=num_heads,
            act_type="softrelu",
            pre_seq="dn",
            post_seq="drn",
        )

        self.encoder = AlternatingTransformerEncoder(
            num_timesteps=self.context_length,
            num_series=self.bottom_count,
            config=self.config,
        )
        self.distr_output = distr_output
        self.embedding_dimension = embedding_dimension
        self.scaling = scaling
        self.reconciliation_method = reconciliation_method
        self.num_parallel_samples = num_parallel_samples
        self.coherent_train_samples = coherent_train_samples

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.val_sampler = ValidationSplitSampler(
            min_future=prediction_length
        )

    # ------------------------------------------------------------------ #
    # transformation                                                     #
    # ------------------------------------------------------------------ #
    def create_transformation(self):
        return Chain(
            [
                RemoveFields(
                    [FieldName.FEAT_DYNAMIC_CAT, FieldName.FEAT_STATIC_REAL]
                ),
                SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0]),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=2),
                AddObservedValuesIndicator(
                    FieldName.TARGET, FieldName.OBSERVED_VALUES
                ),
                AddTimeFeatures(
                    FieldName.START,
                    FieldName.TARGET,
                    FieldName.FEAT_TIME,
                    self.time_features,
                    self.prediction_length,
                ),
                AddAgeFeature(
                    FieldName.TARGET,
                    FieldName.FEAT_AGE,
                    self.prediction_length,
                    True,
                ),
            ]
        )

    # ------------------------------------------------------------------ #
    # instance splitters & loaders                                        #
    # ------------------------------------------------------------------ #
    def _splitter(self, mode: str):
        from gluonts.transform import InstanceSplitter

        sampler = dict(
            training=self.train_sampler,
            validation=self.val_sampler,
            test=TestSplitSampler(),
        )[mode]

        return InstanceSplitter(
            FieldName.TARGET,
            FieldName.IS_PAD,
            FieldName.START,
            FieldName.FORECAST_START,
            sampler,
            self.history_length,
            self.prediction_length,
            [FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES],
        )

    def create_training_data_loader(self, data, **kwargs):
        names = get_hybrid_forward_input_names(AltHierTrainingNetwork)
        return TrainDataLoader(
            dataset=data,
            transform=self._splitter("training") + SelectFields(names),
            batch_size=self.batch_size,
            stack_fn=lambda x: batchify(
                x, ctx=self.trainer.ctx, dtype="float32"
            ),
            **kwargs,
        )

    def create_validation_data_loader(self, data, **kwargs):
        names = get_hybrid_forward_input_names(AltHierTrainingNetwork)
        return ValidationDataLoader(
            dataset=data,
            transform=self._splitter("validation") + SelectFields(names),
            batch_size=self.batch_size,
            stack_fn=lambda x: batchify(
                x, ctx=self.trainer.ctx, dtype="float32"
            ),
        )

    # ------------------------------------------------------------------ #
    # networks                                                           #
    # ------------------------------------------------------------------ #
    def create_training_network(self) -> HybridBlock:
        decoder = HierMLPDecoder(
            bottom_count=self.bottom_count,
            param_dim=self.distr_output.args_dim,
            hidden=self.config["model_dim"] * 2,
            dropout=self.config["dropout_rate"],
        )
        return AltHierTrainingNetwork(
            decoder=decoder,
            coherent_train_samples=self.coherent_train_samples,
            encoder=self.encoder,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            S=self.S,
            distr_output=self.distr_output,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            reconciliation_method=self.reconciliation_method,
            scaling=self.scaling,
        )

    def create_predictor(
        self, transformation, trained_network
    ) -> RepresentableBlockPredictor:
        pred_net = AltHierPredictionNetwork(
            num_parallel_samples=self.num_parallel_samples,
            encoder=self.encoder,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            S=self.S,
            distr_output=self.distr_output,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            reconciliation_method=self.reconciliation_method,
            scaling=self.scaling,
        )
        copy_parameters(trained_network, pred_net)

        return RepresentableBlockPredictor(
            input_transform=transformation + self._splitter("test"),
            prediction_net=pred_net,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
