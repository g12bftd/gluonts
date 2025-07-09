# -----------------------------------------------------------------------------#
# Estimator – deterministic, coherence-aware Transformer                       #
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
from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str
from gluonts.mx.trainer import Trainer
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.mx.batchify import batchify
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.mx.model.predictor import RepresentableBlockPredictor

from .alternating_encoder import AlternatingTransformerEncoder
from ._network import (
    HierMLPDecoder,
    AltHierTrainingNetwork,
    AltHierPredictionNetwork,
)

# -----------------------------------------------------------------------------#
class AltTransformerHierarchicalEstimator(GluonEstimator):
    """
    Encoder depth controlled by `num_layers`.
    MLP decoder outputs deterministic bottom-level forecasts; these are
    reconciled to coherent forecasts inside the networks.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        S: np.ndarray,                    # (M × B)
        cardinality: List[int],
        *,
        num_layers: int = 4,
        embedding_dimension: int = 20,
        context_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
        dropout_rate: float = 0.1,
        model_dim: int = 32,
        inner_ff_dim_scale: int = 4,
        num_heads: int = 8,
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List] = None,
        reconciliation_method: str = "ols",
        batch_size: int = 32,
    ):
        super().__init__(trainer=trainer, batch_size=batch_size)

        self.pred_len = prediction_length
        self.context_len = context_length or prediction_length
        self.S = S.astype("float32")
        self.bottom = S.shape[1]
        self.cardinality = cardinality

        self.lags_seq = lags_seq or get_lags_for_frequency(freq)
        self.t_feat = time_features or time_features_from_frequency_str(freq)
        self.hist_len = self.context_len + max(self.lags_seq)

        cfg = dict(
            model_dim=model_dim,
            dropout_rate=dropout_rate,
            inner_ff_dim_scale=inner_ff_dim_scale,
            num_heads=num_heads,
            num_layers=num_layers,
            pre_seq="dn",
            post_seq="drn",
            act_type="softrelu",
        )
        self.encoder = AlternatingTransformerEncoder(
            num_timesteps=self.context_len, num_series=self.bottom, config=cfg
        )

        self.embedding_dimension = embedding_dimension
        self.scaling = scaling
        self.recon_method = reconciliation_method

        self.train_sampler = ExpectedNumInstanceSampler(1.0, min_future=prediction_length)
        self.val_sampler = ValidationSplitSampler(min_future=prediction_length)

    # transformation -----------------------------------------------------------
    def create_transformation(self):
        return Chain(
            [
                RemoveFields([FieldName.FEAT_DYNAMIC_CAT, FieldName.FEAT_STATIC_REAL]),
                SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0]),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=2),
                AddObservedValuesIndicator(FieldName.TARGET, FieldName.OBSERVED_VALUES),
                AddTimeFeatures(
                    FieldName.START,
                    FieldName.TARGET,
                    FieldName.FEAT_TIME,
                    self.t_feat,
                    self.pred_len,
                ),
                AddAgeFeature(FieldName.TARGET, FieldName.FEAT_AGE, self.pred_len, True),
            ]
        )

    # instance splitter --------------------------------------------------------
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
            self.hist_len,
            self.pred_len,
            [FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES],
        )

    # data loaders -------------------------------------------------------------
    def _loader(self, net, data, mode: str):
        names = get_hybrid_forward_input_names(net)
        return (TrainDataLoader if mode == "training" else ValidationDataLoader)(
            dataset=data,
            transform=self._splitter(mode) + SelectFields(names),
            batch_size=self.batch_size,
            stack_fn=lambda b: batchify(b, ctx=self.trainer.ctx, dtype="float32"),
        )

    def create_training_data_loader(self, data, **kw):
        return self._loader(AltHierTrainingNetwork, data, "training")

    def create_validation_data_loader(self, data, **kw):
        return self._loader(AltHierTrainingNetwork, data, "validation")

    # networks -----------------------------------------------------------------
    def _make_decoder(self):
        return HierMLPDecoder(
            bottom_count=self.bottom,
            pred_len=self.pred_len,
            hidden=self.encoder.config["model_dim"] * 2,
            dropout=self.encoder.config["dropout_rate"],
        )

    def create_training_network(self) -> HybridBlock:
        return AltHierTrainingNetwork(
            decoder=self._make_decoder(),
            encoder=self.encoder,
            history_length=self.hist_len,
            context_length=self.context_len,
            prediction_length=self.pred_len,
            S=self.S,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            reconciliation_method=self.recon_method,
            scaling=self.scaling,
        )

    def create_predictor(self, transformation, trained_network) -> RepresentableBlockPredictor:
        pred_net = AltHierPredictionNetwork(
            decoder=self._make_decoder(),
            encoder=self.encoder,
            history_length=self.hist_len,
            context_length=self.context_len,
            prediction_length=self.pred_len,
            S=self.S,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            reconciliation_method=self.recon_method,
            scaling=self.scaling,
        )
        copy_parameters(trained_network, pred_net)

        return RepresentableBlockPredictor(
            input_transform=transformation + self._splitter("test"),
            prediction_net=pred_net,
            batch_size=self.batch_size,
            prediction_length=self.pred_len,
            ctx=self.trainer.ctx,
        )
