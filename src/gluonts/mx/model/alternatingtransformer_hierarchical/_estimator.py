# src/gluonts/mx/model/alternating_transformer_hierarchical/_estimator.py
import numpy as np
from typing import Dict, List, Optional

from gluonts.core.component import validated
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import QuantileForecast, SampleForecast
from gluonts.model.predictor import Predictor
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str, get_lags_for_frequency
from gluonts.transform import (
    Transformation,
    Chain,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    SetField,
    VstackFeatures,
)

from ._network import AlternatingTransformerHierarchicalTrainingNetwork, AlternatingTransformerHierarchicalPredictionNetwork
from gluonts.mx.distribution import LowrankMultivariateGaussianOutput

class HierarchicalQuantileForecastPredictor(Predictor):
    def __init__(
        self,
        input_transform: Transformation,
        prediction_net: HybridBlock,
        batch_size: int,
        prediction_length: int,
        ctx: mx.Context,
        quantiles: List[float],
        freq: str,
        num_parallel_samples: int,
    ) -> None:
        super().__init__(input_transform=input_transform, batch_size=batch_size)
        self.prediction_net = prediction_net
        self.prediction_length = prediction_length
        self.ctx = ctx
        self.quantiles = quantiles
        self.freq = freq
        self.num_parallel_samples = num_parallel_samples

    def predict(self, dataset, num_samples: Optional[int] = None):
        num_samples = num_samples if num_samples is not None else self.num_parallel_samples
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            stack_fn=lambda data: data,
            ctx=self.ctx,
        )
        for batch in inference_data_loader:
            with mx.autograd.pause():
                past_target = batch[FieldName.PAST_TARGET]
                feat_dynamic_real = batch.get(FieldName.FEAT_DYNAMIC_REAL, None)
                samples = self.prediction_net(past_target, feat_dynamic_real)
                # samples (num_samples, b, p, target_dim)
                b = samples.shape[1]
                for i in range(batch['forecast_start'].shape[0]):
                    samples_i = samples[:, i, :, :]
                    quantile_array = nd.percentile(samples_i, nd.array(self.quantiles) * 100, axis=0).asnumpy().T
                    yield QuantileForecast(
                        forecast_array=quantile_array,
                        start_date = batch[FieldName.START][i] + (len(batch[FieldName.PAST_TARGET][i]) - self.prediction_net.context_length),
                        freq=self.freq,
                        forecast_keys=[str(q) for q in self.quantiles],
                    )

class AlternatingTransformerHierarchicalEstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        S: np.ndarray,
        prediction_length: int,
        context_length: Optional[int] = None,
        freq: str = "H",
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        activation: str = "gelu",
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        feat_dynamic_real_dim: int = 0,
        respect_hierarchy: bool = False,
        num_samples_for_loss: int = 200,
        likelihood_weight: float = 0.0,
        CRPS_weight: float = 1.0,
        sample_LH: bool = False,
        coherent_train_samples: bool = True,
        coherent_pred_samples: bool = True,
        warmstart_epoch_frac: float = 0.0,
        seq_axis: Optional[List[int]] = None,
        log_coherency_error: bool = True,
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        trainer: Trainer = Trainer(hybridize=False),
        batch_size: int = 32,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__(trainer=trainer, batch_size=batch_size)
        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length * 2
        self.freq = freq
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.quantiles = quantiles
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.respect_hierarchy = respect_hierarchy
        self.num_samples_for_loss = num_samples_for_loss
        self.likelihood_weight = likelihood_weight
        self.CRPS_weight = CRPS_weight
        self.sample_LH = sample_LH
        self.coherent_train_samples = coherent_train_samples
        self.coherent_pred_samples = coherent_pred_samples
        self.warmstart_epoch_frac = warmstart_epoch_frac
        self.seq_axis = seq_axis
        self.log_coherency_error = log_coherency_error
        self.scaling = scaling
        self.lags_seq = lags_seq or sorted(list(get_lags_for_frequency(self.freq).values()))
        self.history_length = self.context_length + max(self.lags_seq) if self.lags_seq else self.context_length
        self.time_features = time_features_from_frequency_str(self.freq)
        self.target_dim = S.shape[0]
        self.S = S
        A = constraint_mat(S)
        M = null_space_projection_mat(A)
        self.A = mx.nd.array(A)
        self.M = mx.nd.array(M)
        self.distr_output = LowrankMultivariateGaussianOutput(self.target_dim, rank=0)
        self.num_parallel_samples = num_parallel_samples

    def create_transformation(self) -> Transformation:
        transformations = [
            AsNumpyArray(field=FieldName.TARGET, expected_ndim=2),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET, output_field=FieldName.OBSERVED_VALUES
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=self.time_features,
                pred_length=self.prediction_length,
            ),
            VstackFeatures(
                input_fields=[FieldName.FEAT_TIME],
                output_field=FieldName.FEAT_DYNAMIC_REAL,
            ),
        ]
        if self.feat_dynamic_real_dim > 0:
            transformations += [
                AsNumpyArray(field=FieldName.FEAT_DYNAMIC_REAL, expected_ndim=2),
                VstackFeatures(
                    input_fields=[FieldName.FEAT_DYNAMIC_REAL, FieldName.FEAT_TIME],
                    output_field=FieldName.FEAT_DYNAMIC_REAL,
                ),
            ]
        return Chain(transformations)

    def create_training_network(self) -> HybridBlock:
        return AlternatingTransformerHierarchicalTrainingNetwork(
            M=self.M,
            A=self.A,
            num_samples_for_loss=self.num_samples_for_loss,
            likelihood_weight=self.likelihood_weight,
            CRPS_weight=self.CRPS_weight,
            seq_axis=self.seq_axis,
            coherent_train_samples=self.coherent_train_samples,
            warmstart_epoch_frac=self.warmstart_epoch_frac,
            epochs=self.trainer.epochs,
            num_batches_per_epoch=self.trainer.num_batches_per_epoch,
            sample_LH=self.sample_LH,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=len(self.time_features) + self.feat_dynamic_real_dim,
            respect_hierarchy=self.respect_hierarchy,
            S=self.S,
            distr_output=self.distr_output,
            scaling=self.scaling,
            lags_seq=self.lags_seq,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = AlternatingTransformerHierarchicalPredictionNetwork(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=len(self.time_features) + self.feat_dynamic_real_dim,
            respect_hierarchy=self.respect_hierarchy,
            S=self.S,
            distr_output=self.distr_output,
            scaling=self.scaling,
            lags_seq=self.lags_seq,
            num_parallel_samples=self.num_parallel_samples,
            log_coherency_error=self.log_coherency_error,
            coherent_pred_samples=self.coherent_pred_samples,
            M=self.M,
            A=self.A,
            seq_axis=self.seq_axis,
        )
        copy_parameters(trained_network, prediction_network)

        return HierarchicalQuantileForecastPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            quantiles=self.quantiles,
            freq=self.freq,
            num_parallel_samples=self.num_parallel_samples,
        )
