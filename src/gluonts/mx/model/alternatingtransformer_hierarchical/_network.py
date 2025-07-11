
from typing import List, Optional

import numpy as np

from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.mx import Tensor, F

from gluonts.mx.block.quantile_output import QuantileLoss, crps_weights_pwl

from gluonts.mx.distribution import Distribution, DistributionOutput
from gluonts.mx.distribution import EmpiricalDistribution
from gluonts.mx.distribution.lowrank_multivariate_gaussian import LowrankMultivariateGaussian

from . _layers import AlternatingAttentionLayer, AlternatingDecoderLayer, PositionalEncoding, generate_square_subsequent_mask, _get_activation_fn

def get_lagged_subsequences(F, sequence: Tensor, subsequence_length: int, lags_seq: List[int], axis: int = 1) -> Tensor:
    # This is a utility function to extract lagged subsequences from the input sequence
    # sequence shape (b, time, s)
    lagged_values = []
    max_lag = max(lags_seq)
    for lag in lags_seq:
        begin = -subsequence_length - max_lag + lag - 1
        end = -max_lag + lag - 1 if lag > 0 else None
        lagged_values.append(F.slice_axis(sequence, axis=axis, begin=begin, end=end).expand_dims(axis=-1))
    return F.concat(*lagged_values, dim=-1)

class AlternatingTransformerHierarchicalNetwork(HybridBlock):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        target_dim: int,
        feat_dynamic_real_dim: int,
        respect_hierarchy: bool,
        S: np.ndarray,
        distr_output: DistributionOutput,
        scaling: bool,
        lags_seq: List[int],
        seq_axis: Optional[List[int]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.respect_hierarchy = respect_hierarchy
        self.S = mx.nd.array(S)
        self.distr_output = distr_output
        self.scaling = scaling
        self.lags_seq = lags_seq
        self.seq_axis = seq_axis
        self.num_lags = len(lags_seq)
        self.target_linear = nn.Dense(in_units=1, units=d_model, flatten=False)
        self.lags_linear = nn.Dense(in_units=self.num_lags * target_dim, units=d_model, flatten=False)
        self.series_embed = nn.Embedding(input_dim=target_dim, embedding_dim=d_model)
        self.temporal_pos = PositionalEncoding(d_model)
        self.spatial_pos = PositionalEncoding(d_model)
        if feat_dynamic_real_dim > 0:
            self.feat_project = nn.Dense(in_units=feat_dynamic_real_dim, units=d_model, flatten=False)
        else:
            self.feat_project = None
        self.encoder_layers = nn.Sequential()
        for i in range(num_encoder_layers):
            self.encoder_layers.add(AlternatingAttentionLayer(d_model, nhead, dim_feedforward, dropout, activation, (i % 2 == 0)))
        self.decoder_layers = nn.Sequential()
        for i in range(num_decoder_layers):
            self.decoder_layers.add(AlternatingDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, (i % 2 == 0), context_length))
        self.args_proj = distr_output.get_args_proj()
        if respect_hierarchy:
            connectivity = F.dot(self.S, self.S, transpose_b=True) > 0
            self.spatial_mask = F.where(connectivity, F.zeros_like(connectivity), F.full(shape=connectivity.shape, val=float('-inf')))
        else:
            self.spatial_mask = None

    def get_distr(self, F, past_target: Tensor, feat_dynamic_real: Optional[Tensor] = None) -> Distribution:
        b, t, s = past_target.shape
        assert s == self.target_dim
        past_target = F.expand_dims(past_target, axis=-1)
        value_embed = self.target_linear(past_target)
        if self.lags_seq:
            past_lagged = get_lagged_subsequences(F, past_target.squeeze(-1), self.context_length, self.lags_seq)
            past_lagged = F.reshape(past_lagged, shape=(b, self.context_length, self.target_dim * self.num_lags))
            value_embed = value_embed + self.lags_linear(past_lagged).expand_dims(axis=2)
        series_ids = F.arange(0, s)
        series_embed = F.expand_dims(F.expand_dims(self.series_embed(series_ids), axis=0), axis=0).broadcast_to((b, t, -1, -1))
        temporal_pe = self.temporal_pos.pe[:t].expand_dims(1).broadcast_to((-1, s, -1))
        spatial_pe = self.spatial_pos.pe[:s].expand_dims(0).broadcast_to((t, -1, -1))
        embed = value_embed + series_embed + temporal_pe + spatial_pe
        if self.feat_project is not None:
            past_feat = F.slice_axis(feat_dynamic_real, axis=1, begin=0, end=t)
            feat_embed = self.feat_project(past_feat).expand_dims(2).broadcast_to((-1, -1, s, -1))
            embed = embed + feat_embed
        enc_output = embed
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, attn_mask=self.spatial_mask if layer.is_spatial else None)
        dec_input = F.zeros(shape=(b, self.prediction_length, s, 1))
        value_embed_dec = self.target_linear(dec_input)
        series_embed_dec = F.expand_dims(F.expand_dims(self.series_embed(series_ids), axis=0), axis=0).broadcast_to((b, self.prediction_length, -1, -1))
        temporal_pe_dec = self.temporal_pos.pe[:self.prediction_length].expand_dims(1).broadcast_to((-1, s, -1))
        spatial_pe_dec = self.spatial_pos.pe[:s].expand_dims(0).broadcast_to((self.prediction_length, -1, -1))
        dec_embed = value_embed_dec + series_embed_dec + temporal_pe_dec + spatial_pe_dec
        if self.feat_project is not None:
            future_feat = F.slice_axis(feat_dynamic_real, axis=1, begin=t, end=None)
            feat_embed_dec = self.feat_project(future_feat).expand_dims(2).broadcast_to((-1, -1, s, -1))
            dec_embed = dec_embed + feat_embed_dec
        dec_output = dec_embed
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, self_att_mask=None, cross_att_mask=self.spatial_mask if layer.is_spatial else None)
        distr_args = self.args_proj(dec_output)
        if self.scaling:
            scale = F.mean(F.abs(past_target.squeeze(-1)), axis=1, keepdims=True) + 1e-8
            scale = F.broadcast_to(scale, shape=(b, self.prediction_length, self.target_dim))
        else:
            scale = None
        distr = self.distr_output.distribution(distr_args, scale=scale)
        return distr

class AlternatingTransformerHierarchicalTrainingNetwork(AlternatingTransformerHierarchicalNetwork, DeepVARHierarchicalTrainingNetwork):
    def __init__(
        self,
        num_samples_for_loss: int,
        likelihood_weight: float,
        CRPS_weight: float,
        coherent_train_samples: bool,
        warmstart_epoch_frac: float,
        epochs: float,
        num_batches_per_epoch: float,
        sample_LH: bool,
        **kwargs
    ) -> None:
        AlternatingTransformerHierarchicalNetwork.__init__(self, **kwargs)
        DeepVARHierarchicalTrainingNetwork.__init__(
            self,
            num_samples_for_loss=num_samples_for_loss,
            likelihood_weight=likelihood_weight,
            CRPS_weight=CRPS_weight,
            coherent_train_samples=coherent_train_samples,
            warmstart_epoch_frac=warmstart_epoch_frac,
            epochs=epochs,
            num_batches_per_epoch=num_batches_per_epoch,
            sample_LH=sample_LH,
            **kwargs
        )

    def hybrid_forward(self, F, past_target: Tensor, feat_dynamic_real: Optional[Tensor] = None, future_target: Tensor, future_observed: Tensor) -> Tensor:
        distr = self.get_distr(F, past_target, feat_dynamic_real)
        loss = self.loss(F, future_target, distr)
        return loss

class AlternatingTransformerHierarchicalPredictionNetwork(AlternatingTransformerHierarchicalNetwork, DeepVARHierarchicalPredictionNetwork):
    @validated()
    def __init__(
        self,
        num_parallel_samples: int,
        log_coherency_error: bool,
        coherent_pred_samples: bool,
        **kwargs
    ) -> None:
        AlternatingTransformerHierarchicalNetwork.__init__(self, **kwargs)
        DeepVARHierarchicalPredictionNetwork.__init__(
            self,
            num_parallel_samples=num_parallel_samples,
            log_coherency_error=log_coherency_error,
            coherent_pred_samples=coherent_pred_samples,
            **kwargs
        )

    def hybrid_forward(self, F, past_target: Tensor, feat_dynamic_real: Optional[Tensor] = None) -> Tensor:
        distr = self.get_distr(F, past_target, feat_dynamic_real)
        samples = distr.sample(num_samples=self.num_parallel_samples, dtype='float32')
        samples = self.post_process_samples(samples)
        return samples
