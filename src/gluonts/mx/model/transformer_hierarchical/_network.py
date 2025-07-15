from typing import List, Optional, Tuple

import logging
import numpy as np
import mxnet as mx

from gluonts.core.component import validated
from gluonts.itertools import prod
from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.distribution import (
    DistributionOutput,
    Distribution,
    EmpiricalDistribution,
    LowrankMultivariateGaussian,
)
from gluonts.mx.util import assert_shape, weighted_average

from gluonts.mx.model.deepvar_hierarchical._network import (
    reconcile_samples,
    coherency_error,
)

from .transformer_encoder import HierarchicalTransformerEncoder
from .transformer_decoder import HierarchicalTransformerDecoder

logger = logging.getLogger(__name__)

LARGE_NEGATIVE_VALUE = -9e8 


def _upper_triangular_mask(F, length: int) -> Tensor:
    mask = F.zeros_like(F.eye(length))
    for k in range(length - 1):
        mask = mask + F.eye(length, length, k + 1)
    return mask * LARGE_NEGATIVE_VALUE


class HierarchicalTransformerNetwork(mx.gluon.HybridBlock):
    """
    Identical to the vanilla TransformerNetwork but
    stores the hierarchy’s projection matrix `M` and summation matrix `S`.
    """

    @validated()
    def __init__(
        self,
        S,
        M,
        target_dim: int,
        likelihood_weight: float,
        CRPS_weight: float,
        warmstart_epoch_frac: float,
        coherent_train_samples: float,
        epochs: int,
        sample_LH: float,
        encoder: HierarchicalTransformerEncoder,
        decoder: HierarchicalTransformerDecoder,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        cardinality: List[int],
        embedding_dimension: int,
        lags_seq: List[int],
        seq_axis: Optional[List[int]] = None,
        scaling: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ---------------- hyper‑parameters -----------------------------
        self.history_length     = history_length
        self.context_length     = context_length
        self.prediction_length  = prediction_length
        self.scaling            = scaling

        lags_seq = sorted(set(lags_seq))
        self.lags_seq           = lags_seq
        self.target_shape       = distr_output.event_shape

        self.seq_axis           = seq_axis

        # hierarchy meta
        ctx = kwargs.get("ctx", None) or mx.cpu()
        self.M = M
        self.S = S

        # ---------------- sub‑modules ----------------------------------
        with self.name_scope():
            self.encoder        = encoder
            self.decoder        = decoder
            self.proj_dist_args = distr_output.get_args_proj()
            self.distr_output   = distr_output

            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=[embedding_dimension for _ in cardinality],
            )
            self.scaler = MeanScaler(keepdims=True) if scaling else NOPScaler(keepdims=True)

    @staticmethod
    def get_lagged_subsequences(
        F,
        sequence: Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> Tensor:
        
        assert max(indices) + subsequences_length <= sequence_length, (
            "lags cannot go further than history length, found lag"
            f" {max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(
                F.slice_axis(
                    sequence, axis=1, begin=begin_index, end=end_index
                )
            )

        return F.stack(*lagged_values, axis=-1)

    def create_network_input(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,  # (batch_size, num_features, history_length)
        past_target: Tensor,  # (batch_size, history_length, 1)
        past_observed_values: Tensor,  # (batch_size, history_length)
        future_time_feat: Optional[
            Tensor
        ],  # (batch_size, num_features, prediction_length)
        future_target: Optional[Tensor],  # (batch_size, prediction_length)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Creates inputs for the transformer network.

        All tensor arguments should have NTC layout.
        """

        if future_time_feat is None or future_target is None:
            time_feat = past_time_feat.slice_axis(
                axis=1,
                begin=self.history_length - self.context_length,
                end=None,
            )
            sequence = past_target
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = F.concat(
                past_time_feat.slice_axis(
                    axis=1,
                    begin=self.history_length - self.context_length,
                    end=None,
                ),
                future_time_feat,
                dim=1,
            )
            sequence = F.concat(past_target, future_target, dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags = self.get_lagged_subsequences(
            F=F,
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)
        _, scale = self.scaler(
            past_target.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
        )
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help
        # prediction too(batch_size, num_features + prod(target_shape))
        static_feat = F.concat(
            embedded_cat,
            (
                F.log(scale)
                if len(self.target_shape) == 0
                else F.log(scale.squeeze(axis=1))
            ),
            dim=1,
        )

        repeated_static_feat = static_feat.expand_dims(axis=1).repeat(
            axis=1, repeats=subsequences_length
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled = F.broadcast_div(lags, scale.expand_dims(axis=-1))

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = F.reshape(
            data=lags_scaled,
            shape=(
                -1,
                subsequences_length,
                len(self.lags_seq) * prod(self.target_shape),
            ),
        )

        # (batch_size, sub_seq_len, input_dim)
        inputs = F.concat(input_lags, time_feat, repeated_static_feat, dim=-1)

        return inputs, scale, static_feat

    def get_samples_for_loss(self, distr: Distribution) -> Tensor:
        """
        Get samples to compute the final loss. These are samples directly drawn
        from the given `distr` if coherence is not enforced yet; otherwise the
        drawn samples are reconciled.

        Parameters
        ----------
        distr
            Distribution instances

        Returns
        -------
        samples
            Tensor with shape (num_samples, batch_size, seq_len, target_dim)
        """
        samples = distr.sample_rep(
            num_samples=self.num_samples_for_loss, dtype="float32"
        )

        # Determine which epoch we are currently in.
        self.batch_no += 1
        epoch_no = self.batch_no // self.num_batches_per_epoch + 1
        epoch_frac = epoch_no / self.epochs

        if (
            self.coherent_train_samples
            and epoch_frac > self.warmstart_epoch_frac
        ):
            coherent_samples = reconcile_samples(
                reconciliation_mat=self.M,
                samples=samples,
                seq_axis=self.seq_axis,
            )
            assert_shape(coherent_samples, samples.shape)
            return coherent_samples
        else:
            return samples



class HierarchicalTransformerTrainingNetwork(HierarchicalTransformerNetwork):
    """
    Loss =   CRPS(samples) * CRPS_weight
           + NLL            * likelihood_weight
    """

    @validated()
    def __init__(
        self,
        *,
        num_samples_for_loss: int,
        likelihood_weight: float,
        CRPS_weight: float,
        coherent_train_samples: bool,
        warmstart_epoch_frac: float,
        epochs: int,
        num_batches_per_epoch: int,
        sample_LH: bool,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        # ---------- loss hyper‑params ----------------------------------
        self.num_samples_for_loss   = num_samples_for_loss
        self.likelihood_weight      = likelihood_weight
        self.CRPS_weight            = CRPS_weight
        self.coherent_train_samples = coherent_train_samples
        self.warmstart_iter         = int(warmstart_epoch_frac * epochs * num_batches_per_epoch)
        self.sample_LH              = sample_LH

        # ---------- sanity checks --------------------------------------
        assert self.CRPS_weight >= 0 or self.likelihood_weight >= 0
        assert (self.CRPS_weight + self.likelihood_weight) > 0

        # ---------- step counter ---------------------------------------
        self.batch_no = 0


    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:

        inputs, scale, _ = self.create_network_input(
            F,
            feat_static_cat,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )
        enc_in = inputs.slice_axis(axis=1, begin=0, end=self.context_length)
        dec_in = inputs.slice_axis(axis=1, begin=self.context_length, end=None)

        enc_out = self.encoder(enc_in)
        dec_hid = self.decoder(
            dec_in,
            enc_out,
            _upper_triangular_mask(F, self.prediction_length),
        )
        params  = self.proj_dist_args(dec_hid)
        distr   = self.distr_output.distribution(params, scale=scale)

        # ---------- loss components ------------------------------------
        loss = 0.0

        #  (a) CRPS -----------------------------------------------------
        if self.CRPS_weight > 0.0:
            samples = self.get_samples_for_loss(distr)
            crps = (
                EmpiricalDistribution(samples=samples, event_dim=1)
                .crps_univariate(future_target)
                .expand_dims(axis=-1)  # match shape for weighting
            )
            loss = loss + self.CRPS_weight * crps

        #  (b) NLL ------------------------------------------------------
        if self.likelihood_weight > 0.0:
            if self.sample_LH:
                # replace distr with one fitted on (coherent) samples
                if "samples" not in locals():
                    samples = self.get_samples_for_loss(distr)
                distr_LH = LowrankMultivariateGaussian.fit(
                    F, samples=samples, rank=0
                )
                nll = -distr_LH.log_prob(future_target).expand_dims(axis=-1)
            else:
                nll = distr.loss(future_target)
            loss = loss + self.likelihood_weight * nll

        # ---------- mask & reduce --------------------------------------
        weighted = weighted_average(
            F, x=loss, weights=future_observed_values, axis=1
        )
        return weighted.mean()




class HierarchicalTransformerPredictionNetwork(HierarchicalTransformerNetwork):
    @validated()
    def __init__(
        self,
        *,
        num_parallel_samples: int,
        coherent_pred_samples: bool,
        log_coherency_error: bool,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.num_parallel_samples   = num_parallel_samples
        self.coherent_pred_samples  = coherent_pred_samples
        self.log_coherency_error    = log_coherency_error
        if log_coherency_error:
            self.forecast_time_step = 1

        # lags for autoregressive decoding
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    # ------------------------------------------------------------------
    def sampling_decoder(
        self,
        F,
        static_feat: Tensor,
        past_target: Tensor,
        time_feat: Tensor,
        scale: Tensor,
        enc_out: Tensor,
    ) -> Tensor:
        """
        Autoregressive sampling identical to vanilla Transformer, but with
        an extra reconciliation step at the end (if enabled).
        """

        rep = self.num_parallel_samples
        pt  = past_target.repeat(repeats=rep, axis=0)
        tf  = time_feat.repeat(repeats=rep, axis=0)
        sf  = static_feat.repeat(repeats=rep, axis=0).expand_dims(axis=1)
        eo  = enc_out.repeat(repeats=rep, axis=0).expand_dims(axis=1)
        sc  = scale.repeat(repeats=rep, axis=0)

        future_samples = []

        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                F, pt, self.history_length + k,
                self.shifted_lags, subsequences_length=1,
            )
            lags_scaled = F.broadcast_div(lags, sc.expand_dims(axis=-1))
            input_lags  = F.reshape(
                lags_scaled,
                (-1, 1, len(self.lags_seq) * prod(self.target_shape)),
            )

            dec_in = F.concat(
                input_lags,
                tf.slice_axis(axis=1, begin=k, end=k + 1),
                sf,
                dim=-1,
            )

            dec_out = self.decoder(dec_in, eo, None, False)
            params  = self.proj_dist_args(dec_out)
            distr   = self.distr_output.distribution(params, scale=sc)
            y_new   = distr.sample()                       # (B*rep, 1, ...)
            pt      = F.concat(pt, y_new, dim=1)
            future_samples.append(y_new)

        self.decoder.cache_reset()

        samples = F.concat(*future_samples, dim=1)  # (B*rep, P, ...)
        samples = samples.reshape(
            (-1, rep) + self.target_shape + (self.prediction_length,)
        )  # (batch, rep, ..., P)

        # ------------- reconciliation ----------------------------------
        if self.coherent_pred_samples:
            samples = reconcile_samples(
                reconciliation_mat=self.M,
                samples=samples,
                seq_axis=self.seq_axis,
            )

        # ------------- optional logging --------------------------------
        if self.log_coherency_error:
            err = coherency_error(
                S=self.S,
                samples=samples.asnumpy(),
            )
            logger.info(
                f"Coherency error at horizon step {self.forecast_time_step}:"
                f" {err:.4e}"
            )
            self.forecast_time_step += 1

        return samples

    # hybrid_forward unchanged (same as vanilla) – copy from original file
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
    ) -> Tensor:
        """
        Predicts samples, all tensors should have NTC layout.

        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape)
        future_time_feat : (batch_size, prediction_length, num_features)

        Returns predicted samples
        -------
        """

        # create the inputs for the encoder
        inputs, scale, static_feat = self.create_network_input(
            F=F,
            feat_static_cat=feat_static_cat,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=None,
            future_target=None,
        )

        # pass through encoder
        enc_out = self.encoder(inputs)

        return self.sampling_decoder(
            F=F,
            past_target=past_target,
            time_feat=future_time_feat,
            static_feat=static_feat,
            scale=scale,
            enc_out=enc_out,
        )
