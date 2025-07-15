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
        feat_static_cat: Tensor,          # (B, C_cat)
        past_time_feat: Tensor,           # (B, T_hist, n_time)
        past_target: Tensor,              # (B, T_hist, S)
        past_observed_values: Tensor,     # (B, T_hist, S)
        future_time_feat: Optional[Tensor],   # (B, P, n_time)
        future_target: Optional[Tensor],      # (B, P, S)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        inputs : Tensor  (B, sub_seq_len, S, F_per_series)
        scale  : Tensor  (B, 1, S)
        static_feat : Tensor  (B, S, n_static)
        """
    
        # ------------------------------------------------------------------ #
        # 1) choose the slice we need
        # ------------------------------------------------------------------ #
        if future_time_feat is None or future_target is None:
            time_feat     = past_time_feat[:, -self.context_length :, :]   # (B,T,n_time)
            sequence      = past_target                                    # (B,T,S)
            subseq_len    = self.context_length
            seq_length    = self.history_length
        else:
            time_feat = F.concat(
                past_time_feat[:, -self.context_length :, :],
                future_time_feat,
                dim=1,                              # (B, T+P, n_time)
            )
            sequence   = F.concat(past_target, future_target, dim=1)       # (B,T+P,S)
            subseq_len = self.context_length + self.prediction_length
            seq_length = self.history_length + self.prediction_length
    
        # ------------------------------------------------------------------ #
        # 2) lagged values  (B, sub_seq_len, S, n_lags)
        # ------------------------------------------------------------------ #
        lags = self.get_lagged_subsequences(
            F=F,
            sequence=sequence,
            sequence_length=seq_length,
            indices=self.lags_seq,
            subsequences_length=subseq_len,
        )                               # (B, sub_seq_len, S, n_lags)
    
        # ------------------------------------------------------------------ #
        # 3) scale on the *context* part of past_target, keep per‑series
        # ------------------------------------------------------------------ #
        _, scale = self.scaler(
            past_target[:, -self.context_length :, :],          # (B,C,S)
            past_observed_values[:, -self.context_length :, :], # (B,C,S)
        )                               # (B, 1, S)
    
        # ------------------------------------------------------------------ #
        # 4) categorical + log‑scale  →  static feature (B, S, n_static)
        # ------------------------------------------------------------------ #
        embedded_cat = self.embedder(feat_static_cat)           # (B, C_emb)
        print(f"embedded_cat shape before expand: {embedded_cat.shape}")
        embedded_cat = embedded_cat.expand_dims(axis=1)
        print(f"embedded_cat shape after expand: {embedded_cat.shape}")

        embedded_cat = embedded_cat.repeat(axis=1,
                                   repeats=self.S.shape[0])
        print(f"embedded_cat shape after repeat: {embedded_cat.shape}")
        
        log_scale    = F.log(scale.squeeze(axis=1))             # (B, S)
        print(f"log_scale shape: {log_scale.shape}")
            
        static_feat = F.concat(embedded_cat, log_scale, dim=-1)
        print(f"static_feat shape: {static_feat.shape}")
    
        # broadcast static_feat to (B, sub_seq_len, S, n_static)
        static_feat_b = static_feat.expand_dims(axis=1).repeat(
            axis=1, repeats=subseq_len
        )
    
        # ------------------------------------------------------------------ #
        # 5) broadcast time features  → (B, sub_seq_len, S, n_time)
        # ------------------------------------------------------------------ #
        time_feat_b = time_feat.expand_dims(axis=2).repeat(
            axis=2, repeats=self.S.shape[0]
        )
    
        # ------------------------------------------------------------------ #
        # 6) put everything together  ➜  (B, sub_seq_len, S, F_per_series)
        # ------------------------------------------------------------------ #
        lags_scaled  = F.broadcast_div(lags, scale.expand_dims(axis=-1))
    
        inputs = F.concat(lags_scaled, time_feat_b, static_feat_b, dim=-1)
    
        # ------------- diagnostics ----------------------------------------
        print("lags_scaled :", lags_scaled.shape)
        print("time_feat_b :", time_feat_b.shape)
        print("static_feat:", static_feat_b.shape)
        print("inputs     :", inputs.shape)  # (B, sub_seq_len, S, F_ps)
    
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
        self.sample_LH              = sample_LH
        self.batch_no = 0
        self.num_batches_per_epoch = num_batches_per_epoch
        self.epochs = epochs
        self.warmstart_epoch_frac = warmstart_epoch_frac
        self.warmstart_iter         = int(warmstart_epoch_frac * epochs * num_batches_per_epoch)

        # ---------- sanity checks --------------------------------------
        assert self.CRPS_weight >= 0.0, "CRPS weight must be non-negative"
        assert (
            self.likelihood_weight >= 0.0
        ), "Likelihood weight must be non-negative!"
        assert (
            self.likelihood_weight + self.CRPS_weight > 0.0
        ), "At least one of CRPS or likelihood weights must be non-zero"
        if self.CRPS_weight == 0.0 and self.coherent_train_samples:
            assert (
                "No sampling being performed. "
                "coherent_train_samples flag is ignored"
            )
        if not self.sample_LH == 0.0 and self.coherent_train_samples:
            assert (
                "No sampling being performed. "
                "coherent_train_samples flag is ignored"
            )
        if self.likelihood_weight == 0.0 and self.sample_LH:
            assert (
                "likelihood_weight is 0 but sample likelihoods are still "
                "being calculated. Set sample_LH=0 when likelihood_weight=0"
            )


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
