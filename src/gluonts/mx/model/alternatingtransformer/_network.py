# gluonts/mx/model/alt_hier_transformer/_network.py
"""
Training / prediction networks that
* build 4-D tokens  (B, L, D, H)
* pass them through AlternatingHierEncoder
* keep the **standard Transformer decoder** (temporal only)
* reconcile samples with DeepVAR-Hierarchical utilities
"""
from typing import List, Tuple, Optional

import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import HybridBlock, nn
from gluonts.mx.util import assert_shape
from gluonts.mx import Tensor
from gluonts.mx.model.transformer.trans_decoder import TransformerDecoder 

from gluonts.mx.distribution import StudentT, EmpiricalDistribution
from gluonts.mx.model.deepvar_hierarchical._network import (
    reconcile_samples,
    coherency_error,
)
from .transencoder_alt import AlternatingHierEncoder

__all__: List[str] = [
    "AltHierTransformerTrainingNetwork",
    "AltHierTransformerPredictionNetwork",
]

# ----------------------------------------------------------------------
# helper
# ----------------------------------------------------------------------
def _dense_flatten_last(in_units: int, out_units: int) -> nn.Dense:
    """
    Dense layer that keeps *all* axes except the last untouched.
    """
    return nn.Dense(units=out_units, flatten=False, in_units=in_units)

@staticmethod
def get_lagged_subsequences(
    F,
    sequence: Tensor,
    sequence_length: int,
    indices: List[int],
    subsequences_length: int = 1,
) -> Tensor:
    """
    Returns lagged subsequences of a given sequence.

    Parameters
    ----------
    sequence : Tensor
        the sequence from which lagged subsequences should be extracted.
        Shape: (N, T, C).
    sequence_length : int
        length of sequence in the T (time) dimension (axis = 1).
    indices : List[int]
        list of lag indices to be used.
    subsequences_length : int
        length of the subsequences to be extracted.

    Returns
    --------
    lagged : Tensor
        a tensor of shape (N, S, C, I), where S = subsequences_length and
        I = len(indices), containing lagged subsequences. Specifically,
        lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
    """
    # we must have: sequence_length - lag_index - subsequences_length >= 0
    # for all lag_index, hence the following assert
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


# ----------------------------------------------------------------------
# common base
# ----------------------------------------------------------------------
class _BaseAltHierNetwork(HybridBlock):
    """
    Shared encoder, input build, & projection utilities.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        freq: str,
        lags_seq: List[int],
        num_series: int,
        model_dim: int,
        encoder: AlternatingHierEncoder,
        decoder: TransformerDecoder,
        proj: nn.HybridBlock,
        M: np.ndarray,
        S: np.ndarray,
        num_parallel_samples: int = 100,
        cardinality: Optional[List[int]] = None,
        embedding_dim: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.context_length = context_length
        self.pred_length = prediction_length
        self.freq = freq
        self.lags_seq = lags_seq
        self.num_series = num_series
        self.model_dim = model_dim
        self.num_parallel_samples = num_parallel_samples
        self.M = M  # reconciliation matrix (D, D)
        self.S = S

        with self.name_scope():
            # static cat embeddings
            self.cardinality = cardinality or [1]
            self.embed = (
                nn.Embedding(sum(self.cardinality), embedding_dim)
                if self.cardinality != [1]
                else None
            )

            # dense projection into model_dim, preserves 4-D shape
            self.input_proj = _dense_flatten_last(
                in_units=None, out_units=model_dim
            )

            self.encoder = encoder
            self.decoder = decoder
            self.proj_out = proj  # → dist params

    # ------------------------------------------------------------------
    #  create network input  (returns enc_input_4d, dec_input, scale)
    # ------------------------------------------------------------------
    def _create_network_input(
        self,
        F,
        past_target: Tensor,
        past_time_feat: Tensor,
        past_is_pad: Tensor,
        future_time_feat: Tensor,
        future_target: Optional[Tensor] = None,
        static_feat: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Builds the 4-D token tensor (B, L, D, model_dim)
        and the 3-D decoder input  (B·D, L, model_dim).
        """
        B, _, D = past_target.shape
        L = self.context_length + self.pred_length

        # -------- 1) scale  & lags ------------------------------------
        scale, past_target_scaled, future_target_scaled = _get_scale_and_scaled_targets(
            F,
            past_target,
            future_target,
        )

        full_target = F.concat(past_target_scaled, future_target_scaled, dim=1)
        # lagged subseqs  shape : (B, Lags, L, D) → (B, L, D, Lags)
        lagged = _get_lagged_subsequences(
            F,
            sequence=full_target,
            sequence_length=full_target.shape[1],
            indices=self.lags_seq,
            subsequences_length=L,
        ).transpose((0, 2, 3, 1))  # B, L, D, Lags

        # -------- 3) static features --------------------------------
        if static_feat is not None:
            static_emb = self.embed(static_feat).reshape((B, D, -1))  # (B,D,E)
            static_emb = F.expand_dims(static_emb, 1).broadcast_like(time_feat)
        else:
            static_emb = F.zeros_like(time_feat).slice_axis(axis=-1, begin=0, end=0)

        # log-scale  (B,D,1) → broadcast
        log_scale = F.log(scale + 1e-8).expand_dims(1).broadcast_axes(
            axis=(1,), size=(L,)
        ).expand_dims(-1)  # (B, L, D, 1)

        # -------- 4) concat & project -------------------------------
        token_raw = F.concat(lagged, time_feat, static_emb, log_scale, dim=-1)
        token_emb = self.input_proj(token_raw)  # (B, L, D, H)

        # -------- 5) build decoder input -----------------------------
        # flatten batch & series ⇒ (B·D, L, H)
        enc_out_shape = (B, L, D, self.model_dim)
        dec_input = token_emb.reshape((-1, L, self.model_dim))

        print(f"Encoder tokens shape: {token_emb.shape}")
        print(f"Decoder input shape: {dec_input.shape}")

        return token_emb, dec_input, scale

    # ------------------------------------------------------------------
    #  utils
    # ------------------------------------------------------------------
    def _post_process_samples(self, samples: Tensor, seq_axis: int) -> Tensor:
        """
        Reconcile to obtain coherent samples.
        """
        return reconcile_samples(self.M, samples, seq_axis=seq_axis)


# ----------------------------------------------------------------------
# Training network
# ----------------------------------------------------------------------
class AltHierTransformerTrainingNetwork(_BaseAltHierNetwork):
    """
    Adds DeepVAR-Hierarchical CRPS+NLL loss.
    """

    def __init__(
        self,
        num_samples_for_loss: int = 100,
        likelihood_weight: float = 0.0,
        CRPS_weight: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_samples_for_loss = num_samples_for_loss
        self.likelihood_weight = likelihood_weight
        self.CRPS_weight = CRPS_weight

    # --------------------------------------------------------------
    def hybrid_forward(
        self,
        F,
        past_target,
        past_time_feat,
        past_is_pad,
        future_time_feat,
        future_target,
        static_feat=None,
    ):
        # 1) Build inputs
        enc_input, dec_input, scale = self._create_network_input(
            F,
            past_target,
            past_time_feat,
            past_is_pad,
            future_time_feat,
            future_target,
            static_feat,
        )

        # 2) Encoder  (B,L,D,H)
        enc_output = self.encoder(enc_input)

        # 3) Decoder  (flatten series)
        dec_output = self.decoder(
            dec_input,
            enc_out,
            self.upper_triangular_mask(F, self.prediction_length),
        )

        # 4) Distribution head
        distr_params = self.proj_out(dec_output)  # (B·D, L, n_params)
        distr_params = distr_params.reshape(
            (-1, self.pred_length, self.num_series, -1)
        )
        distr = StudentT(*gluon.utils.split_and_load(distr_params, axis=-1, num_outputs=3))

        # 5) Loss
        samples = distr.sample_rep(self.num_samples_for_loss)  # (K,B,P,D)
        samples = self._post_process_samples(samples, seq_axis=2)

        nll = -distr.log_prob(future_target).expand_dims(axis=0)
        crps = EmpiricalDistribution(samples=samples, event_dim=1).crps_univariate(
            x=future_target
        ).expand_dims(axis=0)
        return self.likelihood_weight * nll + self.CRPS_weight * crps


# ----------------------------------------------------------------------
# Prediction network
# ----------------------------------------------------------------------
class AltHierTransformerPredictionNetwork(_BaseAltHierNetwork):
    """
    Same architecture; just overrides sampling behaviour.
    """

    def __init__(
        self,
        coherent_pred_samples: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.coherent_pred_samples = coherent_pred_samples

    def hybrid_forward(
        self,
        F,
        past_target,
        past_time_feat,
        past_is_pad,
        future_time_feat,
        static_feat=None,
    ):
        # Build inputs (future_target unknown)
        enc_input, dec_input, _ = self._create_network_input(
            F,
            past_target,
            past_time_feat,
            past_is_pad,
            future_time_feat,
            future_target=None,
            static_feat=static_feat,
        )

        enc_output = self.encoder(enc_input)

        dec_output, _ = self.decoder(
            dec_input,
            enc_output.reshape((-1, enc_output.shape[1], self.model_dim)),
            _make_causal_mask(F, dec_input, past_valid_length=self.context_length),
        )

        distr_params = self.proj_out(dec_output)  # (B·D, L, n_params)
        distr_params = distr_params.reshape(
            (-1, self.pred_length, self.num_series, -1)
        )
        distr = StudentT(*gluon.utils.split_and_load(distr_params, axis=-1, num_outputs=3))

        # draw samples
        samples = distr.sample_rep(self.num_parallel_samples)  # (K,B,P,D)
        if self.coherent_pred_samples:
            samples = self._post_process_samples(samples, seq_axis=2)

        return samples
