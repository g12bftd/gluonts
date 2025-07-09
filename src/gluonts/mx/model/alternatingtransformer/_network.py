# -----------------------------------------------------------------------------#
# Alternating-Attention Transformer with Hierarchical Reconciliation            #
# -----------------------------------------------------------------------------#
# Two reconciliation styles:                                                    #
#   • "ols"        – ordinary-least-squares projection P = S (SᵀS)⁻¹ Sᵀ        #
#   • "bottom_up"  – simple aggregation      y_all = S · y_bottom              #
# -----------------------------------------------------------------------------#

from typing import List, Optional, Tuple

import numpy as np
import mxnet as mx
from mxnet.gluon import HybridBlock

from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.util import weighted_average
from gluonts.itertools import prod
from gluonts.core.component import validated

from .alternating_encoder import AlternatingTransformerEncoder
from .mlp_decoder import HierMLPDecoder

# -----------------------------------------------------------------------------#
# helpers                                                                      #
# -----------------------------------------------------------------------------#


def projection_matrix(S: np.ndarray) -> np.ndarray:
    """
    Ordinary-least-squares reconciliation matrix.

    S : (M × B) summation matrix,  B bottom series → M all nodes
    P = S (Sᵀ S)⁻¹ Sᵀ
    """
    St = S.T
    inv = np.linalg.inv(St @ S)
    return (S @ inv @ St).astype("float32")


LARGE_NEGATIVE_VALUE = -1.0e9

# -----------------------------------------------------------------------------#
# Backbone (shared by training & inference)                                    #
# -----------------------------------------------------------------------------#


class AltHierNetworkBase(HybridBlock):
    """Encoder/decoder backbone with reconciliation."""

    @validated()
    def __init__(
        self,
        *,
        encoder: AlternatingTransformerEncoder,
        history_length: int,
        context_length: int,
        prediction_length: int,
        S: np.ndarray,  # (M × B)
        distr_output: DistributionOutput,
        cardinality: List[int],
        embedding_dimension: int,
        lags_seq: List[int],
        reconciliation_method: str = "ols",  # "ols" | "bottom_up"
        scaling: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert reconciliation_method in {"ols", "bottom_up"}
        self.recon_method = reconciliation_method

        # ---------- static members ------------------------------------------- #
        self.encoder = encoder
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.scaling = scaling
        self.lags_seq = sorted(lags_seq)
        self.target_shape = distr_output.event_shape or (1,)

        # ---------- hierarchy matrices (NumPy) ------------------------------- #
        self.S_np = S.astype("float32")  # (M, B)
        self.R_np = (
            projection_matrix(S)
            if reconciliation_method == "ols"
            else S.astype("float32")
        )  # (M, B)

        # ---------- blocks --------------------------------------------------- #
        with self.name_scope():
            self.proj_dist_args = distr_output.get_args_proj()
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=[embedding_dimension for _ in cardinality],
            )
            self.scaler = (
                MeanScaler(keepdims=True)
                if scaling
                else NOPScaler(keepdims=True)
            )

    # ------------------------------------------------------------------ #
    # lag helper (same as vanilla Transformer)                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_lagged_subsequences(
        F, sequence: Tensor, sequence_length: int, indices: List[int], subsequences_length: int = 1
    ) -> Tensor:
        assert max(indices) + subsequences_length <= sequence_length
        lagged = [
            F.slice_axis(
                sequence,
                axis=1,
                begin=-lag - subsequences_length,
                end=-lag if lag > 0 else None,
            )
            for lag in indices
        ]
        return F.stack(*lagged, axis=-1)

    @staticmethod
    def upper_triangular_mask(F, d):
        mask = F.zeros_like(F.eye(d))
        for k in range(d - 1):
            mask = mask + F.eye(d, d, k + 1)
        return mask * LARGE_NEGATIVE_VALUE


# -----------------------------------------------------------------------------#
# Training network                                                             #
# -----------------------------------------------------------------------------#


class AltHierTrainingNetwork(AltHierNetworkBase):
    """Network used during training."""

    @validated()
    def __init__(
        self,
        *,
        decoder: HierMLPDecoder,
        coherent_train_samples: bool,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        self.decoder = decoder
        self.coherent_train_samples = coherent_train_samples

    # ---------------- create_network_input (keeps bottom axis) --------------- #
    def create_network_input(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,  # (B, H, BOTTOM)
        past_observed_values: Tensor,
        future_time_feat: Optional[Tensor],
        future_target: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if future_time_feat is None or future_target is None:
            time_feat = past_time_feat.slice_axis(
                axis=1, begin=self.history_length - self.context_length, end=None
            )
            sequence = past_target
            seq_len = self.history_length
            sub_len = self.context_length
        else:
            time_feat = np.concat(
                past_time_feat.slice_axis(
                    axis=1, begin=self.history_length - self.context_length, end=None
                ),
                future_time_feat,
                dim=1,
            )
            sequence = np.concat(past_target, future_target, dim=1)
            seq_len = self.history_length + self.prediction_length
            sub_len = self.context_length + self.prediction_length

        lags = self.get_lagged_subsequences(
            F, sequence, seq_len, self.lags_seq, sub_len
        )  # (B, sub_len, BOTTOM, num_lags)

        # scale on context window
        _, scale = self.scaler(
            past_target.slice_axis(axis=1, begin=-self.context_length, end=None),
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
        )  # (B, 1, BOTTOM)

        embedded_cat = self.embedder(feat_static_cat)
        static_feat = np.concat(
            embedded_cat, np.log(scale.squeeze(axis=1)), dim=1
        )
        repeated_static = static_feat.expand_dims(axis=1).repeat(
            axis=1, repeats=sub_len
        )

        lags_scaled = np.broadcast_div(lags, scale.expand_dims(axis=-1))
        input_lags = lags_scaled.reshape(
            (-1, sub_len, len(self.lags_seq) * prod(self.target_shape))
        )

        # shape (B, sub_len, 1, input_dim) – we keep bottom dim placeholder
        inputs = np.concat(input_lags, time_feat, repeated_static, dim=-1).expand_dims(axis=2)
        return inputs, scale, static_feat

    # ------------------------------------------------------------------ #
    # forward / loss                                                     #
    # ------------------------------------------------------------------ #
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

        print("[Train] enc in :", inputs.shape)
        enc_out = self.encoder(inputs)
        print("[Train] enc out:", enc_out.shape)

        dec_out = self.decoder(enc_out)  # (B, T_pred, BOTTOM, param)
        print("[Train] dec out:", dec_out.shape)
        distr_args = self.proj_dist_args(dec_out)
        distr = self.distr_output.distribution(distr_args, scale=scale)

        # -------------------------------------------------- reconciliation ---- #
        if self.coherent_train_samples:
            samples = distr.sample()  # (B, T_pred, BOTTOM)
            R = np.array(self.R_np, ctx=samples.context)  # (M, B)
            coh = np.dot(samples.reshape((-1, R.shape[1])), R.T).reshape(
                samples.shape[:-1] + (R.shape[0],)
            )
            loss = distr.loss(coh)
        else:
            loss = distr.loss(future_target)

        weighted = weighted_average(
            F=F, x=loss, weights=future_observed_values, axis=1
        )
        return weighted.mean()


# -----------------------------------------------------------------------------#
# Prediction network                                                            #
# -----------------------------------------------------------------------------#


class AltHierPredictionNetwork(AltHierNetworkBase):
    """Network used for probabilistic forecasting."""

    @validated()
    def __init__(self, *, num_parallel_samples: int = 100, **base_kwargs):
        super().__init__(**base_kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.decoder = HierMLPDecoder(
            bottom_count=self.S_np.shape[1],
            param_dim=self.distr_output.args_dim,
            hidden=base_kwargs["encoder"].config["model_dim"] * 2,
            dropout=base_kwargs["encoder"].config["dropout_rate"],
        )

    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
    ) -> Tensor:
        inputs, scale, _ = self.create_network_input(
            F,
            feat_static_cat,
            past_time_feat,
            past_target,
            past_observed_values,
            None,
            None,
        )

        print("[Pred] enc in :", inputs.shape)
        enc_out = self.encoder(inputs)
        print("[Pred] enc out:", enc_out.shape)

        dec_out = self.decoder(enc_out)
        distr_args = self.proj_dist_args(dec_out)
        distr = self.distr_output.distribution(distr_args, scale=scale)
        samples = distr.sample(num_samples=self.num_parallel_samples)
        print("[Pred] bottom samples:", samples.shape)

        R = np.array(self.R_np, ctx=samples.context)
        coh = np.dot(samples.reshape((-1, R.shape[1])), R.T).reshape(
            samples.shape[:-1] + (R.shape[0],)
        )
        print("[Pred] coherent samples:", coh.shape)
        return coh
