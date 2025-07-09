# -----------------------------------------------------------------------------#
# Alternating-Attention Transformer – deterministic, hierarchy-coherent        #
# -----------------------------------------------------------------------------#
from typing import List, Optional, Tuple

import numpy as np
import mxnet as mx
from mxnet.gluon import HybridBlock, nn
from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.util import weighted_average
from gluonts.core.component import validated
from gluonts.itertools import prod

from .alternating_encoder import AlternatingTransformerEncoder

# -----------------------------------------------------------------------------#
# helpers                                                                       #
# -----------------------------------------------------------------------------#
def projection_matrix(S: np.ndarray) -> np.ndarray:
    """OLS reconciliation matrix  P = S (Sᵀ S)⁻¹ Sᵀ ."""
    St = S.T
    return (S @ np.linalg.inv(St @ S) @ St).astype("float32")

# simple dense decoder ---------------------------------------------------------#
class HierMLPDecoder(HybridBlock):
    """
    Map encoder output to bottom-level forecasts.
    Output: (B, bottom_count, prediction_length)
    """
    def __init__(self, bottom_count: int, pred_len: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        out_units = bottom_count * pred_len
        with self.name_scope():
            self.net = nn.HybridSequential()
            self.net.add(
                nn.Dense(hidden, activation="relu"),
                nn.Dropout(dropout),
                nn.Dense(out_units),
            )
        self.bottom = bottom_count
        self.pred_len = pred_len

    def hybrid_forward(self, F, x):  # x: (B, T, bottom, d)
        b = x.shape[0]
        y = self.net(x.reshape((b, -1)))               # (B, bottom*pred)
        return y.reshape((b, self.bottom, self.pred_len))

# -----------------------------------------------------------------------------#
class AltHierNetworkBase(HybridBlock):
    """Shared utilities for training & prediction."""
    @validated()
    def __init__(
        self,
        *,
        encoder: AlternatingTransformerEncoder,
        history_length: int,
        context_length: int,
        prediction_length: int,
        S: np.ndarray,                          # (M × B)
        cardinality: List[int],
        embedding_dimension: int,
        lags_seq: List[int],
        reconciliation_method: str = "ols",     # "ols" | "bottom_up"
        scaling: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert reconciliation_method in {"ols", "bottom_up"}

        self.encoder = encoder
        self.history_length = history_length
        self.context_length = context_length
        self.pred_len = prediction_length
        self.lags_seq = sorted(lags_seq)
        self.scaling = scaling

        self.S_np = S.astype("float32")
        self.R_np = projection_matrix(S) if reconciliation_method == "ols" else self.S_np

        with self.name_scope():
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=[embedding_dimension] * len(cardinality),
            )
            self.scaler = MeanScaler(keepdims=True) if scaling else NOPScaler(keepdims=True)

    # lag helper ----------------------------------------------------------------
    @staticmethod
    def get_lagged_subsequences(F, seq: Tensor, seq_len: int, idx: List[int], sub_len: int):
        assert max(idx) + sub_len <= seq_len
        out = [F.slice_axis(seq, 1, -lag - sub_len, None if lag == 0 else -lag) for lag in idx]
        return F.stack(*out, axis=-1)

# -----------------------------------------------------------------------------#
class AltHierTrainingNetwork(AltHierNetworkBase):
    """Returns MSE loss."""
    @validated()
    def __init__(self, *, decoder: HierMLPDecoder, **base):
        super().__init__(**base)
        self.decoder = decoder

    # input builder -------------------------------------------------------------
    def _make_input(
        self,
        F,
        feat_static_cat,
        past_time_feat,
        past_target,
        past_observed,
        future_time_feat,
        future_target,
    ) -> Tuple[Tensor, Tensor]:
        if future_time_feat is None:
            time_feat = past_time_feat.slice_axis(axis=1, begin=self.history_length - self.context_length, end=None)
            sequence, seq_len, sub_len = past_target, self.history_length, self.context_length
        else:
            time_feat = F.concat(
                past_time_feat.slice_axis(axis=1, begin=self.history_length - self.context_length, end=None),
                future_time_feat,
                dim=1,
            )
            sequence = F.concat(past_target, future_target, dim=1)
            seq_len = self.history_length + self.pred_len
            sub_len = self.context_length + self.pred_len

        lags = self.get_lagged_subsequences(F, sequence, seq_len, self.lags_seq, sub_len)
        _, scale = self.scaler(
            past_target.slice_axis(axis=1, begin=-self.context_length, end=None),
            past_observed.slice_axis(axis=1, begin=-self.context_length, end=None),
        )
        emb = self.embedder(feat_static_cat)
        static = F.concat(emb, F.log(scale.squeeze(axis=1)), dim=1)
        rep_static = static.expand_dims(1).repeat(axis=1, repeats=sub_len)

        lags_scaled = F.broadcast_div(lags, scale.expand_dims(-1))
        input_lags = lags_scaled.reshape((-1, sub_len, len(self.lags_seq)))
        inputs = F.concat(input_lags, time_feat, rep_static, dim=-1).expand_dims(axis=2)
        return inputs, scale

    # forward -------------------------------------------------------------------
    def hybrid_forward(
        self,
        F,
        feat_static_cat,
        past_time_feat,
        past_target,
        past_observed_values,
        future_time_feat,
        future_target,
        future_observed_values,
    ):
        x, scale = self._make_input(
            F,
            feat_static_cat,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )

        print("[Train] encoder in :", x.shape)
        enc_out = self.encoder(x)
        print("[Train] encoder out:", enc_out.shape)

        pred_bottom_scaled = self.decoder(enc_out)                 # (B, BOTTOM, pred)
        pred_bottom = pred_bottom_scaled * scale                   # rescale
        print("[Train] bottom preds:", pred_bottom.shape)

        # MSE on bottom level -----------------------------------------------
        mse = F.square(pred_bottom.swapaxes(1, 2) - future_target)  # align dims (B, pred, BOTTOM)
        loss = weighted_average(F, mse, future_observed_values, axis=1)
        return loss.mean()

# -----------------------------------------------------------------------------#
class AltHierPredictionNetwork(AltHierNetworkBase):
    """Outputs coherent forecasts (B, M, pred_len)."""
    @validated()
    def __init__(self, *, decoder: HierMLPDecoder, **base):
        super().__init__(**base)
        self.decoder = decoder

    def hybrid_forward(
        self,
        F,
        feat_static_cat,
        past_time_feat,
        past_target,
        past_observed_values,
        future_time_feat,
    ):
        x, scale = self._make_input(
            F,
            feat_static_cat,
            past_time_feat,
            past_target,
            past_observed_values,
            None,
            None,
        )
        enc = self.encoder(x)
        bottom = self.decoder(enc) * scale                         # (B, BOTTOM, pred)
        B, BOTTOM, P = bottom.shape
        R = mx.nd.array(self.R_np, ctx=bottom.context)             # (M, BOTTOM)

        # reconcile per horizon step -----------------------------------------
        flat = bottom.transpose((0, 2, 1)).reshape((-1, BOTTOM))   # (B*P, BOTTOM)
        all_nodes = mx.nd.dot(flat, R.T).reshape((B, P, -1))       # (B, P, M)
        coherent = all_nodes.transpose((0, 2, 1))                  # (B, M, P)

        print("[Pred] coherent shape:", coherent.shape)
        return coherent
