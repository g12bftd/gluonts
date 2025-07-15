# --------------------------------------------------------------------------
# HierarchicalTransformerDecoder.py
# --------------------------------------------------------------------------
#
# A causal Transformer decoder that can work either
#   • with **full self‑attention** on the flattened token axis, or
#   • with the **alternating temporal / spatial** pattern described earlier.
#
# Each block keeps the canonical Transformer order:
#   masked‑self‑attn → add&norm → enc‑dec‑attn → add&norm → feed‑forward
#
# The only difference between the two modes is how the queries/keys/values
# are *reshaped* before the self‑attention step.  Cross‑attention and the
# feed‑forward network stay untouched.
# --------------------------------------------------------------------------

from typing import Dict, Optional

from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential

from gluonts.core.component import validated
from gluonts.mx import Tensor

from gluonts.mx.model.transformer.layers import (
    InputLayer,
    MultiHeadSelfAttention,
    MultiHeadAttention,
    TransformerFeedForward,
    TransformerProcessBlock,
)

# --------------------------------------------------------------------------
# 1.  Helpers for view switching (flatten  <->  temporal / spatial)
# --------------------------------------------------------------------------

class _ReshapeMixIn:
    """
    Utilities to convert between flat (B, S·T, D) and
    temporal  (B·S, T, D)     or    spatial (B·T, S, D) views.
    """

    def _time_view(self, F, x: Tensor, num_series: int, reverse: bool = False):
        # forward  : (B, S·T, D) -> (B·S, T, D)
        # reverse  : (B·S, T, D) -> (B, S·T, D)
        if not reverse:
            x = F.reshape(x, shape=(0, -1, num_series, 0))       # B, T, S, D
            x = F.transpose(x, axes=(0, 2, 1, 3))               # B, S, T, D
            x = F.reshape(x, shape=(-1, 0, 0))                  # B·S, T, D
        else:
            x = F.reshape(x, shape=(-1, num_series, 0, 0))      # B, S, T, D
            x = F.transpose(x, axes=(0, 2, 1, 3))               # B, T, S, D
            x = F.reshape(x, shape=(0, -1, 0))                  # B, S·T, D
        return x

    def _space_view(self, F, x: Tensor, num_series: int, reverse: bool = False):
        # forward  : (B, S·T, D) -> (B·T, S, D)
        # reverse  : (B·T, S, D) -> (B, S·T, D)
        if not reverse:
            x = F.reshape(x, shape=(0, -1, num_series, 0))      # B, T, S, D
            x = F.reshape(x, shape=(-1, num_series, 0))         # B·T, S, D
        else:
            x = F.reshape(x, shape=(-1, 0, num_series, 0))      # B, T, S, D
            x = F.reshape(x, shape=(0, -1, 0))                  # B, S·T, D
        return x

    # ----- causal mask for temporal view ----------------------------------
    def _causal_mask(self, F, T: int) -> Tensor:
        ones  = F.ones((T, T))
        tril  = F.np.tril(ones) if hasattr(F.np, "tril") else F.linalg.tril(ones)
        return tril  # (T, T)  with 1.0 where j <= i


# --------------------------------------------------------------------------
# 2.  Decoder block variants
# --------------------------------------------------------------------------

class SelfAttentionDecoderLayer(HybridBlock):
    """
    One causal decoder layer with *full* self‑attention on the flattened axis.
    """
    def __init__(self, config: Dict, prefix: str = "", **kwargs):
        super().__init__(prefix=prefix, **kwargs)

        with self.name_scope():
            self.pre  = TransformerProcessBlock(
                sequence=config["pre_seq"],
                dropout=config["dropout_rate"],
                prefix="pre_",
            )
            self.self_att = MultiHeadSelfAttention(
                att_dim_in=config["model_dim"],
                heads=config["num_heads"],
                att_dim_out=config["model_dim"],
                dropout=config["dropout_rate"],
                prefix="mha_",
            )
            self.post_self = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="post_self_",
            )
            self.enc_att = MultiHeadAttention(
                att_dim_in=config["model_dim"],
                heads=config["num_heads"],
                att_dim_out=config["model_dim"],
                dropout=config["dropout_rate"],
                prefix="xatt_",
            )
            self.post_enc = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="post_enc_",
            )
            self.ff  = TransformerFeedForward(
                inner_dim=config["model_dim"] * config["inner_ff_dim_scale"],
                out_dim=config["model_dim"],
                act_type=config["act_type"],
                dropout=config["dropout_rate"],
                prefix="ff_",
            )
            self.post_ff = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="post_ff_",
            )

    # no caching here for brevity; GluonTS will still work in training
    def hybrid_forward(self, F, data, enc_out, mask=None, is_train=True):
        residual = data
        x_pre = self.pre(data, None)
        x, _  = self.self_att(x_pre, mask, None)
        x     = self.post_self(x, residual)

        residual = x
        x_att   = self.enc_att(x, enc_out)
        x       = self.post_enc(x_att, residual)

        residual = x
        x_ff     = self.ff(x)
        x        = self.post_ff(x_ff, residual)
        return x


class AlternatingDecoderLayer(_ReshapeMixIn, HybridBlock):
    """
    Decoder layer that performs **temporal** or **spatial** self‑attention,
    then the usual encoder–decoder attention, then the feed‑forward block.
    """
    def __init__(self, config: Dict, mode: str, prefix: str = "", **kwargs):
        assert mode in ("temporal", "spatial")
        super().__init__(prefix=prefix, **kwargs)

        self.mode = mode
        self.num_series = int(config["num_series"])

        with self.name_scope():
            self.pre  = TransformerProcessBlock(
                sequence=config["pre_seq"],
                dropout=config["dropout_rate"],
                prefix="pre_",
            )
            self.self_att = MultiHeadSelfAttention(
                att_dim_in=config["model_dim"],
                heads=config["num_heads"],
                att_dim_out=config["model_dim"],
                dropout=config["dropout_rate"],
                prefix="mha_",
            )
            self.post_self = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="post_self_",
            )
            self.enc_att = MultiHeadAttention(
                att_dim_in=config["model_dim"],
                heads=config["num_heads"],
                att_dim_out=config["model_dim"],
                dropout=config["dropout_rate"],
                prefix="xatt_",
            )
            self.post_enc = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="post_enc_",
            )
            self.ff  = TransformerFeedForward(
                inner_dim=config["model_dim"] * config["inner_ff_dim_scale"],
                out_dim=config["model_dim"],
                act_type=config["act_type"],
                dropout=config["dropout_rate"],
                prefix="ff_",
            )
            self.post_ff = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="post_ff_",
            )

    # ------------------------------------------------------------------ #
    def _reshape_qkv(
        self,
        F,
        q: Tensor,
        kv: Tensor,
    ):
        """
        Returns   q_reshaped, kv_reshaped, causal_mask_or_None
        The shapes are:
            temporal :  (B·S, T_q,  D)   , (B·S, T_kv, D) ,  (T_q, T_q)
            spatial  :  (B·T, S_q,  D)   , (B·T, S_kv, D) ,  None
        """
        if self.mode == "temporal":
            q_  = self._time_view(F, q,  self.num_series, reverse=False)
            kv_ = self._time_view(F, kv, self.num_series, reverse=False)
            T   = F.shape_array(q_)[1]
            mask = self._causal_mask(F, int(T.asscalar()))  # (T, T)
            return q_, kv_, mask
        else:  # spatial
            q_  = self._space_view(F, q,  self.num_series, reverse=False)
            kv_ = self._space_view(F, kv, self.num_series, reverse=False)
            return q_, kv_, None

    # ------------------------------------------------------------------ #
    def hybrid_forward(
        self,
        F,
        data: Tensor,
        enc_out: Tensor,
        mask: Optional[Tensor] = None,
        is_train: bool = True,
    ) -> Tensor:

        # -------------------- Self‑attention ----------------------------
        q, kv, causal = self._reshape_qkv(F, data, data)
        residual = q
        x_pre    = self.pre(q, None)
        x, _     = self.self_att(x_pre, causal, None)
        x        = self.post_self(x, residual)

        # back to flat view
        if self.mode == "temporal":
            x = self._time_view(F, x, self.num_series, reverse=True)
        else:
            x = self._space_view(F, x, self.num_series, reverse=True)

        # -------------------- Cross‑attention ---------------------------
        # reshape enc_out to match the *current* view
        q2, kv2, _ = self._reshape_qkv(F, x, enc_out)
        residual = q2
        x_att   = self.enc_att(q2, kv2)
        x       = self.post_enc(x_att, residual)

        # feed‑forward (flat view)
        residual = x
        x_ff     = self.ff(x)
        x        = self.post_ff(x_ff, residual)
        return x


# --------------------------------------------------------------------------
# 3.  HierarchicalTransformerDecoder (public class)
# --------------------------------------------------------------------------

class HierarchicalTransformerDecoder(HybridBlock):
    """
    A stack of `num_decoder_layers` causal decoder blocks.

    Parameters
    ----------
    num_decoder_layers
        How many layers to stack.
    config
        Must contain at least:
            - "model_dim", "pre_seq", "post_seq", "inner_ff_dim_scale",
              "num_heads", "dropout_rate"
            - "num_series"                (required if attention_scheme="alternating")
    attention_scheme
        "full"         – every layer uses flattened self‑attention
        "alternating"  – even layers: temporal; odd layers: spatial
    """
    @validated()
    def __init__(
        self,
        num_decoder_layers: int,
        config: Dict,
        attention_scheme: str = "full",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if attention_scheme not in {"full", "alternating"}:
            raise ValueError(
                "attention_scheme must be 'full' or 'alternating', "
                f"got '{attention_scheme}'."
            )
        if attention_scheme == "alternating" and "num_series" not in config:
            raise KeyError("'num_series' must be in config for alternating mode.")

        self.attention_scheme = attention_scheme
        self.num_layers = num_decoder_layers

        with self.name_scope():
            self.input_layer = InputLayer(model_size=config["model_dim"])

            self.blocks = HybridSequential()
            for i in range(num_decoder_layers):
                if attention_scheme == "full":
                    blk = SelfAttentionDecoderLayer(
                        config, prefix=f"dec_full_{i}_"
                    )
                else:
                    mode = "temporal" if i % 2 == 0 else "spatial"
                    blk = AlternatingDecoderLayer(
                        config,
                        mode=mode,
                        prefix=f"dec_{mode}_{i}_",
                    )
                self.blocks.add(blk)

    # ------------------------------------------------------------------ #
    def cache_reset(self):
        #  placeholder – extend if you later add per‑layer caches
        pass

    # ------------------------------------------------------------------ #
    def hybrid_forward(
        self,
        F,
        data: Tensor,               # (B, pred_len · S, D)  after flattening
        encoder_output: Tensor,     # (B, ctx_len · S, D)
        mask: Optional[Tensor] = None,  # triangular on flat axis
        is_train: bool = True,
    ) -> Tensor:

        x = self.input_layer(data)
        for blk in self.blocks:
            x = blk(x, encoder_output, mask, is_train=is_train)
        return x
