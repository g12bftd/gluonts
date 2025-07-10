# gluonts/mx/model/alt_hier_transformer/transencoder_alt.py
"""
Alternating spatial / temporal encoder for hierarchical time-series.

Input / output tensor shape: (B, L, D, H)
B = batch, L = sequence length, D = number of parallel series, H = model_dim
"""
from typing import Dict

import mxnet as mx
from mxnet.gluon import HybridBlock, nn
from gluonts.core.component import validated
from gluonts.mx.model.transformer.layers import (
    MultiHeadSelfAttention,
    TransformerFeedForward,
    TransformerProcessBlock,
)

class _AltEncoderLayer(HybridBlock):
    """
    One encoder block that runs **spatial** or **temporal** self-attention
    depending on `mode`.
    """
    @validated()
    def __init__(
        self,
        mode: str,
        model_dim: int,
        num_heads: int,
        inner_ff_dim_scale: int = 4,
        dropout_rate: float = 0.1,
        act_type: str = "gelu",
        pre_seq: str = "dn",
        post_seq: str = "drn",
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(prefix=prefix, **kwargs)
        assert mode in {"spatial", "temporal"}
        self.mode = mode
        with self.name_scope():
            self.pre = TransformerProcessBlock(
                sequence=pre_seq, dropout=dropout_rate, prefix=f"{mode}_pre_"
            )
            self.mha = MultiHeadSelfAttention(
                att_dim_in=model_dim,
                att_dim_out=model_dim,
                heads=num_heads,
                dropout=dropout_rate,
                prefix=f"{mode}_att_",
            )
            self.post = TransformerProcessBlock(
                sequence=post_seq, dropout=dropout_rate, prefix=f"{mode}_post_"
            )
            self.ff = TransformerFeedForward(
                inner_dim=model_dim * inner_ff_dim_scale,
                out_dim=model_dim,
                act_type=act_type,
                dropout=dropout_rate,
                prefix=f"{mode}_ff_",
            )
            self.post_ff = TransformerProcessBlock(
                sequence=post_seq, dropout=dropout_rate, prefix=f"{mode}_postff_"
            )

    def _spatial(self, F, x):
        # (B, L, D, H)  →  (B·L, D, H)
        B, L, D, H = x.shape
        y, _ = self.mha(self.pre(x, None).reshape((-1, D, H)))
        y = y.reshape((B, L, D, H))
        return self.post(y, x)

    def _temporal(self, F, x):
        # (B, L, D, H) → transpose → (B·D, L, H)
        B, L, D, H = x.shape
        xt = x.transpose((0, 2, 1, 3))          # B, D, L, H
        y, _ = self.mha(self.pre(xt, None).reshape((-1, L, H)))
        y = y.reshape((B, D, L, H)).transpose((0, 2, 1, 3))
        return self.post(y, x)

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        if self.mode == "spatial":
            x = self._spatial(F, x)
        else:
            x = self._temporal(F, x)
        return self.post_ff(self.ff(x), x)


class AlternatingHierEncoder(HybridBlock):
    """
    Stack of `num_layers` alternating spatial / temporal blocks.
    """
    @validated()
    def __init__(self, num_layers: int, **layer_kwargs):
        super().__init__()
        assert num_layers % 2 == 0, "Need an EVEN number of layers."
        with self.name_scope():
            self.blocks = nn.HybridSequential()
            for i in range(num_layers):
                mode = "spatial" if i % 2 == 0 else "temporal"
                self.blocks.add(
                    _AltEncoderLayer(mode=mode, prefix=f"{mode}_{i}_", **layer_kwargs)
                )

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        for blk in self.blocks:
            x = blk(x)
        return x
