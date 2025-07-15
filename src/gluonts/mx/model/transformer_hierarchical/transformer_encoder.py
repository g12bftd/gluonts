from typing import Dict, Optional

from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential


from gluonts.core.component import validated
from gluonts.mx import Tensor

from gluonts.mx.model.transformer.layers import (
    InputLayer,
    MultiHeadSelfAttention,
    TransformerFeedForward,
    TransformerProcessBlock,
)



class SelfAttentionEncoderLayer(HybridBlock):
    @validated()
    def __init__(self, config: Dict, **kwargs) -> None:
        super().__init__(**kwargs)

        with self.name_scope():

            self.enc_pre_self_att = TransformerProcessBlock(
                sequence=config["pre_seq"],
                dropout=config["dropout_rate"],
                prefix="pretransformerprocessblock_",
            )
            self.enc_self_att = MultiHeadSelfAttention(
                att_dim_in=config["model_dim"],
                heads=config["num_heads"],
                att_dim_out=config["model_dim"],
                dropout=config["dropout_rate"],
                prefix="multiheadselfattention_",
            )
            self.enc_post_self_att = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="postselfatttransformerprocessblock_",
            )
            self.enc_ff = TransformerFeedForward(
                inner_dim=config["model_dim"] * config["inner_ff_dim_scale"],
                out_dim=config["model_dim"],
                act_type=config["act_type"],
                dropout=config["dropout_rate"],
                prefix="transformerfeedforward_",
            )
            self.enc_post_ff = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="postfftransformerprocessblock_",
            )

    def hybrid_forward(
        self,
        F,
        data: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:

        residual = data
        x = self.enc_pre_self_att(data, None)
        x, _ = self.enc_self_att(x, attn_mask)
        x = self.enc_post_self_att(x, residual)

        residual = x
        x = self.enc_ff(x)
        x = self.enc_post_ff(x, residual)
        return x


class AlternatingEncoderLayer(HybridBlock):
    """
    `mode` == "temporal":  self‑attention over timesteps of each series
    `mode` == "spatial":   self‑attention over series at each timestep
    """
    @validated()
    def __init__(self, config: Dict, mode: str, prefix: str = "", **kw):
        assert mode in ("temporal", "spatial")
        super().__init__(prefix=prefix, **kw)
        self.mode = mode
        self.num_series = int(config["num_series"])
        self.model_dim = int(config["model_dim"])

        with self.name_scope():
            self.pre  = TransformerProcessBlock(
                sequence=config["pre_seq"],
                dropout=config["dropout_rate"],
                prefix="pre_",
            )
            self.mha  = MultiHeadSelfAttention(
                att_dim_in=self.model_dim,
                heads=config["num_heads"],
                att_dim_out=self.model_dim,
                dropout=config["dropout_rate"],
                prefix="mha_",
            )
            self.post = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="post_att_",
            )
            self.ff   = TransformerFeedForward(
                inner_dim=self.model_dim * config["inner_ff_dim_scale"],
                out_dim=self.model_dim,
                act_type=config["act_type"],
                dropout=config["dropout_rate"],
                prefix="ff_",
            )
            self.post_ff = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="post_ff_",
            )

    def _time_view(self, F, x: Tensor, reverse: bool = False) -> Tensor:
        """
        B S T D  <->  (B·S) T D   (temporal view)
        """
        if not reverse:
            # B (S·T) D -> B T S D -> (B·S) T D
            x = F.reshape(x, shape=(0, -1, self.num_series, 0))
            x = F.transpose(x, axes=(0, 2, 1, 3))
            x = F.reshape(x, shape=(-1, 0, 0))
        else:
            # (B·S) T D -> B S T D -> B (S·T) D
            x = F.reshape(x, shape=(-1, self.num_series, 0, 0))
            x = F.transpose(x, axes=(0, 2, 1, 3))
            x = F.reshape(x, shape=(0, -1, 0))
        return x

    def _space_view(self, F, x: Tensor, reverse: bool = False) -> Tensor:
        """
        B T S D  <->  (B·T) S D   (spatial view)
        """
        if not reverse:
            # B (S·T) D -> B T S D -> (B·T) S D
            x = F.reshape(x, shape=(0, -1, self.num_series, 0))
            x = F.reshape(x, shape=(-1, self.num_series, 0))
        else:
            # (B·T) S D -> B T S D -> B (S·T) D
            x = F.reshape(x, shape=(-1, 0, self.num_series, 0))
            x = F.reshape(x, shape=(0, -1, 0))
        return x

    def hybrid_forward(self, F, data: Tensor, attn_mask=None) -> Tensor:

        residual = data

        # view‑switch -------------------------------------------------------
        if self.mode == "temporal":
            x = self._time_view(F, data)
        else:  # spatial
            x = self._space_view(F, data)

        # self‑attention ----------------------------------------------------
        x = self.pre(x, None)
        x, _ = self.mha(x, attn_mask)
        x = self.post(x, self.pre(x, None))   # residual inside post‑block

        # back to original layout ------------------------------------------
        if self.mode == "temporal":
            x = self._time_view(F, x, reverse=True)
        else:
            x = self._space_view(F, x, reverse=True)

        # feed‑forward ------------------------------------------------------
        residual = x
        x = self.ff(x)
        x = self.post_ff(x, residual)
        return x




class HierarchicalTransformerEncoder(HybridBlock):
    """
    A stack of encoder blocks that supports two attention schemes:

    Parameters
    ----------
    config
        Dictionary that must contain:
            - "model_dim": embedding size
            - "num_series": (# spatial locations) – **needed for 'alternating'**
            - all keys required by the individual encoder layers
            - "num_encoder_layers"
    attention_scheme : {"full", "alternating"}
        * "full"        – every layer is a full self‑attention block
        * "alternating" – even layers: temporal attention, odd layers: spatial
    """
    @validated()
    def __init__(
        self,
        config: Dict,
        attention_scheme: str = "full",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if attention_scheme not in {"full", "alternating"}:
            raise ValueError(
                f"attention_scheme must be 'full' or 'alternating', "
                f"got '{attention_scheme}'."
            )

        if attention_scheme == "alternating" and "num_series" not in config:
            raise KeyError(
                "'alternating' attention requires config['num_series']."
            )

        self.attention_scheme = attention_scheme
        self.num_encoder_layers = int(config["num_encoder_layers"])

        with self.name_scope():
            self.input_layer = InputLayer(model_size=config["model_dim"])

            self.blocks = HybridSequential()
            for i in range(self.num_encoder_layers):
                if attention_scheme == "full":
                    block = SelfAttentionEncoderLayer(
                        config, prefix=f"full_blk{i}_"
                    )
                else:  # alternating
                    mode = "temporal" if i % 2 == 0 else "spatial"
                    block = AlternatingEncoderLayer(
                        config,
                        mode=mode,
                        prefix=f"{mode}_blk{i}_",
                    )
                self.blocks.add(block)
             self.final_norm = mx.gluon.nn.LayerNorm(axis=-1)

    def hybrid_forward(
        self,
        F,
        data: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        data
            Shape (batch, num_tokens, model_dim) where
            num_tokens = num_series * num_timesteps.
        attn_mask
            Optional causal / padding mask; semantics are preserved even
            in 'alternating' mode because the reshape keeps token order
            within each per‑series / per‑timestep slice.
        """
        x = self.input_layer(data)
        for blk in self.blocks:
            x = blk(x, attn_mask)

        if hasattr(self, "final_norm"):
            x = self.final_norm(x)

        return x 

    
