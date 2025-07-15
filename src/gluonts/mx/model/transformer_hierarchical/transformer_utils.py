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



class HierarchicalTransformerEncoder(HybridBlock):
    def __init__(self, num_layers: int, config: Dict, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.enc_input_layer = InputLayer(model_size=config["model_dim"])
            self.blocks = HybridSequential()
            for i in range(num_layers):
                self.blocks.add(
                    SelfAttentionEncoderLayer(config, prefix=f"blk{i}_")
                )

    def hybrid_forward(
        self,
        F,
        data: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:

        x = self.enc_input_layer(data)
        for blk in self.blocks:
            x = blk(x, attn_mask)

        # Add final norm if needed here

        return x  



class HierarchicalTransformerDecoder(HybridBlock):
    def __init__(self, num_decoder_layers: int, config: Dict, **kwargs) -> None:
        super().__init__(**kwargs)

    def cache_reset(self):
        self.cache = {}
    
    def hybrid_forward(
        self, 
        F, 
        data: Tensor,
        encoder_output: Tensor,
        mask: Optional[Tensor],
        is_train: bool = True,
        ) -> Tensor:

        return data

    
