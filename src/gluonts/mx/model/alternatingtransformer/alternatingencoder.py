# Copyright 2025 ...
#
# Alternating-attention encoder for GluonTS/MXNet.
#
# A **temporal step**  attends along the time axis  (shape  B·S × T × d)
# A **spatial  step**  attends along the series axis (shape  B·T × S × d)
# The two steps alternate layer-by-layer:
#   layer-0 → temporal  • layer-1 → spatial • layer-2 → temporal • …

from typing import Dict, List

from mxnet.gluon import HybridBlock, nn

from gluonts.mx import Tensor
from gluonts.core.component import validated

from .layers import (
    InputLayer,
    MultiHeadSelfAttention,
    TransformerFeedForward,
    TransformerProcessBlock,
)


class _AltBlock(HybridBlock):
    """
    ONE transformer layer that can operate either temporally or spatially.

    Parameters
    ----------
    axis : {'temporal', 'spatial'}
        Which axis the self-attention mixes.
    cfg  : Dict
        Same hyper-parameter dict the vanilla encoder receives.
    """

    def __init__(self, axis: str, cfg: Dict, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

        with self.name_scope():
            self.pre  = TransformerProcessBlock(
                sequence=cfg["pre_seq"],
                dropout=cfg["dropout_rate"],
                prefix=f"pre_{axis}_",
            )
            self.attn = MultiHeadSelfAttention(
                att_dim_in = cfg["model_dim"],
                heads      = cfg["num_heads"],
                att_dim_out= cfg["model_dim"],
                dropout    = cfg["dropout_rate"],
                prefix     = f"msa_{axis}_",
            )
            self.post_attn = TransformerProcessBlock(
                sequence=cfg["post_seq"],
                dropout=cfg["dropout_rate"],
                prefix=f"post_attn_{axis}_",
            )
            self.ff = TransformerFeedForward(
                inner_dim = cfg["model_dim"] * cfg["inner_ff_dim_scale"],
                out_dim   = cfg["model_dim"],
                act_type  = cfg["act_type"],
                dropout   = cfg["dropout_rate"],
                prefix    = f"ff_{axis}_",
            )
            self.post_ff = TransformerProcessBlock(
                sequence=cfg["post_seq"],
                dropout=cfg["dropout_rate"],
                prefix=f"post_ff_{axis}_",
            )

    def hybrid_forward(self, F, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, T, S, d)
        """
        B, T, S, D = x.shape

        if self.axis == "temporal":
            # (B, T, S, d) -> (B*S, T, d)
            y = x.reshape((B * S, T, D))
            y, _ = self.attn(self.pre(y, None))
            y = y.reshape((B, S, T, D)).swapaxes(1, 2)  # -> back to (B,T,S,d)
        else:  # spatial
            # (B, T, S, d) -> (B*T, S, d)
            y = x.swapaxes(1, 2).reshape((B * T, S, D))
            y, _ = self.attn(self.pre(y, None))
            y = y.reshape((B, T, S, D))                 # already (B,T,S,d)

        x = self.post_attn(y, x)        # residual + norm/drop
        y  = self.ff(x)
        x  = self.post_ff(y, x)
        return x


class AlternatingTransformerEncoder(HybridBlock):
    """
    Encoder with layer-by-layer alternation between **temporal** and **spatial**
    self-attention.

    Parameters
    ----------
    num_timesteps    : int
        Length of the (past) temporal dimension fed to the encoder.
    num_series       : int
        Number of parallel time-series (spatial dimension).
    num_layers       : int, default 4
        Total transformer layers (odd layers are spatial).
    config           : Dict
        Same hyper-parameter dict used for the vanilla TransformerEncoder.
    """

    @validated()
    def __init__(
        self,
        num_timesteps: int,
        num_series: int,
        config: Dict,
        num_layers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.T, self.S = num_timesteps, num_series
        self.layers: List[_AltBlock] = []

        with self.name_scope():
            self.input_layer = InputLayer(model_size=config["model_dim"])

            self.alt_layers = nn.HybridSequential()
            for i in range(num_layers):
                axis = "temporal" if i % 2 == 0 else "spatial"
                self.alt_layers.add(_AltBlock(axis, config, prefix=f"layer{i}_"))

    # --------------------------------------------------------------------- #
    # x expected shape: (B, T, S, d_model)
    # --------------------------------------------------------------------- #
    def hybrid_forward(self, F, data: Tensor) -> Tensor:
        print(f"x shape: {x.shape}")
        x = self.input_layer(data)         # still (B,T,S,d)
        for layer in self.alt_layers:
            x = layer(x)
        return x

