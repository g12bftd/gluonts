# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Building blocks for the Alternating Transformer model."""

from typing import Dict

import mxnet as mx
from mxnet.gluon import HybridBlock

from gluonts.mx import Tensor
from gluonts.mx.model.transformer.layers import (
    MultiHeadSelfAttention,
    TransformerFeedForward,
    TransformerProcessBlock,
)


class AlternatingTransformerLayer(HybridBlock):
    """Single layer of alternating temporal and spatial attention."""

    def __init__(self, config: Dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_dim = config["model_dim"]
        with self.name_scope():
            self.temporal_pre = TransformerProcessBlock(
                sequence=config["pre_seq"], dropout=config["dropout_rate"]
            )
            self.temporal_att = MultiHeadSelfAttention(
                att_dim_in=self.model_dim,
                heads=config["num_heads"],
                att_dim_out=self.model_dim,
                dropout=config["dropout_rate"],
            )
            self.temporal_post = TransformerProcessBlock(
                sequence=config["post_seq"], dropout=config["dropout_rate"]
            )

            self.spatial_pre = TransformerProcessBlock(
                sequence=config["pre_seq"], dropout=config["dropout_rate"]
            )
            self.spatial_att = MultiHeadSelfAttention(
                att_dim_in=self.model_dim,
                heads=config["num_heads"],
                att_dim_out=self.model_dim,
                dropout=config["dropout_rate"],
            )
            self.spatial_post = TransformerProcessBlock(
                sequence=config["post_seq"], dropout=config["dropout_rate"]
            )

            self.ff = TransformerFeedForward(
                inner_dim=self.model_dim * config["inner_ff_dim_scale"],
                out_dim=self.model_dim,
                act_type=config["act_type"],
                dropout=config["dropout_rate"],
            )
            self.post_ff = TransformerProcessBlock(
                sequence=config["post_seq"], dropout=config["dropout_rate"]
            )

    def hybrid_forward(self, F, data: Tensor) -> Tensor:
        """Apply temporal then spatial self-attention."""
        # data: (batch, num_series, num_timesteps, model_dim)
        residual = data

        # temporal attention
        tmp = F.reshape(data, shape=(-3, -1, self.model_dim))
        tmp, _ = self.temporal_att(self.temporal_pre(tmp, None))
        tmp = F.reshape(tmp, shape=(-4, -1, 0, self.model_dim))
        data = self.temporal_post(tmp, residual)
        residual = data

        # spatial attention
        tmp = F.transpose(data, axes=(0, 2, 1, 3))
        tmp = F.reshape(tmp, shape=(-3, -1, self.model_dim))
        tmp, _ = self.spatial_att(self.spatial_pre(tmp, None))
        tmp = F.reshape(tmp, shape=(-4, -1, 0, self.model_dim))
        tmp = F.transpose(tmp, axes=(0, 2, 1, 3))
        data = self.spatial_post(tmp, residual)

        # feed forward
        ff = self.ff(data)
        return self.post_ff(ff, data)
