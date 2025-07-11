
import math

import mxnet as mx
from mxnet import nd
from mxnet.gluon import HybridBlock, nn

from gluonts.mx import Tensor

def _get_activation_fn(F, activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError(f"Invalid activation: '{activation}'")

def generate_square_subsequent_mask(F, sz: int) -> Tensor:
    mask = F.triu(F.ones((sz, sz)), k=1)
    mask = F.transpose(mask == 0, axes=(0, 1))
    mask = F.where(mask, F.zeros_like(mask), F.full(shape=(sz, sz), val=float('-inf')))
    return mask

class PositionalEncoding(HybridBlock):
    def __init__(self, d_model: int, max_len: int = 5000, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        position = nd.arange(0, max_len, ctx=mx.cpu()).expand_dims(1)
        div_term = nd.exp(nd.arange(0, d_model, 2, ctx=mx.cpu()) * -(math.log(10000.0) / d_model))
        pe = nd.zeros((max_len, d_model), ctx=mx.cpu())
        pe[:, 0::2] = nd.sin(position * div_term)
        pe[:, 1::2] = nd.cos(position * div_term)
        self.pe = pe.expand_dims(1)

    def hybrid_forward(self, F, x: Tensor) -> Tensor:
        return x + self.pe[:x.shape[0]]

class AlternatingAttentionLayer(HybridBlock):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str, is_spatial: bool, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self.is_spatial = is_spatial
        self.self_attn = nn.MultiHeadAttention(units=d_model, num_heads=nhead, dropout=dropout)
        self.linear1 = nn.Dense(units=dim_feedforward, in_units=d_model, flatten=False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Dense(units=d_model, in_units=dim_feedforward, flatten=False)
        self.norm1 = nn.LayerNorm(in_channels=d_model)
        self.norm2 = nn.LayerNorm(in_channels=d_model)
        self.activation_fn = activation

    def hybrid_forward(self, F, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        b, t, s, d = x.shape
        if self.is_spatial:
            x = F.reshape(x, shape=(b * t, s, d))
            if attn_mask is not None:
                attn_mask = F.broadcast_to(attn_mask, shape=(b * t, -1, -1))
        else:
            x = F.swapaxes(x, 1, 2)
            x = F.reshape(x, shape=(b * s, t, d))
        residual = x
        x = self.norm1(x)
        x2 = self.self_attn(x, x, x, mask=attn_mask)[0]
        x = residual + self.dropout(x2)
        residual = x
        x = self.norm2(x)
        x2 = self.linear1(x)
        x2 = _get_activation_fn(F, self.activation_fn)(x2)
        x2 = self.dropout(x2)
        x2 = self.linear2(x2)
        x = residual + self.dropout(x2)
        if self.is_spatial:
            x = F.reshape(x, shape=(b, t, s, d))
        else:
            x = F.reshape(x, shape=(b, s, t, d))
            x = F.swapaxes(x, 1, 2)
        return x

class AlternatingDecoderLayer(HybridBlock):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str, is_spatial: bool, context_length: int, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self.is_spatial = is_spatial
        self.self_attn = nn.MultiHeadAttention(units=d_model, num_heads=nhead, dropout=dropout)
        self.cross_attn = nn.MultiHeadAttention(units=d_model, num_heads=nhead, dropout=dropout)
        self.linear1 = nn.Dense(units=dim_feedforward, in_units=d_model, flatten=False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Dense(units=d_model, in_units=dim_feedforward, flatten=False)
        self.norm1 = nn.LayerNorm(in_channels=d_model)
        self.norm2 = nn.LayerNorm(in_channels=d_model)
        self.norm3 = nn.LayerNorm(in_channels=d_model)
        self.activation_fn = activation
        self.context_length = context_length

    def hybrid_forward(self, F, x: Tensor, memory: Tensor, self_att_mask: Optional[Tensor] = None, cross_att_mask: Optional[Tensor] = None) -> Tensor:
        b, p, s, d = x.shape
        if self.is_spatial:
            x = F.reshape(x, shape=(b * p, s, d))
            memory = F.reshape(memory, shape=(b * self.context_length, s, d))
        else:
            x = F.swapaxes(x, 1, 2)
            x = F.reshape(x, shape=(b * s, p, d))
            memory = F.swapaxes(memory, 1, 2)
            memory = F.reshape(memory, shape=(b * s, self.context_length, d))
            if self_att_mask is None:
                self_att_mask = generate_square_subsequent_mask(F, p)
        residual = x
        x = self.norm1(x)
        x2 = self.self_attn(x, x, x, mask=self_att_mask)[0]
        x = residual + self.dropout(x2)
        residual = x
        x = self.norm2(x)
        x2 = self.cross_attn(x, memory, memory, mask=cross_att_mask)[0]
        x = residual + self.dropout(x2)
        residual = x
        x = self.norm3(x)
        x2 = self.linear1(x)
        x2 = _get_activation_fn(F, self.activation_fn)(x2)
        x2 = self.dropout(x2)
        x2 = self.linear2(x2)
        x = residual + self.dropout(x2)
        if self.is_spatial:
            x = F.reshape(x, shape=(b, p, s, d))
        else:
            x = F.reshape(x, shape=(b, s, p, d))
            x = F.swapaxes(x, 1, 2)
        return x
