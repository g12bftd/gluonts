# mlp_decoder.py
from mxnet.gluon import HybridBlock, nn
from gluonts.mx import Tensor

class HierMLPDecoder(HybridBlock):
    """
    Fan-out MLP that turns the encoder's hidden states into distribution
    parameters for **bottom-level** series.

    Expects input of shape   (B, T_pred, d_model)
    Returns tensor           (B, T_pred, BOTTOM, param_dim)
    """
    def __init__(self, bottom_count: int, param_dim: int, hidden: int = 256,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.mlp = nn.HybridSequential()
            self.mlp.add(
                nn.Dense(hidden, activation="relu"),
                nn.Dropout(dropout),
                nn.Dense(bottom_count * param_dim)
            )
        self.bottom = bottom_count
        self.param_dim = param_dim

    def hybrid_forward(self, F, x: Tensor) -> Tensor:
        B, T, D = x.shape
        y = self.mlp(x)                            # (B, T, bottom*param)
        y = y.reshape((B, T, self.bottom, self.param_dim))
        return y
