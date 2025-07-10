# gluonts/mx/model/alt_hier_transformer/_estimator.py
"""
Public estimator that plugs the alternating encoder into GluonTS
and reuses DeepVAR-Hierarchical reconciliation utilities.
"""
from typing import List, Optional

import mxnet as mx
import numpy as np
from gluonts.core.component import validated
from gluonts.mx.trainer import Trainer
from gluonts.mx.model.transformer._estimator import TransformerEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.model.deepvar_hierarchical._estimator import projection_mat
from gluonts.mx.model.transformer.trans_decoder import TransformerDecoder
from mxnet.gluon import nn

from .transencoder_alt import AlternatingHierEncoder
from ._network import (
    AltHierTransformerTrainingNetwork,
    AltHierTransformerPredictionNetwork,
)


class AlternatingHierarchicalTransformerEstimator(TransformerEstimator):
    """
    Transformer with alternating spatial / temporal encoder
    and DeepVAR-Hierarchical coherence.
    """

    @validated()
    def __init__(
        self,
        S,                          # (D_total, D_bottom) hierarchy matrix
        D: Optional[np.ndarray] = None,         # If you use weights
        encoder_layers: int = 6,
        model_dim: int = 32,
        num_heads: int = 8,
        inner_ff_dim_scale: int = 4,
        dropout_rate: float = 0.1,
        act_type: str = "gelu",
        pre_seq: str = "dn",
        post_seq: str = "drn",
        num_samples_for_loss: int = 200,
        likelihood_weight: float = 0.0,
        CRPS_weight: float = 1.0,
        coherent_pred_samples: bool = True,
        trainer: Trainer = Trainer(),
        **kwargs,
    ):
        assert (
            encoder_layers % 2 == 0
        ), "encoder_layers must be even for alternating attention"

        self.S = mx.nd.array(S, ctx=trainer.ctx)
        self.M = mx.nd.array(projection_mat(S, D), ctx=trainer.ctx)
        self.num_series = S.shape[0]

        # build encoder / decoder / projection head
        encoder = AlternatingHierEncoder(
            num_layers=encoder_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            inner_ff_dim_scale=inner_ff_dim_scale,
            dropout_rate=dropout_rate,
            act_type=act_type,
            pre_seq=pre_seq,
            post_seq=post_seq,
        )
        decoder = TransformerDecoder(
            model_dim=model_dim,
            num_heads=num_heads,
            inner_ff_dim_scale=inner_ff_dim_scale,
            dropout_rate=dropout_rate,
            act_type=act_type,
            pre_seq=pre_seq,
            post_seq=post_seq,
        )
        proj_head = nn.Dense(
            units=3, flatten=False, in_units=model_dim
        )  # Student-T μ, σ, ν

        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            inner_ff_dim_scale=inner_ff_dim_scale,
            dropout_rate=dropout_rate,
            act_type=act_type,
            trainer=trainer,
            **kwargs,
        )

        # store modules
        self.encoder = encoder
        self.decoder = decoder
        self.proj_head = proj_head

        # loss / pred settings
        self.num_samples_for_loss = num_samples_for_loss
        self.likelihood_weight = likelihood_weight
        self.CRPS_weight = CRPS_weight
        self.coherent_pred_samples = coherent_pred_samples

    # ------------------------------------------------------------------
    #  overrides
    # ------------------------------------------------------------------
    def create_training_network(self) -> AltHierTransformerTrainingNetwork:
        return AltHierTransformerTrainingNetwork(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            freq=self.freq,
            lags_seq=self.lags_seq,
            num_series=self.num_series,
            model_dim=self.model_dim,
            encoder=self.encoder,
            decoder=self.decoder,
            proj=self.proj_head,
            M=self.M,
            S=self.S,
            num_samples_for_loss=self.num_samples_for_loss,
            likelihood_weight=self.likelihood_weight,
            CRPS_weight=self.CRPS_weight,
            cardinality=self.cardinality,
        )

    def create_predictor(
        self, transformation, trained_network
    ) -> RepresentableBlockPredictor:
        pred_net = AltHierTransformerPredictionNetwork(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            freq=self.freq,
            lags_seq=self.lags_seq,
            num_series=self.num_series,
            model_dim=self.model_dim,
            encoder=self.encoder,
            decoder=self.decoder,
            proj=self.proj_head,
            M=self.M,
            S=self.S,
            coherent_pred_samples=self.coherent_pred_samples,
            cardinality=self.cardinality,
        )
        self.copy_parameters(trained_network, pred_net)
        return RepresentableBlockPredictor(
            input_transform=transformation + self._create_instance_splitter("test"),
            prediction_net=pred_net,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
