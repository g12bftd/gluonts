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

"""Networks used by the Alternating Transformer model."""

from typing import Dict
from itertools import product

import mxnet as mx
from mxnet.gluon import HybridBlock, nn

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.distribution import EmpiricalDistribution

from .layers import AlternatingTransformerLayer


class AlternatingTransformerNetwork(HybridBlock):
    """Stack of alternating transformer layers producing hidden features."""

    @validated()
    def __init__(
        self,
        num_layers: int,
        num_series: int,
        num_timesteps: int,
        config: Dict,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_series = num_series
        self.num_timesteps = num_timesteps
        self.model_dim = config["model_dim"]
        self.debug = debug

        with self.name_scope():
            self.spatial_token = self.params.get(
                "spatial_token",
                shape=(num_series, self.model_dim),
                init=mx.init.Normal(),
            )
            self.temporal_token = self.params.get(
                "temporal_token",
                shape=(num_timesteps, self.model_dim),
                init=mx.init.Normal(),
            )
            self.input_proj = nn.Dense(units=self.model_dim, flatten=False)
            self.layers = nn.HybridSequential()
            for _ in range(num_layers):
                self.layers.add(AlternatingTransformerLayer(config))

    def hybrid_forward(
        self, F, data: Tensor, spatial_token: Tensor, temporal_token: Tensor
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        data
            Input tensor of shape (batch, num_series, num_timesteps).
        """
        x = self.input_proj(data.expand_dims(-1))
        s_tok = F.expand_dims(spatial_token, axis=0)
        s_tok = F.expand_dims(s_tok, axis=2)
        t_tok = F.expand_dims(temporal_token, axis=0)
        t_tok = F.expand_dims(t_tok, axis=1)
        if self.debug:
            try:
                print("spatial_token:", getattr(s_tok, "shape", None))
                print("temporal_token:", getattr(t_tok, "shape", None))
                print("tokenized input:", getattr(x, "shape", None))
            except Exception:
                pass
        x = F.broadcast_add(x, s_tok)
        x = F.broadcast_add(x, t_tok)
        for i, layer in enumerate(self.layers):
            if self.debug:
                try:
                    print(f"input to layer {i}", getattr(x, "shape", None))
                except Exception:
                    pass
            x = layer(x)
        return x


class AlternatingTransformerTrainingNetwork(HybridBlock):
    """Training network producing distribution parameters."""

    @validated()
    def __init__(
        self,
        base_network: AlternatingTransformerNetwork,
        prediction_length: int,
        distr_output,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.prediction_length = prediction_length
        self.base_network = base_network
        self.distr_output = distr_output

        with self.name_scope():
            self.proj = distr_output.get_args_proj()

    def hybrid_forward(
        self, F, data: Tensor, spatial_token: Tensor, temporal_token: Tensor
    ) -> Tensor:
        features = self.base_network(data, spatial_token, temporal_token)
        pred = F.slice_axis(
            features, axis=2, begin=-self.prediction_length, end=None
        )
        params = self.proj(pred)
        distr = self.distr_output.distribution(params)
        return distr


class AlternatingTransformerPredictionNetwork(
    AlternatingTransformerTrainingNetwork
):
    """Prediction network generating samples."""

    @validated()
    def __init__(self, num_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_samples = num_samples

    def hybrid_forward(
        self, F, data: Tensor, spatial_token: Tensor, temporal_token: Tensor
    ) -> Tensor:
        distr = super().hybrid_forward(F, data, spatial_token, temporal_token)
        return distr.sample(num_samples=self.num_samples)


def reconcile_samples(
    reconciliation_mat: Tensor, samples: Tensor, seq_axis=None
) -> Tensor:
    """Project ``samples`` using ``reconciliation_mat`` as done in the
    hierarchical DeepVAR model."""
    if not seq_axis:
        return mx.nd.dot(samples, reconciliation_mat, transpose_b=True)
    num_dims = len(samples.shape)
    last_dim_in_seq_axis = num_dims - 1 in seq_axis or -1 in seq_axis
    assert not last_dim_in_seq_axis, (
        "The last dimension cannot be processed iteratively. Remove axis"
        f" {num_dims - 1} (or -1) from `seq_axis`."
    )
    num_seq_axes = len(seq_axis)
    samples = mx.nd.moveaxis(samples, seq_axis, list(range(num_seq_axes)))
    seq_axes_sizes = samples.shape[:num_seq_axes]
    out = [
        mx.nd.dot(samples[idx], reconciliation_mat, transpose_b=True)
        for idx in product(*[range(size) for size in seq_axes_sizes])
    ]
    out = mx.nd.concat(*out, dim=0).reshape(samples.shape)
    out = mx.nd.moveaxis(out, list(range(len(seq_axis))), seq_axis)
    return out


class PredictionHead(HybridBlock):
    """Simple prediction head to allow deterministic or probabilistic outputs."""

    @validated()
    def __init__(self, output_dim: int, distr_output=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.probabilistic = distr_output is not None
        with self.name_scope():
            if self.probabilistic:
                self.proj = distr_output.get_args_proj()
                self.distr_output = distr_output
            else:
                self.proj = nn.Dense(units=output_dim, flatten=False)

    def hybrid_forward(self, F, x: Tensor) -> Tensor:
        out = self.proj(x)
        if self.probabilistic:
            return self.distr_output.distribution(out)
        return out


class AlternatingTransformerHierarchicalTrainingNetwork(HybridBlock):
    """Training network forecasting bottom level and reconciling outputs."""

    @validated()
    def __init__(
        self,
        base_network: AlternatingTransformerNetwork,
        prediction_length: int,
        S: mx.nd.NDArray,
        loss: str = "mse",
        distr_output=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.prediction_length = prediction_length
        self.base_network = base_network
        self.S = S
        self.loss = loss
        self.num_bottom = int(S.shape[1])
        with self.name_scope():
            self.head = PredictionHead(self.num_bottom, distr_output)

    def hybrid_forward(
        self, F, data: Tensor, spatial_token: Tensor, temporal_token: Tensor, target: Tensor
    ) -> Tensor:
        features = self.base_network(data, spatial_token, temporal_token)
        pred = F.slice_axis(features, axis=2, begin=-self.prediction_length, end=None)
        bottom_feat = F.slice_axis(pred, axis=1, begin=-self.num_bottom, end=None)
        out = self.head(bottom_feat)

        target = F.slice_axis(target, axis=2, begin=-self.prediction_length, end=None)
        target_bottom = F.slice_axis(target, axis=1, begin=-self.num_bottom, end=None)

        if isinstance(out, mx.gluon.HybridBlock):
            raise RuntimeError("unexpected block returned by prediction head")

        if self.head.probabilistic:
            distr = out
            samples = distr.sample(num_samples=100)
            samples = samples.swapaxes(0, 1)
            samples_full = reconcile_samples(self.S.T, samples)
            crps = (
                EmpiricalDistribution(samples=samples_full, event_dim=1)
                .crps_univariate(x=target)
                .expand_dims(axis=-1)
            )
            return crps
        else:
            preds = out
            preds = preds.swapaxes(1, 2)
            full = F.linalg_gemm2(preds, self.S.T)
            if self.loss == "mae":
                return F.abs(full - target).mean(axis=1, keepdims=True)
            else:
                return F.square(full - target).mean(axis=1, keepdims=True)


class AlternatingTransformerHierarchicalPredictionNetwork(
    AlternatingTransformerHierarchicalTrainingNetwork
):
    """Prediction network generating coherent samples."""

    @validated()
    def __init__(self, num_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_samples = num_samples

    def hybrid_forward(
        self, F, data: Tensor, spatial_token: Tensor, temporal_token: Tensor, target: Tensor = None
    ) -> Tensor:
        features = self.base_network(data, spatial_token, temporal_token)
        pred = F.slice_axis(features, axis=2, begin=-self.prediction_length, end=None)
        bottom_feat = F.slice_axis(pred, axis=1, begin=-self.num_bottom, end=None)
        out = self.head(bottom_feat)

        if self.head.probabilistic:
            distr = out
            samples = distr.sample(num_samples=self.num_samples)
            samples = samples.swapaxes(0, 1)
            samples_full = reconcile_samples(self.S.T, samples)
            return samples_full
        else:
            preds = out.swapaxes(1, 2)
            full = F.linalg_gemm2(preds, self.S.T)
            return F.expand_dims(full, axis=0)
