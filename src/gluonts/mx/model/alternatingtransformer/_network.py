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
