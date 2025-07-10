import logging
from typing import List, Optional, Tuple
from itertools import product

import numpy as np
import mxnet as mx

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.distribution import Distribution, DistributionOutput
from gluonts.mx.distribution import EmpiricalDistribution
from gluonts.mx.util import assert_shape
from gluonts.mx.distribution import LowrankMultivariateGaussian
from gluonts.mx.block.feature import FeatureEmbedder

#TODO: add AlternatingTransformerNetwork, AlternatingTransformerTrainingNetwork, AlternatingTransformerPredictNetwork



logger = logging.getLogger(__name__)

class AlternatingTransformerNetwork(mx.gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        cardinality: List[int],
        embedding_dimension: int,
        lags_seq: List[int],
        scaling: bool = True,
        num_series: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.scaling = scaling
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.distr_output = distr_output
        self.num_series = num_series

        assert len(set(lags_seq)) == len(
            lags_seq
        ), "no duplicated lags allowed!"
        lags_seq.sort()

        self.lags_seq = lags_seq

        self.target_shape = distr_output.event_shape

        with self.name_scope():
            self.proj_dist_args = distr_output.get_args_proj()
            self.encoder = encoder
            self.decoder = decoder
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=[embedding_dimension for _ in cardinality],
            )

            if scaling:
                self.scaler = MeanScaler(keepdims=True)
            else:
                self.scaler = NOPScaler(keepdims=True)

    @staticmethod
    def get_lagged_subsequences(
        F,
        sequence: Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> Tensor:
        """
        Returns lagged subsequences of a given sequence.

        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length : int
            length of sequence in the T (time) dimension (axis = 1).
        indices : List[int]
            list of lag indices to be used.
        subsequences_length : int
            length of the subsequences to be extracted.

        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: sequence_length - lag_index - subsequences_length >= 0
        # for all lag_index, hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            "lags cannot go further than history length, found lag"
            f" {max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(
                F.slice_axis(
                    sequence, axis=1, begin=begin_index, end=end_index
                )
            )

        return F.stack(*lagged_values, axis=-1)

    def create_network_input(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,  # (batch_size, num_features, history_length)
        past_target: Tensor,  # (batch_size, history_length, 1)
        past_observed_values: Tensor,  # (batch_size, history_length)
        future_time_feat: Optional[
            Tensor
        ],  # (batch_size, num_features, prediction_length)
        future_target: Optional[Tensor],  # (batch_size, prediction_length)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Creates inputs for the transformer network.

        All tensor arguments should have NTC layout.
        """

        if future_time_feat is None or future_target is None:
            time_feat = past_time_feat.slice_axis(
                axis=1,
                begin=self.history_length - self.context_length,
                end=None,
            )
            sequence = past_target
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = F.concat(
                past_time_feat.slice_axis(
                    axis=1,
                    begin=self.history_length - self.context_length,
                    end=None,
                ),
                future_time_feat,
                dim=1,
            )
            sequence = F.concat(past_target, future_target, dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags = self.get_lagged_subsequences(
            F=F,
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)
        _, scale = self.scaler(
            past_target.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
        )
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help
        # prediction too(batch_size, num_features + prod(target_shape))
        static_feat = F.concat(
            embedded_cat,
            (
                F.log(scale)
                if len(self.target_shape) == 0
                else F.log(scale.squeeze(axis=1))
            ),
            dim=1,
        )

        repeated_static_feat = static_feat.expand_dims(axis=1).repeat(
            axis=1, repeats=subsequences_length
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled = F.broadcast_div(lags, scale.expand_dims(axis=-1))

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = F.reshape(
            data=lags_scaled,
            shape=(
                -1,
                subsequences_length,
                len(self.lags_seq) * prod(self.target_shape),
            ),
        )

        # (batch_size, sub_seq_len, input_dim)
        inputs = F.concat(input_lags, time_feat, repeated_static_feat, dim=-1)

        return inputs, scale, static_feat

    @staticmethod
    def upper_triangular_mask(F, d):
        mask = F.zeros_like(F.eye(d))
        for k in range(d - 1):
            mask = mask + F.eye(d, d, k + 1)
        return mask * LARGE_NEGATIVE_VALUE


class AlternatingTransformerTrainingNetwork(AlternatingTransformerNetwork):
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        """
        Computes the loss for training Transformer, all inputs tensors
        representing time series have NTC layout.

        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape,
            seq_len)
        future_time_feat : (batch_size, prediction_length, num_features)
        future_target : (batch_size, prediction_length, *target_shape)
        future_observed_values: (batch_size, prediction_length, *target_shape)

        Returns
        -------
        Loss with shape (batch_size, context + prediction_length, 1)
        """

        # create the inputs for the encoder
        inputs, scale, _ = self.create_network_input(
            F=F,
            feat_static_cat=feat_static_cat,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )

        enc_input = F.slice_axis(
            inputs, axis=1, begin=0, end=self.context_length
        )
        dec_input = F.slice_axis(
            inputs, axis=1, begin=self.context_length, end=None
        )

        # pass through encoder
        enc_out = self.encoder(enc_input)

        # input to decoder
        # TODO: check the masking operation in the decoder
        # TODO: modify the decoder
        dec_output = self.decoder(
            dec_input,
            enc_out,
            self.upper_triangular_mask(F, self.prediction_length),
        )

        # compute loss
        distr_args = self.proj_dist_args(dec_output)
        distr = self.distr_output.distribution(distr_args, scale=scale)
        loss = distr.loss(future_target)

        # mask loss
        weighted_loss = weighted_average(
            F=F,
            x=loss,
            weights=future_observed_values,
            axis=1,
        )

        return weighted_loss.mean()

class TransformerPredictionNetwork(TransformerNetwork):
    @validated()
    def __init__(self, num_parallel_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one, at the first time-step of
        # the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        F,
        static_feat: Tensor,
        past_target: Tensor,
        time_feat: Tensor,
        scale: Tensor,
        enc_out: Tensor,
    ) -> Tensor:
        """
        Computes sample paths by unrolling the LSTM starting with a initial
        input and state.

        Parameters
        ----------
        static_feat : Tensor
            static features. Shape: (batch_size, num_static_features).
        past_target : Tensor
            target history. Shape: (batch_size, history_length, 1).
        time_feat : Tensor
            time features. Shape:
            (batch_size, prediction_length, num_time_features).
        scale : Tensor
            tensor containing the scale of each element in the batch.
            Shape: (batch_size, ).
        enc_out: Tensor
            output of the encoder. Shape: (batch_size, num_cells)

        Returns
        --------
        sample_paths : Tensor
            a tensor containing sampled paths.
            Shape: (batch_size, num_sample_paths, prediction_length).
        """

        # blows-up the dimension of each tensor to batch_size *
        # self.num_parallel_samples for increasing parallelism
        repeated_past_target = past_target.repeat(
            repeats=self.num_parallel_samples, axis=0
        )
        repeated_time_feat = time_feat.repeat(
            repeats=self.num_parallel_samples, axis=0
        )
        repeated_static_feat = static_feat.repeat(
            repeats=self.num_parallel_samples, axis=0
        ).expand_dims(axis=1)
        repeated_enc_out = enc_out.repeat(
            repeats=self.num_parallel_samples, axis=0
        ).expand_dims(axis=1)
        repeated_scale = scale.repeat(
            repeats=self.num_parallel_samples, axis=0
        )

        future_samples = []

        # for each future time-units we draw new samples for this time-unit and
        # update the state
        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                F=F,
                sequence=repeated_past_target,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags_scaled = F.broadcast_div(
                lags, repeated_scale.expand_dims(axis=-1)
            )

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags = F.reshape(
                data=lags_scaled,
                shape=(-1, 1, prod(self.target_shape) * len(self.lags_seq)),
            )

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags +
            # num_time_features + num_static_features)
            dec_input = F.concat(
                input_lags,
                repeated_time_feat.slice_axis(axis=1, begin=k, end=k + 1),
                repeated_static_feat,
                dim=-1,
            )

            dec_output = self.decoder(dec_input, repeated_enc_out, None, False)

            distr_args = self.proj_dist_args(dec_output)

            # compute likelihood of target given the predicted parameters
            distr = self.distr_output.distribution(
                distr_args, scale=repeated_scale
            )

            # (batch_size * num_samples, 1, *target_shape)
            new_samples = distr.sample()

            # (batch_size * num_samples, seq_len, *target_shape)
            repeated_past_target = F.concat(
                repeated_past_target, new_samples, dim=1
            )
            future_samples.append(new_samples)

        # reset cache of the decoder
        self.decoder.cache_reset()

        # (batch_size * num_samples, prediction_length, *target_shape)
        samples = F.concat(*future_samples, dim=1)

        # (batch_size, num_samples, *target_shape, prediction_length)
        return samples.reshape(
            shape=(
                (-1, self.num_parallel_samples)
                + self.target_shape
                + (self.prediction_length,)
            )
        )

    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
    ) -> Tensor:
        """
        Predicts samples, all tensors should have NTC layout.

        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape)
        future_time_feat : (batch_size, prediction_length, num_features)

        Returns predicted samples
        -------
        """

        # create the inputs for the encoder
        inputs, scale, static_feat = self.create_network_input(
            F=F,
            feat_static_cat=feat_static_cat,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=None,
            future_target=None,
        )

        # pass through encoder
        enc_out = self.encoder(inputs)

        return self.sampling_decoder(
            F=F,
            past_target=past_target,
            time_feat=future_time_feat,
            static_feat=static_feat,
            scale=scale,
            enc_out=enc_out,
        )
    


def reconcile_samples(
    reconciliation_mat: Tensor,
    samples: Tensor,
    seq_axis: Optional[List] = None,
) -> Tensor:
    """
    Computes coherent samples by multiplying unconstrained `samples` with
    `reconciliation_mat`.

    Parameters
    ----------
    reconciliation_mat
        Shape: (target_dim, target_dim)
    samples
        Unconstrained samples
        Shape: `(*batch_shape, target_dim)`
        During training: (num_samples, batch_size, seq_len, target_dim)
        During prediction: (num_parallel_samples x batch_size, seq_len,
        target_dim)
    seq_axis
        Specifies the list of axes that should be reconciled sequentially.
        By default, all axes are processeed in parallel.

    Returns
    -------
    Tensor, shape same as that of `samples`
        Coherent samples
    """
    if not seq_axis:
        return mx.nd.dot(samples, reconciliation_mat, transpose_b=True)
    else:
        num_dims = len(samples.shape)

        last_dim_in_seq_axis = num_dims - 1 in seq_axis or -1 in seq_axis
        assert not last_dim_in_seq_axis, (
            "The last dimension cannot be processed iteratively. Remove axis"
            f" {num_dims - 1} (or -1) from `seq_axis`."
        )

        # In this case, reconcile samples by going over each index in
        # `seq_axis` iteratively. Note that `seq_axis` can be more than one
        # dimension.
        num_seq_axes = len(seq_axis)

        # bring the axes to iterate in the beginning
        samples = mx.nd.moveaxis(samples, seq_axis, list(range(num_seq_axes)))

        seq_axes_sizes = samples.shape[:num_seq_axes]
        out = [
            mx.nd.dot(samples[idx], reconciliation_mat, transpose_b=True)
            # get the sequential index from the cross-product of their sizes.
            for idx in product(*[range(size) for size in seq_axes_sizes])
        ]

        # put the axis in the correct order again
        out = mx.nd.concat(*out, dim=0).reshape(samples.shape)
        out = mx.nd.moveaxis(out, list(range(len(seq_axis))), seq_axis)
        return out


def coherency_error(S: np.ndarray, samples: np.ndarray) -> float:
    r"""
    Computes the maximum relative coherency error.

    .. math::

                \max_i | (S @ y_b)_i - y_i | / y_i

    where :math:`y` refers to the `samples` and :math:`y_b` refers to the
    samples at the bottom level.

    Parameters
    ----------
    S
        The summation matrix S. Shape:
        (total_num_time_series, num_bottom_time_series)
    samples
        Samples. Shape: `(*batch_shape, target_dim)`.

    Returns
    -------
    Float
        Coherency error
    """
    samples_bottom_level = samples[..., -S.shape[1] :]

    errs = np.abs(samples_bottom_level @ S.T - samples)
    rel_errs = np.where(
        samples == 0.0,
        errs,
        errs / np.abs(samples),
    )
    return rel_errs.max()


class AlternatingTransformerHierarchicalNetwork(AlternatingTransformerNetwork):
    @validated()
    def __init__(
        self,
        S,
        num_heads,
        num_layers,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        dropout_rate: float,
        lags_seq: List[int],
        target_dim: int,
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
        scaling: bool = True,
        seq_axis: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            num_cells=num_cells,
            cell_type=cell_type,
            history_length=history_length,
            context_length=context_length,
            prediction_length=prediction_length,
            distr_output=distr_output,
            dropout_rate=dropout_rate,
            lags_seq=lags_seq,
            target_dim=target_dim,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            scaling=scaling,
            **kwargs,
        )

        self.S = S
        self.seq_axis = seq_axis

    def get_samples_for_loss(self, distr: Distribution) -> Tensor:
        """
        Get samples to compute the final loss. These are samples directly drawn
        from the given `distr` if coherence is not enforced yet; otherwise the
        drawn samples are reconciled.

        Parameters
        ----------
        distr
            Distribution instances

        Returns
        -------
        samples
            Tensor with shape (num_samples, batch_size, seq_len, target_dim)
        """
        samples = distr.sample_rep(
            num_samples=self.num_samples_for_loss, dtype="float32"
        )

        # Determine which epoch we are currently in.
        self.batch_no += 1
        epoch_no = self.batch_no // self.num_batches_per_epoch + 1
        epoch_frac = epoch_no / self.epochs

        if (
            self.coherent_train_samples
            and epoch_frac > self.warmstart_epoch_frac
        ):
            coherent_samples = reconcile_samples(
                reconciliation_mat=self.M,
                samples=samples,
                seq_axis=self.seq_axis,
            )
            assert_shape(coherent_samples, samples.shape)
            return coherent_samples
        else:
            return samples

    def loss(self, F, target: Tensor, distr: Distribution) -> Tensor:
        """
        Computes loss given the output of the network in the form of
        distribution. The loss is given by:

            `self.CRPS_weight` * `loss_CRPS` + `self.likelihood_weight` *
            `neg_likelihoods`,

         where
          * `loss_CRPS` is computed on the samples drawn from the predicted
            `distr` (optionally after reconciling them),
          * `neg_likelihoods` are either computed directly using the predicted
            `distr` or from the estimated distribution based on (coherent)
            samples, depending on the `sample_LH` flag.

        Parameters
        ----------
        F
        target
            Tensor with shape (batch_size, seq_len, target_dim)
        distr
            Distribution instances

        Returns
        -------
        Loss
            Tensor with shape (batch_size, seq_length, 1)
        """

        # Sample from the predicted distribution if we are computing CRPS loss
        # or likelihood using the distribution based on (coherent) samples.
        # Samples shape: (num_samples, batch_size, seq_len, target_dim)
        if self.sample_LH or (self.CRPS_weight > 0.0):
            samples = self.get_samples_for_loss(distr=distr)

        if self.sample_LH:
            # Estimate the distribution based on (coherent) samples.
            distr = LowrankMultivariateGaussian.fit(F, samples=samples, rank=0)

        neg_likelihoods = -distr.log_prob(target).expand_dims(axis=-1)

        loss_CRPS = F.zeros_like(neg_likelihoods)
        if self.CRPS_weight > 0.0:
            loss_CRPS = (
                EmpiricalDistribution(samples=samples, event_dim=1)
                .crps_univariate(x=target)
                .expand_dims(axis=-1)
            )

        return (
            self.CRPS_weight * loss_CRPS
            + self.likelihood_weight * neg_likelihoods
        )

    def post_process_samples(self, samples: Tensor) -> Tensor:
        """
        Reconcile samples if `coherent_pred_samples` is True.

        Parameters
        ----------
        samples
            Tensor of shape (num_parallel_samples*batch_size, 1, target_dim)

        Returns
        -------
            Tensor of coherent samples.
        """
        if not self.coherent_pred_samples:
            samples_to_return = samples
        else:
            samples_to_return = reconcile_samples(
                reconciliation_mat=self.M,
                samples=samples,
                seq_axis=self.seq_axis,
            )
            assert_shape(samples_to_return, samples.shape)

        # Show coherency error: A*X_proj
        if self.log_coherency_error:
            coh_error = coherency_error(
                S=self.S, samples=samples_to_return.asnumpy()
            )
            logger.info(
                "Coherency error of the predicted samples for time step"
                f" {self.forecast_time_step}: {coh_error}"
            )
            self.forecast_time_step += 1

        return samples_to_return

class AlternatingTransformerHierarchicalTrainingNetwork(
    AlternatingTransformerHierarchicalNetwork, AlternatingTransformerTrainingNetwork
):
    def __init__(
        self,
        num_samples_for_loss: int,
        likelihood_weight: float,
        CRPS_weight: float,
        coherent_train_samples: bool,
        warmstart_epoch_frac: float,
        epochs: float,
        num_batches_per_epoch: float,
        sample_LH: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_samples_for_loss = num_samples_for_loss
        self.likelihood_weight = likelihood_weight
        self.CRPS_weight = CRPS_weight
        self.coherent_train_samples = coherent_train_samples
        self.warmstart_epoch_frac = warmstart_epoch_frac
        self.epochs = epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.batch_no = 0
        self.sample_LH = sample_LH

        # Assert CRPS_weight, likelihood_weight, and coherent_train_samples
        # have harmonious values
        assert self.CRPS_weight >= 0.0, "CRPS weight must be non-negative"
        assert (
            self.likelihood_weight >= 0.0
        ), "Likelihood weight must be non-negative!"
        assert (
            self.likelihood_weight + self.CRPS_weight > 0.0
        ), "At least one of CRPS or likelihood weights must be non-zero"
        if self.CRPS_weight == 0.0 and self.coherent_train_samples:
            assert (
                "No sampling being performed. "
                "coherent_train_samples flag is ignored"
            )
        if not self.sample_LH == 0.0 and self.coherent_train_samples:
            assert (
                "No sampling being performed. "
                "coherent_train_samples flag is ignored"
            )
        if self.likelihood_weight == 0.0 and self.sample_LH:
            assert (
                "likelihood_weight is 0 but sample likelihoods are still "
                "being calculated. Set sample_LH=0 when likelihood_weight=0"
            )


class AlternatingTransformerHierarchicalPredictionNetwork(
    AlternatingTransformerHierarchicalNetwork, AlternatingTransformerPredictionNetwork
):
    @validated()
    def __init__(
        self,
        num_parallel_samples: int,
        log_coherency_error: bool,
        coherent_pred_samples: bool,
        **kwargs,
    ) -> None:
        super().__init__(num_parallel_samples=num_parallel_samples, **kwargs)
        self.coherent_pred_samples = coherent_pred_samples
        self.log_coherency_error = log_coherency_error
        if log_coherency_error:
            self.forecast_time_step = 1
    
    
