# -----------------------------------------------------------------------------#
# Alternating-Attention Transformer – deterministic, hierarchy-coherent        #
# -----------------------------------------------------------------------------#
from typing import List, Optional, Tuple

import numpy as np
import mxnet as mx
from mxnet.gluon import HybridBlock, nn
from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.util import weighted_average
from gluonts.core.component import validated
from gluonts.itertools import prod

from .alternating_encoder import AlternatingTransformerEncoder

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
        return np.dot(samples, reconciliation_mat, transpose_b=True)
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
        samples = np.moveaxis(samples, seq_axis, list(range(num_seq_axes)))

        seq_axes_sizes = samples.shape[:num_seq_axes]
        out = [
            np.dot(samples[idx], reconciliation_mat, transpose_b=True)
            # get the sequential index from the cross-product of their sizes.
            for idx in product(*[range(size) for size in seq_axes_sizes])
        ]

        # put the axis in the correct order again
        out = np.concat(*out, dim=0).reshape(samples.shape)
        out = np.moveaxis(out, list(range(len(seq_axis))), seq_axis)
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
        M,
        S,
        num_layers: int,
        num_cells: int,
        cell_type: str,
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

        self.M = M
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
