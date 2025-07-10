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

# simple dense decoder ---------------------------------------------------------#
class HierMLPDecoder(HybridBlock):
    """
    Map encoder output to bottom-level forecasts.
    Output: (B, bottom_count, prediction_length)
    """
    def __init__(self, bottom_count: int, pred_len: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        out_units = bottom_count * pred_len
        with self.name_scope():
            self.net = nn.HybridSequential()
            self.net.add(
                nn.Dense(hidden, activation="relu"),
                nn.Dropout(dropout),
                nn.Dense(out_units),
            )
        self.bottom = bottom_count
        self.pred_len = pred_len

    def hybrid_forward(self, F, x):  # x: (B, T, bottom, d)
        b = x.shape[0]
        y = self.net(x.reshape((b, -1)))               # (B, bottom*pred)
        return y.reshape((b, self.bottom, self.pred_len))

# -----------------------------------------------------------------------------#
class AltHierNetworkBase(HybridBlock):
    """Shared utilities for training & prediction."""
    @validated()
    def __init__(
        self,
        *,
        encoder: AlternatingTransformerEncoder,
        history_length: int,
        context_length: int,
        prediction_length: int,
        S: np.ndarray,                          # (M × B)
        cardinality: List[int],
        embedding_dimension: int,
        lags_seq: List[int],
        reconciliation_method: str = "ols",     # "ols" | "bottom_up"
        scaling: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert reconciliation_method in {"ols", "bottom_up"}

        self.encoder = encoder
        self.history_length = history_length
        self.context_length = context_length
        self.pred_len = prediction_length
        self.lags_seq = sorted(lags_seq)
        self.scaling = scaling

        self.S_np = S.astype("float32")
        self.R_np = projection_matrix(S) if reconciliation_method == "ols" else self.S_np

        with self.name_scope():
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=[embedding_dimension] * len(cardinality),
            )
            self.scaler = MeanScaler(keepdims=True) if scaling else NOPScaler(keepdims=True)

    # lag helper ----------------------------------------------------------------
    @staticmethod
    def get_lagged_subsequences(F, seq: Tensor, seq_len: int, idx: List[int], sub_len: int):
        assert max(idx) + sub_len <= seq_len
        out = [F.slice_axis(seq, 1, -lag - sub_len, None if lag == 0 else -lag) for lag in idx]
        return F.stack(*out, axis=-1)

# -----------------------------------------------------------------------------#
class AltHierTrainingNetwork(AltHierNetworkBase):
    """Returns MSE loss."""
    @validated()
    def __init__(self, *, decoder: HierMLPDecoder, **base):
        super().__init__(**base)
        self.decoder = decoder

    # input builder -------------------------------------------------------------
    def _make_input(
        self,
        F,
        feat_static_cat,
        past_time_feat,
        past_target,
        past_observed,
        future_time_feat,
        future_target,
    ) -> Tuple[Tensor, Tensor]:
        if future_time_feat is None:
            time_feat = past_time_feat.slice_axis(axis=1, begin=self.history_length - self.context_length, end=None)
            sequence, seq_len, sub_len = past_target, self.history_length, self.context_length
        else:
            time_feat = F.concat(
                past_time_feat.slice_axis(axis=1, begin=self.history_length - self.context_length, end=None),
                future_time_feat,
                dim=1,
            )
            sequence = F.concat(past_target, future_target, dim=1)
            seq_len = self.history_length + self.pred_len
            sub_len = self.context_length + self.pred_len

        lags = self.get_lagged_subsequences(F, sequence, seq_len, self.lags_seq, sub_len)
        _, scale = self.scaler(
            past_target.slice_axis(axis=1, begin=-self.context_length, end=None),
            past_observed.slice_axis(axis=1, begin=-self.context_length, end=None),
        )
        emb = self.embedder(feat_static_cat)
        static = F.concat(emb, F.log(scale.squeeze(axis=1)), dim=1)
        rep_static = static.expand_dims(1).repeat(axis=1, repeats=sub_len)

        lags_scaled = F.broadcast_div(lags, scale.expand_dims(-1))
        input_lags = lags_scaled.reshape((-1, sub_len, len(self.lags_seq)))
        inputs = F.concat(input_lags, time_feat, rep_static, dim=-1).expand_dims(axis=2)
        return inputs, scale

    # forward -------------------------------------------------------------------
    def hybrid_forward(
        self,
        F,
        feat_static_cat,
        past_time_feat,
        past_target,
        past_observed_values,
        future_time_feat,
        future_target,
        future_observed_values,
    ):
        x, scale = self._make_input(
            F,
            feat_static_cat,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )

        print("[Train] encoder in :", x.shape)
        enc_out = self.encoder(x)
        print("[Train] encoder out:", enc_out.shape)

        pred_bottom_scaled = self.decoder(enc_out)                 # (B, BOTTOM, pred)
        pred_bottom = pred_bottom_scaled * scale                   # rescale
        print("[Train] bottom preds:", pred_bottom.shape)

        # MSE on bottom level -----------------------------------------------
        mse = F.square(pred_bottom.swapaxes(1, 2) - future_target)  # align dims (B, pred, BOTTOM)
        loss = weighted_average(F, mse, future_observed_values, axis=1)
        return loss.mean()

# -----------------------------------------------------------------------------#
class AltHierPredictionNetwork(AltHierNetworkBase):
    """Outputs coherent forecasts (B, M, pred_len)."""
    @validated()
    def __init__(self, *, decoder: HierMLPDecoder, **base):
        super().__init__(**base)
        self.decoder = decoder

    def hybrid_forward(
        self,
        F,
        feat_static_cat,
        past_time_feat,
        past_target,
        past_observed_values,
        future_time_feat,
    ):
        x, scale = self._make_input(
            F,
            feat_static_cat,
            past_time_feat,
            past_target,
            past_observed_values,
            None,
            None,
        )
        enc = self.encoder(x)
        bottom = self.decoder(enc) * scale                         # (B, BOTTOM, pred)
        B, BOTTOM, P = bottom.shape
        R = np.array(self.R_np, ctx=bottom.context)             # (M, BOTTOM)

        # reconcile per horizon step -----------------------------------------
        flat = bottom.transpose((0, 2, 1)).reshape((-1, BOTTOM))   # (B*P, BOTTOM)
        all_nodes = np.dot(flat, R.T).reshape((B, P, -1))       # (B, P, M)
        coherent = all_nodes.transpose((0, 2, 1))                  # (B, M, P)

        print("[Pred] coherent shape:", coherent.shape)
        return coherent
