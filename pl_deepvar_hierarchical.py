# pl_deepvar_hierarchical.py
# PyTorch Lightning reproduction of GluonTS DeepVARHierarchical (MXNet)
# Parity notes:
# - Time features & lags match gluonts.mx.model.deepvar._estimator
# - Windowing mimics InstanceSplitter + ExpectedNumInstanceSampler
# - Scaling follows MeanScaler semantics (minimum_scale=1e-10 + batch fallback)
# - Target-dimension embedding uses embedding_dimension=5
# - Diagonal Normal head (rank=0 parity). For rank>0, implement low-rank MVN.
# - Coherent train/pred samples via P = S(SᵀS)^+Sᵀ

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from hierarchicalforecast.evaluation import scaled_crps, rel_mse, msse
from datasetsforecast.hierarchical import HierarchicalInfo, HierarchicalData
from pandas.tseries.frequencies import to_offset
from packaging.version import Version

# ------------------------------
# ICML'21 configs (as given)
# ------------------------------
CONFIGS = {
    "Labour": {
        "epochs": 50,
        "num_batches_per_epoch": 50,
        "scaling": True,
        "pick_incomplete": False,
        "batch_size": 32,
        "num_parallel_samples": 200,
        "hybridize": False,
        "learning_rate": 0.001,
        "context_length": 24,
        "rank": 0,
        "assert_reconciliation": False,
        "num_deep_models": 1,
        "num_layers": 2,
        "num_cells": 40,
        "coherent_train_samples": True,
        "coherent_pred_samples": True,
        "likelihood_weight": 0.0,
        "CRPS_weight": 1.0,
        "num_samples_for_loss": 200,
        "sample_LH": False,
        "rec_weight": 0.0,
        "seq_axis": [1],
        "warmstart_epoch_frac": 0.1,
    },
    "Traffic": {
        "epochs": 50,
        "num_batches_per_epoch": 50,
        "scaling": True,
        "pick_incomplete": False,
        "batch_size": 32,
        "num_parallel_samples": 200,
        "hybridize": False,
        "learning_rate": 0.001,
        "context_length": 40,
        "rank": 0,
        "assert_reconciliation": False,
        "num_deep_models": 1,
        "num_layers": 2,
        "num_cells": 40,
        "coherent_train_samples": True,
        "coherent_pred_samples": True,
        "likelihood_weight": 1.0,
        "CRPS_weight": 0.0,
        "num_samples_for_loss": 50,
        "sample_LH": True,
        "seq_axis": [1],
        "warmstart_epoch_frac": 0.1,
    },
    "OldTraffic": {
        "epochs": 50,
        "num_batches_per_epoch": 50,
        "scaling": True,
        "pick_incomplete": False,
        "batch_size": 32,
        "num_parallel_samples": 200,
        "hybridize": False,
        "learning_rate": 0.001,
        "context_length": 40,
        "rank": 0,
        "assert_reconciliation": False,
        "num_deep_models": 1,
        "num_layers": 2,
        "num_cells": 40,
        "coherent_train_samples": True,
        "coherent_pred_samples": True,
        "likelihood_weight": 1.0,
        "CRPS_weight": 0.0,
        "num_samples_for_loss": 50,
        "sample_LH": True,
        "seq_axis": [1],
        "warmstart_epoch_frac": 0.1,
    },
    "TourismSmall": {
        "epochs": 10,
        "num_batches_per_epoch": 50,
        "scaling": True,
        "pick_incomplete": True,
        "batch_size": 32,
        "num_parallel_samples": 200,
        "hybridize": False,
        "learning_rate": 0.001,
        "context_length": 24,
        "rank": 0,
        "assert_reconciliation": False,
        "num_deep_models": 1,
        "num_layers": 2,
        "num_cells": 40,
        "coherent_train_samples": True,
        "coherent_pred_samples": True,
        "likelihood_weight": 1.0,
        "CRPS_weight": 0.0,
        "num_samples_for_loss": 50,
        "sample_LH": True,
        "seq_axis": [],
        "warmstart_epoch_frac": 0.0,
    },
    "TourismLarge": {
        "epochs": 40,
        "num_batches_per_epoch": 50,
        "scaling": True,
        "pick_incomplete": False,
        "batch_size": 4,
        "num_parallel_samples": 200,
        "hybridize": False,
        "learning_rate": 0.001,
        "context_length": 36,
        "rank": 0,
        "assert_reconciliation": False,
        "num_deep_models": 1,
        "num_layers": 2,
        "num_cells": 40,
        "coherent_train_samples": True,
        "coherent_pred_samples": True,
        "likelihood_weight": 1.0,
        "CRPS_weight": 0.0,
        "num_samples_for_loss": 50,
        "sample_LH": True,
        "seq_axis": [1],
        "warmstart_epoch_frac": 0.0,
    },
    "OldTourismLarge": {
        "epochs": 40,
        "num_batches_per_epoch": 50,
        "scaling": True,
        "pick_incomplete": False,
        "batch_size": 4,
        "num_parallel_samples": 200,
        "hybridize": False,
        "learning_rate": 0.001,
        "context_length": 36,
        "rank": 0,
        "assert_reconciliation": False,
        "num_deep_models": 1,
        "num_layers": 2,
        "num_cells": 40,
        "coherent_train_samples": True,
        "coherent_pred_samples": True,
        "likelihood_weight": 1.0,
        "CRPS_weight": 0.0,
        "num_samples_for_loss": 50,
        "sample_LH": True,
        "seq_axis": [1],
        "warmstart_epoch_frac": 0.0,
    },
    "Wiki2": {
        "epochs": 50,
        "num_batches_per_epoch": 50,
        "scaling": True,
        "pick_incomplete": False,
        "batch_size": 32,
        "num_parallel_samples": 200,
        "hybridize": False,
        "learning_rate": 0.001,
        "context_length": 15,
        "rank": 0,
        "assert_reconciliation": False,
        "num_deep_models": 1,
        "num_layers": 2,
        "num_cells": 40,
        "coherent_train_samples": True,
        "coherent_pred_samples": True,
        "likelihood_weight": 0.0,
        "CRPS_weight": 1.0,
        "num_samples_for_loss": 100,
        "sample_LH": False,
        "rec_weight": 0.0,
        "seq_axis": [1],
        "warmstart_epoch_frac": 0.1,
    },
}


# ------------------------------
# Time features & lags (from GluonTS DeepVAR estimator)
# ------------------------------
class FourierDateFeatures:
    # Matches gluonts.mx.model.deepvar._estimator.FourierDateFeatures
    def __init__(self, freq: str) -> None:
        freqs = [
            "month",
            "quarter",
            "day",
            "hour",
            "minute",
            "weekofyear",
            "weekday",
            "dayofweek",
            "dayofyear",
            "daysinmonth",
        ]
        assert freq in freqs, f"Unsupported Fourier feature: {freq}"
        self.freq = freq

    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        values = getattr(index, self.freq)
        num_values = int(np.max(values)) + 1
        steps = np.array(values) * 2.0 * np.pi / float(num_values)
        return np.vstack([np.cos(steps), np.sin(steps)])  # [2, T]


def norm_freq_str(freq_str: str) -> str:
    base_freq = freq_str.split("-")[0]
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        base_freq = base_freq[:-1]
        if Version(pd.__version__) >= Version("2.2.0"):
            base_freq += "E"
    return base_freq


def time_features_from_frequency_str(freq_str: str) -> List[FourierDateFeatures]:
    # Matches mapping in DeepVAR estimator (no Q*/QE* support)
    features = {
        "M": ["weekofyear"],
        "ME": ["weekofyear"],
        "Q": ["quarter"],
        "QE": ["quarter"],
        "W": ["daysinmonth", "weekofyear"],
        "D": ["dayofweek"],
        "B": ["dayofweek", "dayofyear"],
        "H": ["hour", "dayofweek"],
        "h": ["hour", "dayofweek"],
        "min": ["minute", "hour", "dayofweek"],
        "T": ["minute", "hour", "dayofweek"],
    }
    offset = to_offset(freq_str)
    granularity = norm_freq_str(offset.name)
    if granularity not in features and offset.name in features:
        granularity = offset.name
    assert granularity in features, f"freq {granularity} not supported by DeepVAR mapping"
    return [FourierDateFeatures(freq=f) for f in features[granularity]]


def get_lags_for_frequency(freq_str: str, num_lags: Optional[int] = None) -> List[int]:
    # Matches DeepVAR estimator
    offset = to_offset(freq_str)
    granularity = norm_freq_str(offset.name)
    if granularity in ["M", "ME"]:
        lags = [[1, 12]]
    elif granularity in ["Q", "QE"]:
        lags = [[1, 4]]
    elif granularity == "D":
        lags = [[1, 7, 14]]
    elif granularity == "B":
        lags = [[1, 2]]
    elif granularity in ["H", "h"]:
        lags = [[1, 24, 168]]
    elif granularity in ("min", "T"):
        lags = [[1, 4, 12, 24, 48]]
    else:
        lags = [[1]]
    output_lags = sorted({int(lag) for sub in lags for lag in sub})
    return output_lags[:num_lags] if num_lags is not None else output_lags


# ------------------------------
# Projection matrix (hierarchical reconciliation)
# ------------------------------
def projection_mat(S: np.ndarray, D: Optional[np.ndarray] = None) -> np.ndarray:
    if D is None:
        return S @ np.linalg.pinv(S.T @ S) @ S.T
    else:
        assert np.allclose(D, D.T), "`D` must be symmetric."
        eig = np.linalg.eigvals(D)
        assert np.all(eig > 0), "`D` must be positive definite."
        return S @ np.linalg.pinv(S.T @ D @ S) @ S.T @ D


# ------------------------------
# Data loading (all nodes)
# ------------------------------
def sort_y_df_like_s(Y_df: pd.DataFrame, S_df: pd.DataFrame) -> pd.DataFrame:
    Y_df = Y_df.copy()
    Y_df.unique_id = Y_df.unique_id.astype("category")
    Y_df.unique_id = Y_df.unique_id.cat.set_categories(S_df.index)
    Y_df = Y_df.sort_values(by=["unique_id", "ds"])
    return Y_df


@dataclass
class HierDataBundle:
    Y_all_TD: np.ndarray  # [T, n_all]
    Y_bottom_TB: np.ndarray  # [T, n_bottom]
    S_df: pd.DataFrame
    A_all_from_bottom: np.ndarray  # [n_all, n_bottom]
    horizon: int
    freq_raw: str
    freq_for_features: str
    hier_idxs: List[np.ndarray]
    hier_levels: List[str]
    time_feats_TF: np.ndarray  # [T, F]
    period_index: pd.PeriodIndex


def load_hierarchical_dataset_all_nodes(dataset: str, directory: str = "./data") -> HierDataBundle:
    info = HierarchicalInfo[dataset]
    Y_df, S_df, tags = HierarchicalData.load(directory=directory, group=dataset)
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])
    Y_df = sort_y_df_like_s(Y_df, S_df)

    # bottom wide
    Y_bottom_df = (
        Y_df.pivot(index="ds", columns="unique_id", values="y").loc[:, S_df.columns]
    )
    idx = Y_bottom_df.index

    # all nodes
    A = S_df.values.astype(np.float32)  # [n_all, n_bottom]
    Y_all = (A @ Y_bottom_df.to_numpy().T).T.astype(np.float32)  # [T, n_all]

    # frequency used for TF/lag mapping must be supported by DeepVAR mapping
    freq_raw = info.freq
    freq_for_features = norm_freq_str(to_offset(freq_raw).name)

    # time features exactly like DeepVAR: Fourier over mapping
    pidx = idx.to_period(freq_raw)
    feats = time_features_from_frequency_str(freq_raw)
    feat_blocks = [feat(pidx) for feat in feats]  # each [2, T]
    time_feats = np.concatenate(feat_blocks, axis=0).T.astype(np.float32)  # [T, F]

    hier_levels = ["Overall"] + list(tags.keys())
    hier_idxs = [np.arange(len(S_df))] + [S_df.index.get_indexer(tags[level]) for level in list(tags.keys())]

    return HierDataBundle(
        Y_all_TD=Y_all,
        Y_bottom_TB=Y_bottom_df.to_numpy().astype(np.float32),
        S_df=S_df,
        A_all_from_bottom=A,
        horizon=info.papers_horizon,
        freq_raw=freq_raw,
        freq_for_features=freq_for_features,
        hier_idxs=hier_idxs,
        hier_levels=hier_levels,
        time_feats_TF=time_feats,
        period_index=pidx,
    )


# ------------------------------
# InstanceSplitter-like dataset (multivariate)
# ------------------------------
class MultiSeriesWindows(Dataset):
    """
    Deterministic windows; the DataLoader sampler controls which indices are drawn,
    mirroring GluonTS (ExpectedNumInstanceSampler -> InstanceSplitter).
    """

    def __init__(
        self,
        Y_all_TD: np.ndarray,  # [T, D]
        time_feats_TF: np.ndarray,  # [T, F]
        context_length: int,
        prediction_length: int,
        freq_str: str,
        pick_incomplete: bool,
        train: bool,
    ):
        super().__init__()
        self.Y = torch.as_tensor(Y_all_TD, dtype=torch.float32)  # [T, D]
        self.F = torch.as_tensor(time_feats_TF, dtype=torch.float32)  # [T, F]
        self.T, self.D = self.Y.shape
        self.cl = context_length
        self.pl = prediction_length
        self.lags = get_lags_for_frequency(freq_str)
        self.maxlag = max(self.lags) if self.lags else 0
        self.H = self.cl + self.maxlag
        self.pick_incomplete = pick_incomplete
        self.train = train

        self.min_past = 0 if pick_incomplete else self.H
        self.min_t = self.min_past
        self.max_t = self.T - self.pl

        self.t_indices: List[int] = []
        if self.max_t >= self.min_t:
            if train:
                # all admissible cut points; sampler decides which ones per epoch
                self.t_indices = list(range(self.min_t, self.max_t + 1))
            else:
                # single final split for validation (ValidationSplitSampler)
                self.t_indices = [self.max_t]
        else:
            # Fallback: one padded example at the end
            self.t_indices = [max(self.min_t, 0)]

    def __len__(self):
        return len(self.t_indices)

    def _build_window(self, t: int):
        start = max(0, t - self.H)
        past_y = self.Y[start:t, :]  # [L, D]
        past_f = self.F[start:t, :]  # [L, F]
        L = past_y.shape[0]
        pad_len = self.H - L
        if pad_len > 0:
            pad_y = torch.zeros(pad_len, self.D, dtype=past_y.dtype)
            pad_f = torch.zeros(pad_len, self.F.shape[1], dtype=past_f.dtype)
            past_y = torch.cat([pad_y, past_y], dim=0)
            past_f = torch.cat([pad_f, past_f], dim=0)
        past_is_pad = torch.zeros(self.H, dtype=past_y.dtype)
        if pad_len > 0:
            past_is_pad[:pad_len] = 1.0
        past_obsv = torch.ones(self.H, self.D, dtype=past_y.dtype)
        if pad_len > 0:
            past_obsv[:pad_len, :] = 0.0

        fut_y = self.Y[t : t + self.pl, :]  # [P, D]
        fut_f = self.F[t : t + self.pl, :]  # [P, F]
        return past_y, past_f, past_is_pad, past_obsv, fut_y, fut_f

    def __getitem__(self, idx):
        t = self.t_indices[idx]
        return self._build_window(t)


# ------------------------------
# CRPS (sample-based)
# ------------------------------
def crps_empirical_samples(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # samples: [K,B,S,D], target: [B,S,D]
    term1 = (samples - target.unsqueeze(0)).abs().mean(dim=0)  # [B,S,D]
    diff = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs()  # [K,K,B,S,D]
    term2 = 0.5 * diff.mean(dim=(0, 1))
    return (term1 - term2).mean()


# ------------------------------
# LightningModule: DeepVARHierarchical (rank=0 parity)
# ------------------------------
class DeepVARHierPL(pl.LightningModule):
    def __init__(
        self,
        target_dim: int,
        context_length: int,
        prediction_length: int,
        freq_str: str,
        num_layers: int = 2,
        num_cells: int = 40,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.1,
        coherent_train_samples: bool = True,
        coherent_pred_samples: bool = True,
        warmstart_epoch_frac: float = 0.0,
        num_samples_for_loss: int = 200,
        num_parallel_samples: int = 200,
        CRPS_weight: float = 1.0,
        likelihood_weight: float = 0.0,
        sample_LH: bool = False,
        num_batches_per_epoch: int = 50,
        epochs: int = 10,
        S: Optional[np.ndarray] = None,
        D_weight: Optional[np.ndarray] = None,
        time_feat_dim: int = 0,
        embedding_dimension: int = 5,  # DeepVAR default
        scaling: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["S", "D_weight"])
        self.D = target_dim
        self.cl = context_length
        self.pl = prediction_length
        self.freq_str = freq_str
        self.lags = get_lags_for_frequency(freq_str)
        self.maxlag = max(self.lags) if self.lags else 0
        self.H = self.cl + self.maxlag
        self.lr = learning_rate
        self.dropout = dropout_rate
        self.coherent_train_samples = coherent_train_samples
        self.coherent_pred_samples = coherent_pred_samples
        self.warmstart_epoch_frac = warmstart_epoch_frac
        self.K_loss = num_samples_for_loss
        self.K_pred = num_parallel_samples
        self.CRPS_w = CRPS_weight
        self.LH_w = likelihood_weight
        self.sample_LH = sample_LH
        self.num_batches_per_epoch = num_batches_per_epoch
        self.epochs_total = epochs
        self.batch_no = 0
        self.scaling = scaling

        if S is not None:
            P = projection_mat(
                S.astype(np.float64),
                D_weight.astype(np.float64) if D_weight is not None else None,
            ).astype(np.float32)
            self.register_buffer("P", torch.from_numpy(P))
        else:
            self.P = None

        # Target-dimension indicator embedding
        self.embed_dim = embedding_dimension
        self.embed = nn.Embedding(self.D, self.embed_dim)

        # Input size: lags * D  +  D*embed_dim  + time features
        self.input_size = len(self.lags) * self.D + self.D * self.embed_dim + time_feat_dim

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout if num_layers > 1 else 0.0,
        )
        # Diagonal Normal: mean & log-scale per dim
        self.proj = nn.Linear(num_cells, 2 * self.D)

    def get_lagged_subsequences(
        self,
        sequence: torch.Tensor,  # [B, T, D]
        sequence_length: int,
        indices: List[int],
        subsequences_length: int,
    ) -> torch.Tensor:
        assert max(indices) + subsequences_length <= sequence_length, \
            f"lags go beyond history: max {max(indices)}, hist {sequence_length}"
        B, T, D = sequence.shape
        cols = []
        for lag in indices:
            begin = T - lag - subsequences_length
            end = T - lag if lag > 0 else None
            cols.append(sequence[:, begin:end, :].unsqueeze(2))  # [B,S,1,D]
        lags = torch.cat(cols, dim=2).permute(0, 1, 3, 2).contiguous()  # [B,S,D,I]
        return lags

    def compute_mean_scale(self, past_target: torch.Tensor, past_obsv: torch.Tensor) -> torch.Tensor:
        """
        MeanScaler semantics: per-series average absolute value over the last context window,
        using observed mask; items with no observed values fall back to the batch mean.
        minimum_scale = 1e-10.
        """
        x = past_target[:, -self.cl :, :]  # [B, cl, D]
        m = past_obsv[:, -self.cl :, :]  # [B, cl, D]
        num = (x.abs() * m).sum(dim=1)  # [B, D]
        den = m.sum(dim=1)  # [B, D]

        scale = torch.zeros_like(num)
        has_obs = den > 0
        scale[has_obs] = num[has_obs] / den[has_obs]

        # batch fallback for series with no observations
        if (~has_obs).any():
            if has_obs.any():
                batch_mean = scale[has_obs].mean()
            else:
                batch_mean = torch.tensor(1.0, dtype=scale.dtype, device=scale.device)
            scale[~has_obs] = batch_mean

        scale = scale.clamp_min(1e-10)  # minimum_scale
        return scale.unsqueeze(1)  # [B,1,D]

    def build_distribution(self, rnn_out: torch.Tensor, scale: torch.Tensor):
        B, S, Hdim = rnn_out.shape
        params = self.proj(rnn_out).view(B, S, self.D, 2)
        mu_hat = params[..., 0]
        log_sig = params[..., 1]
        sigma_hat = F.softplus(log_sig) + 1e-6
        # If scaling disabled, use scale=1
        if not self.scaling:
            scale = torch.ones_like(scale)
        mu = mu_hat * scale
        sigma = sigma_hat * scale
        distr = torch.distributions.Normal(loc=mu, scale=sigma)
        return distr, (mu, sigma)

    def unroll_encoder(
        self,
        past_time: torch.Tensor,  # [B, H, F]
        past_target: torch.Tensor,  # [B, H, D]
        past_obsv: torch.Tensor,  # [B, H, D]
        past_is_pad: torch.Tensor,  # [B, H]
        future_time: Optional[torch.Tensor],
        future_target: Optional[torch.Tensor],
        target_dim_indicator: torch.Tensor,  # [B, D]
    ):
        B, H, D = past_target.shape
        P = self.pl
        # observed mask = OBSERVED_VALUES ∧ ¬IS_PAD
        past_obsv = torch.minimum(past_obsv, (1 - past_is_pad.unsqueeze(-1)))

        if future_time is None or future_target is None:
            time_feat = past_time[:, -self.cl :, :]
            sequence = past_target
            total_len = self.H
            subseq_len = self.cl
        else:
            time_feat = torch.cat([past_time[:, -self.cl :, :], future_time], dim=1)
            sequence = torch.cat([past_target, future_target], dim=1)
            total_len = self.H + P
            subseq_len = self.cl + P

        lags = self.get_lagged_subsequences(sequence, total_len, self.lags, subseq_len)
        scale = self.compute_mean_scale(past_target, past_obsv)
        lags_scaled = lags / scale.unsqueeze(-1)

        emb = self.embed(target_dim_indicator)  # [B,D,E]
        emb_rep = emb.unsqueeze(1).repeat(1, subseq_len, 1, 1)  # [B,S,D,E]

        input_lags = lags_scaled.reshape(B, subseq_len, self.D * len(self.lags))
        input_emb = emb_rep.reshape(B, subseq_len, self.D * self.embed_dim)
        x = torch.cat([input_lags, input_emb, time_feat], dim=-1)

        out, state = self.rnn(x)
        return out, state, scale, lags_scaled, x

    def training_step(self, batch, batch_idx):
        self.batch_no += 1
        (past_y, past_f, past_is_pad, past_obsv, fut_y, fut_f) = batch
        B = past_y.shape[0]
        idx = torch.arange(self.D, device=self.device).long().unsqueeze(0).repeat(B, 1)

        rnn_out, _, scale, _, _ = self.unroll_encoder(
            past_time=past_f,
            past_target=past_y,
            past_obsv=past_obsv,
            past_is_pad=past_is_pad,
            future_time=fut_f,
            future_target=fut_y,
            target_dim_indicator=idx,
        )
        seq_len = self.cl + self.pl
        target = torch.cat([past_y[:, -self.cl :, :], fut_y], dim=1)  # [B, S, D]
        obsv = torch.cat(
            [
                torch.minimum(
                    past_obsv[:, -self.cl :, :],
                    1 - past_is_pad[:, -self.cl :].unsqueeze(-1),
                ),
                torch.ones_like(fut_y),
            ],
            dim=1,
        )
        loss_weights = obsv.min(dim=-1, keepdim=True).values  # [B, S, 1]

        distr, _ = self.build_distribution(rnn_out, scale)
        samples = distr.rsample((self.K_loss,))  # [K,B,S,D]

        # warmstart gating for training-time coherence
        epoch_no = self.batch_no // max(1, self.num_batches_per_epoch) + 1
        epoch_frac = epoch_no / max(1, self.epochs_total)
        if (
            self.coherent_train_samples
            and (self.P is not None)
            and (epoch_frac > self.warmstart_epoch_frac)
        ):
            samples = samples @ self.P.T

        crps_full = crps_empirical_samples(samples, target)

        nll_per_entry = -distr.log_prob(target)  # [B,S,D]
        nll_t = nll_per_entry.sum(dim=-1, keepdim=True)
        nll = (nll_t * loss_weights).sum(dim=1) / (loss_weights.sum(dim=1).clamp_min(1.0))
        nll = nll.mean()

        if self.sample_LH and self.LH_w > 0.0:
            # likelihood from sample-based Normal approx (parity with sample_LH=True)
            samp_mean = samples.mean(dim=0)
            samp_std = samples.std(dim=0).clamp_min(1e-6)
            distr_s = torch.distributions.Normal(samp_mean, samp_std)
            nll_per_entry = -distr_s.log_prob(target)
            nll_t = nll_per_entry.sum(dim=-1, keepdim=True)
            nll = (nll_t * loss_weights).sum(dim=1) / (loss_weights.sum(dim=1).clamp_min(1.0))
            nll = nll.mean()

        loss = self.CRPS_w * crps_full + self.LH_w * nll
        self.log_dict(
            {"train_crps": crps_full, "train_nll": nll, "train_loss": loss},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        (past_y, past_f, past_is_pad, past_obsv, fut_y, fut_f) = batch
        B = past_y.shape[0]
        idx = torch.arange(self.D, device=self.device).long().unsqueeze(0).repeat(B, 1)
        rnn_out, _, scale, _, _ = self.unroll_encoder(
            past_time=past_f,
            past_target=past_y,
            past_obsv=past_obsv,
            past_is_pad=past_is_pad,
            future_time=fut_f,
            future_target=fut_y,
            target_dim_indicator=idx,
        )
        distr, _ = self.build_distribution(rnn_out, scale)
        samples = distr.rsample((min(self.K_loss, 100),))[:, :, -self.pl :, :]  # [K,B,P,D]
        val_crps = crps_empirical_samples(samples, fut_y)
        self.log("val_crps", val_crps, prog_bar=True, on_step=False, on_epoch=True)
        return val_crps

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def predict_samples(
        self,
        past_y: torch.Tensor,
        past_f: torch.Tensor,
        past_obsv: torch.Tensor,
        past_is_pad: torch.Tensor,
        fut_f: torch.Tensor,
    ) -> np.ndarray:
        B = past_y.shape[0]
        assert B == 1, "Predicting single multivariate series as one batch."
        idx = torch.arange(self.D, device=self.device).long().unsqueeze(0).repeat(B, 1)

        _, state, scale, _, _ = self.unroll_encoder(
            past_time=past_f,
            past_target=past_y,
            past_obsv=past_obsv,
            past_is_pad=past_is_pad,
            future_time=None,
            future_target=None,
            target_dim_indicator=idx,
        )

        K = self.K_pred
        rep = lambda x: x.repeat(K, 1, 1)
        rep_scale = rep(scale)
        rep_idx = torch.arange(self.D, device=self.device).long().unsqueeze(0).repeat(K, 1)

        seq = past_y.repeat(K, 1, 1)
        h, c = state
        h = h.repeat(1, K, 1)
        c = c.repeat(1, K, 1)
        stateK = (h, c)

        samples_list = []
        for k in range(self.pl):
            lags = self.get_lagged_subsequences(seq, self.H + k, self.lags, 1)  # [K,1,D,I]
            lags_scaled = lags / rep_scale.unsqueeze(-1)
            input_lags = lags_scaled.reshape(K, 1, self.D * len(self.lags))
            emb = self.embed(rep_idx).unsqueeze(1).repeat(1, 1, 1, 1)  # [K,1,D,E]
            input_emb = emb.reshape(K, 1, self.D * self.embed_dim)
            x = torch.cat(
                [
                    input_lags,
                    input_emb,
                    fut_f[:, k : k + 1, :].repeat(K, 1, 1),
                ],
                dim=-1,
            )
            out, stateK = self.rnn(x, stateK)
            params = self.proj(out).view(K, 1, self.D, 2)
            mu_hat = params[..., 0]
            sig_hat = F.softplus(params[..., 1]) + 1e-6
            mu = mu_hat * rep_scale
            sigma = sig_hat * rep_scale
            new = mu + sigma * torch.randn_like(mu)  # [K,1,D]
            if self.coherent_pred_samples and (self.P is not None):
                new = new @ self.P.T
            samples_list.append(new)
            seq = torch.cat([seq, new], dim=1)
        samples = torch.cat(samples_list, dim=1).squeeze(1)  # [K,P,D]
        return samples.cpu().numpy()


# ------------------------------
# DataModule
# ------------------------------
class HierDataModule(pl.LightningDataModule):
    def __init__(
        self,
        bundle: "HierDataBundle",
        context_length: int,
        prediction_length: int,
        batch_size: int,
        pick_incomplete: bool,
        num_batches_per_epoch: int,
    ):
        super().__init__()
        self.bundle = bundle
        self.cl = context_length
        self.pl = prediction_length
        self.bs = batch_size
        self.pick_incomplete = pick_incomplete
        self.num_batches_per_epoch = num_batches_per_epoch

    def setup(self, stage: Optional[str] = None):
        # Train on series up to T - P (like hts_train)
        self.train_ds = MultiSeriesWindows(
            Y_all_TD=self.bundle.Y_all_TD[: -self.pl, :],
            time_feats_TF=self.bundle.time_feats_TF[: -self.pl, :],
            context_length=self.cl,
            prediction_length=self.pl,
            freq_str=self.bundle.freq_raw,  # DeepVAR mapping expects the same string GluonTS sees
            pick_incomplete=self.pick_incomplete,
            train=True,
        )
        # Validation on last cut of full series
        self.val_ds = MultiSeriesWindows(
            Y_all_TD=self.bundle.Y_all_TD,
            time_feats_TF=self.bundle.time_feats_TF,
            context_length=self.cl,
            prediction_length=self.pl,
            freq_str=self.bundle.freq_raw,
            pick_incomplete=self.pick_incomplete,
            train=False,
        )

    def train_dataloader(self):
        # ExpectedNumInstanceSampler(num_instances=1.0, ...) analogue:
        # replacement sampling with exactly batch_size * num_batches_per_epoch draws
        num_samples = self.bs * self.num_batches_per_epoch
        sampler = RandomSampler(self.train_ds, replacement=True, num_samples=num_samples)
        return DataLoader(
            self.train_ds,
            batch_size=self.bs,
            sampler=sampler,
            num_workers=0,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            sampler=SequentialSampler(self.val_ds),
            num_workers=0,
            drop_last=False,
        )


# ------------------------------
# Hier metrics helpers
# ------------------------------
def hierarchical_scrps(
    hier_idxs: List[np.ndarray], Y: np.ndarray, Yq_hat: np.ndarray, quantiles: np.ndarray
) -> List[float]:
    out = []
    for idxs in hier_idxs:
        y = Y[idxs, :]
        yq = Yq_hat[idxs, :, :]
        out.append(scaled_crps(y, yq, quantiles))
    return out


def hierarchical_msse(
    hier_idxs: List[np.ndarray], Y: np.ndarray, Y_hat: np.ndarray, Y_train: np.ndarray
) -> List[float]:
    out = []
    for idxs in hier_idxs:
        y = Y[idxs, :]
        yh = Y_hat[idxs, :]
        yt = Y_train[idxs, :]
        out.append(msse(y, yh, yt))
    return out


def hierarchical_rel_mse(
    hier_idxs: List[np.ndarray], Y: np.ndarray, Y_hat: np.ndarray, Y_train: np.ndarray
) -> List[float]:
    out = []
    for idxs in hier_idxs:
        y = Y[idxs, :]
        yh = Y_hat[idxs, :]
        yt = Y_train[idxs, :]
        out.append(rel_mse(y, yh, yt))
    return out


# ------------------------------
# Main
# ------------------------------
def main():
    seed_everything(42, workers=True)

    DATASET = "OldTourismLarge"  # "TourismSmall"
    LEVEL = np.arange(0, 100, 2)
    qs = [[50 - lv / 2, 50 + lv / 2] for lv in LEVEL]
    QUANTILES = np.sort(np.concatenate(qs) / 100.0)

    cfg = CONFIGS[DATASET]
    bundle = load_hierarchical_dataset_all_nodes(DATASET, directory="./data")

    dm = HierDataModule(
        bundle=bundle,
        context_length=cfg["context_length"],
        prediction_length=bundle.horizon,
        batch_size=cfg["batch_size"],
        pick_incomplete=cfg.get("pick_incomplete", False),
        num_batches_per_epoch=cfg.get("num_batches_per_epoch", 50),
    )
    dm.setup()

    model = DeepVARHierPL(
        target_dim=bundle.S_df.shape[0],
        context_length=cfg["context_length"],
        prediction_length=bundle.horizon,
        freq_str=bundle.freq_raw,  # must be supported by DeepVAR mapping
        num_layers=cfg["num_layers"],
        num_cells=cfg["num_cells"],
        learning_rate=cfg["learning_rate"],
        dropout_rate=0.1,
        coherent_train_samples=cfg["coherent_train_samples"],
        coherent_pred_samples=cfg["coherent_pred_samples"],
        warmstart_epoch_frac=cfg["warmstart_epoch_frac"],
        num_samples_for_loss=cfg["num_samples_for_loss"],
        num_parallel_samples=cfg["num_parallel_samples"],
        CRPS_weight=cfg["CRPS_weight"],
        likelihood_weight=cfg["likelihood_weight"],
        sample_LH=cfg["sample_LH"],
        num_batches_per_epoch=cfg["num_batches_per_epoch"],
        epochs=cfg["epochs"],
        S=bundle.A_all_from_bottom,
        D_weight=None,
        time_feat_dim=bundle.time_feats_TF.shape[1],
        embedding_dimension=5,
        scaling=cfg["scaling"],
    )

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, dm)

    # ---- Prediction on last cut ----
    T, D = bundle.Y_all_TD.shape
    H = model.H
    P = model.pl
    t = T - P
    start = max(0, t - H)

    past_y = torch.from_numpy(bundle.Y_all_TD[start:t, :]).float()
    past_f = torch.from_numpy(bundle.time_feats_TF[start:t, :]).float()
    if past_y.shape[0] < H:
        pad = H - past_y.shape[0]
        past_y = torch.cat([torch.zeros(pad, D), past_y], dim=0)
        past_f = torch.cat([torch.zeros(pad, past_f.shape[1]), past_f], dim=0)
    past_y = past_y.unsqueeze(0).to(model.device)  # [1,H,D]
    past_f = past_f.unsqueeze(0).to(model.device)  # [1,H,F]
    past_is_pad = torch.zeros(1, H, device=model.device)
    past_obsv = torch.ones(1, H, D, device=model.device)

    fut_f = (
        torch.from_numpy(bundle.time_feats_TF[t : t + P, :])
        .float()
        .unsqueeze(0)
        .to(model.device)
    )
    samples_all_nodes = model.predict_samples(
        past_y, past_f, past_obsv, past_is_pad, fut_f
    )  # [K,P,D]

    # Mean and quantiles
    Y_hat_all = samples_all_nodes.mean(axis=0).T  # [D,P]
    QUANT = QUANTILES
    Yq_all = (
        np.quantile(samples_all_nodes, q=QUANT, axis=0).transpose(2, 1, 0)
    )  # [D,P,Q]

    # Ground truth & metrics
    Y_all_full = (
        bundle.A_all_from_bottom @ bundle.Y_bottom_TB.T
    ).astype(np.float32)  # [D,T]
    Y_train = Y_all_full[:, : T - P]
    Y_test = Y_all_full[:, T - P :]

    scrps = hierarchical_scrps(bundle.hier_idxs, Y=Y_test, Yq_hat=Yq_all, quantiles=QUANT)
    msse_vals = hierarchical_msse(bundle.hier_idxs, Y=Y_test, Y_hat=Y_hat_all, Y_train=Y_train)
    relmse_vals = hierarchical_rel_mse(bundle.hier_idxs, Y=Y_test, Y_hat=Y_hat_all, Y_train=Y_train)

    results_df = pd.DataFrame(dict(level=bundle.hier_levels))
    results_df["scrps"] = scrps
    results_df["rel_mse"] = relmse_vals
    results_df["msse"] = msse_vals
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", None)
    print("\n== Results ==")
    print(results_df)


if __name__ == "__main__":
    main()
