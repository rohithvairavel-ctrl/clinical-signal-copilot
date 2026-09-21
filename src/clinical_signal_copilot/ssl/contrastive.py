"""SimCLR-style NT-Xent contrastive pretraining on unlabeled windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoder import TemporalEncoder, featurize_windows
from .views import augment_pair


@dataclass
class ContrastiveConfig:
    epochs: int = 8
    batch_size: int = 64
    temperature: float = 0.2
    lr: float = 5e-3
    hidden: int = 64
    out_dim: int = 32
    seed: int = 42
    max_windows: int = 512


def nt_xent_loss(
    z1: np.ndarray,
    z2: np.ndarray,
    temperature: float = 0.2,
) -> tuple[float, np.ndarray, np.ndarray]:
    """NT-Xent for paired embeddings. Returns loss and grads wrt z1, z2."""
    B = z1.shape[0]
    z = np.concatenate([z1, z2], axis=0)  # (2B, D)
    sim = (z @ z.T) / temperature
    # Mask self-similarity
    mask = np.eye(2 * B, dtype=bool)
    sim = np.where(mask, -1e9, sim)
    # Positives: i ↔ i+B
    labels = np.concatenate([np.arange(B, 2 * B), np.arange(0, B)])
    # Softmax cross-entropy
    sim_shift = sim - sim.max(axis=1, keepdims=True)
    exp = np.exp(sim_shift)
    probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
    loss = float(-np.mean(np.log(probs[np.arange(2 * B), labels] + 1e-12)))
    # Gradient of CE wrt sim logits
    d_logits = probs
    d_logits[np.arange(2 * B), labels] -= 1.0
    d_logits /= 2 * B
    d_logits /= temperature
    dz = d_logits @ z
    dz1, dz2 = dz[:B], dz[B:]
    return loss, dz1, dz2


class ContrastivePretrainer:
    """Pretrain TemporalEncoder with SimCLR/CLOCS-style paired views."""

    def __init__(self, config: ContrastiveConfig | None = None) -> None:
        self.config = config or ContrastiveConfig()
        self.encoder: TemporalEncoder | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        self.history_: list[float] = []

    def _featurize_raw(self, windows: np.ndarray) -> np.ndarray:
        return featurize_windows(windows)

    def fit(self, windows: np.ndarray) -> "ContrastivePretrainer":
        """windows: (N, C, T) unlabeled."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        N = windows.shape[0]
        idx = np.arange(N)
        if N > cfg.max_windows:
            idx = rng.choice(N, size=cfg.max_windows, replace=False)
            windows = windows[idx]
            N = windows.shape[0]

        # Fit normalization on unaugmented features
        base_feat = self._featurize_raw(windows)
        self.feature_mean_ = base_feat.mean(axis=0)
        self.feature_std_ = base_feat.std(axis=0) + 1e-6
        in_dim = base_feat.shape[1]
        self.encoder = TemporalEncoder(in_dim, cfg.hidden, cfg.out_dim, seed=cfg.seed)
        self.history_.clear()

        for epoch in range(cfg.epochs):
            perm = rng.permutation(N)
            losses = []
            for start in range(0, N, cfg.batch_size):
                batch_idx = perm[start : start + cfg.batch_size]
                if len(batch_idx) < 4:
                    continue
                v1_list, v2_list = [], []
                for i in batch_idx:
                    a, b = augment_pair(windows[i], rng)
                    v1_list.append(a)
                    v2_list.append(b)
                f1 = (self._featurize_raw(np.stack(v1_list)) - self.feature_mean_) / self.feature_std_
                f2 = (self._featurize_raw(np.stack(v2_list)) - self.feature_mean_) / self.feature_std_
                z1, c1 = self.encoder.forward(f1)
                z2, c2 = self.encoder.forward(f2)
                loss, dz1, dz2 = nt_xent_loss(z1, z2, cfg.temperature)
                self.encoder.sgd_step(c1, dz1, lr=cfg.lr)
                self.encoder.sgd_step(c2, dz2, lr=cfg.lr)
                losses.append(loss)
            ep = float(np.mean(losses)) if losses else float("nan")
            self.history_.append(ep)
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self.encoder is None or self.feature_mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        feat = (self._featurize_raw(windows) - self.feature_mean_) / self.feature_std_
        return self.encoder.encode(feat)

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        self.fit(windows)
        return self.transform(windows)
