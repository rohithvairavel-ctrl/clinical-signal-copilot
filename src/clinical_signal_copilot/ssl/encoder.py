"""Lightweight temporal encoder (numpy) + deterministic window featurization."""

from __future__ import annotations

import numpy as np


def flatten_window(x: np.ndarray) -> np.ndarray:
    """(C, T) → 1-D vector."""
    return np.asarray(x, dtype=float).ravel()


def featurize_windows(X: np.ndarray) -> np.ndarray:
    """Hand-crafted multi-channel features as a strong lean baseline.

    X: (N, C, T) → (N, F)
    """
    X = np.asarray(X, dtype=float)
    N, C, T = X.shape
    feats = []
    for i in range(N):
        row = []
        for c in range(C):
            s = X[i, c]
            d = np.diff(s)
            row.extend(
                [
                    float(np.mean(s)),
                    float(np.std(s)),
                    float(np.min(s)),
                    float(np.max(s)),
                    float(np.percentile(s, 25)),
                    float(np.percentile(s, 75)),
                    float(np.mean(np.abs(d))),
                    float(np.std(d)),
                    float(np.sqrt(np.mean(s**2))),
                    float(np.mean(s**2)),
                ]
            )
            # Coarse spectral energy via rFFT bands
            spec = np.abs(np.fft.rfft(s - s.mean()))
            bands = np.array_split(spec, 4)
            row.extend([float(np.sum(b**2)) for b in bands])
        # Cross-channel correlations
        for a in range(C):
            for b in range(a + 1, C):
                ca, cb = X[i, a], X[i, b]
                denom = (np.std(ca) * np.std(cb)) + 1e-8
                row.append(float(np.mean((ca - ca.mean()) * (cb - cb.mean())) / denom))
        feats.append(row)
    return np.asarray(feats, dtype=float)


class TemporalEncoder:
    """Small MLP encoder trained via contrastive SGD (numpy).

    Maps featurized windows → projection embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        out_dim: int = 32,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.W1 = rng.normal(0, 0.05, size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.05, size=(hidden, out_dim))
        self.b2 = np.zeros(out_dim)

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0.0)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return L2-normalized embeddings and cache for backprop."""
        h_pre = x @ self.W1 + self.b1
        h = self._relu(h_pre)
        z = h @ self.W2 + self.b2
        nrm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        z_n = z / nrm
        cache = {"x": x, "h_pre": h_pre, "h": h, "z": z, "nrm": nrm, "z_n": z_n}
        return z_n, cache

    def encode(self, x: np.ndarray) -> np.ndarray:
        z, _ = self.forward(x)
        return z

    def sgd_step(self, cache: dict, d_zn: np.ndarray, lr: float = 1e-2, weight_decay: float = 1e-4) -> None:
        """Backprop through normalize → linear → ReLU → linear given dL/dz_n."""
        z, nrm, z_n = cache["z"], cache["nrm"], cache["z_n"]
        # dL/dz through L2 normalize
        # z_n = z / ||z|| ; Jacobian: (I - z_n z_n^T) / ||z||
        dot = np.sum(d_zn * z_n, axis=1, keepdims=True)
        dz = (d_zn - z_n * dot) / nrm
        h = cache["h"]
        dW2 = h.T @ dz + weight_decay * self.W2
        db2 = dz.sum(axis=0)
        dh = dz @ self.W2.T
        dh_pre = dh * (cache["h_pre"] > 0)
        x = cache["x"]
        dW1 = x.T @ dh_pre + weight_decay * self.W1
        db1 = dh_pre.sum(axis=0)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
