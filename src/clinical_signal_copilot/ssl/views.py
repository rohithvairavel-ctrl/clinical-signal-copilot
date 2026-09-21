"""CLOCS / SimCLR-style temporal + channel augmentation views."""

from __future__ import annotations

import numpy as np


def _temporal_crop_resize(x: np.ndarray, rng: np.random.Generator, crop_frac: float = 0.8) -> np.ndarray:
    """Crop a contiguous fraction then linearly resample back to original length."""
    C, T = x.shape
    crop = max(4, int(T * crop_frac))
    start = int(rng.integers(0, T - crop + 1))
    piece = x[:, start : start + crop]
    # Resample each channel
    src = np.linspace(0, 1, crop)
    dst = np.linspace(0, 1, T)
    out = np.zeros_like(x)
    for c in range(C):
        out[c] = np.interp(dst, src, piece[c])
    return out


def _channel_drop(x: np.ndarray, rng: np.random.Generator, drop_prob: float = 0.3) -> np.ndarray:
    out = x.copy()
    C = out.shape[0]
    if C <= 1:
        return out
    mask = rng.random(C) < drop_prob
    if mask.all():
        mask[int(rng.integers(0, C))] = False
    out[mask] = 0.0
    return out


def _add_noise(x: np.ndarray, rng: np.random.Generator, scale: float = 0.05) -> np.ndarray:
    amp = float(np.std(x) + 1e-8)
    return x + rng.normal(0, scale * amp, size=x.shape)


def _time_shift(x: np.ndarray, rng: np.random.Generator, max_frac: float = 0.1) -> np.ndarray:
    T = x.shape[1]
    shift = int(rng.integers(-int(T * max_frac), int(T * max_frac) + 1))
    return np.roll(x, shift, axis=1)


def augment_once(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random composition of CLOCS-style views to window (C, T)."""
    y = x.astype(float, copy=True)
    if rng.random() < 0.8:
        y = _temporal_crop_resize(y, rng, crop_frac=float(rng.uniform(0.7, 0.95)))
    if rng.random() < 0.7:
        y = _time_shift(y, rng)
    if rng.random() < 0.5:
        y = _channel_drop(y, rng)
    if rng.random() < 0.8:
        y = _add_noise(y, rng, scale=float(rng.uniform(0.02, 0.12)))
    return y


def augment_pair(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two correlated views of the same window."""
    return augment_once(x, rng), augment_once(x, rng)
