"""Signal quality index (SQI) heuristics for multi-channel windows."""

from __future__ import annotations

import numpy as np


def signal_quality_index(x: np.ndarray, fs: float = 250.0) -> float:
    """Scalar SQI in [0, 1] for a 1-D channel.

    Combines amplitude stability, high-frequency noise ratio, and flatline detection.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 8:
        return 0.0
    amp = float(np.std(x))
    if amp < 1e-8:
        return 0.0  # flatline
    # High-frequency noise proxy via successive differences
    dx = np.diff(x)
    hf = float(np.std(dx)) / (amp + 1e-8)
    hf_score = float(np.clip(1.0 - max(0.0, hf - 0.8) / 2.0, 0.0, 1.0))
    # Amplitude in a plausible physio range (synthetic units)
    amp_score = float(np.clip(1.0 - abs(np.log10(amp + 1e-8) - 0.0) / 2.0, 0.0, 1.0))
    # Kurtosis-ish peakedness (QRS-like content)
    z = (x - x.mean()) / (amp + 1e-8)
    kurt = float(np.mean(z**4))
    kurt_score = float(np.clip((kurt - 1.5) / 6.0, 0.0, 1.0))
    return float(np.clip(0.4 * amp_score + 0.35 * hf_score + 0.25 * kurt_score, 0.0, 1.0))


def window_sqi(window: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """Per-channel SQI for window shaped (C, T) or (T,)."""
    w = np.asarray(window, dtype=float)
    if w.ndim == 1:
        return np.array([signal_quality_index(w, fs)], dtype=float)
    return np.array([signal_quality_index(w[c], fs) for c in range(w.shape[0])], dtype=float)
