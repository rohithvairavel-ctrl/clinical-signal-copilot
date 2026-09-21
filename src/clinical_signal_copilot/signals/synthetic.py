"""Fully synthetic PhysioNet-style multi-channel physiological signals.

Generates ECG-like waveforms with QRS peaks, optional arrhythmia-like segments,
channel-wise noise/quality variation, event labels, and a synthetic protected
subgroup attribute. No real patient data — no licensing issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SyntheticConfig:
    n_records: int = 40
    n_channels: int = 3
    duration_s: float = 20.0
    fs: float = 250.0
    hr_bpm: tuple[float, float] = (55.0, 95.0)
    noise_std: tuple[float, float] = (0.02, 0.25)
    arrhythmia_prob: float = 0.35
    subgroup_names: tuple[str, ...] = ("A", "B")
    subgroup_noise_boost: dict = field(default_factory=lambda: {"A": 1.0, "B": 1.7})
    seed: int = 42


@dataclass
class SyntheticDataset:
    """Container for synthetic multi-channel recordings."""

    signals: np.ndarray  # (N, C, T)
    event_indices: list  # list of arrays of sample indices (QRS / event peaks)
    event_labels: list  # per-record binary labels for arrhythmia-like segments (T,)
    subgroups: np.ndarray  # (N,) string/object codes
    sqi: np.ndarray  # (N, C) mean SQI
    fs: float
    config: SyntheticConfig
    meta: dict = field(default_factory=dict)

    @property
    def n_records(self) -> int:
        return int(self.signals.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.signals.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.signals.shape[2])


def _qrs_template(fs: float, width_ms: float = 100.0) -> np.ndarray:
    """Simple bipolar QRS-like impulse (differentiated Gaussian)."""
    n = max(5, int(fs * width_ms / 1000.0))
    t = np.linspace(-2.5, 2.5, n)
    g = np.exp(-0.5 * t**2)
    qrs = -np.gradient(g)
    qrs = qrs / (np.max(np.abs(qrs)) + 1e-8)
    return qrs.astype(float)


def _place_beats(
    n_samples: int,
    fs: float,
    hr_bpm: float,
    rng: np.random.Generator,
    arrhythmic: bool,
) -> np.ndarray:
    """Return beat peak sample indices."""
    rr = 60.0 / hr_bpm
    peaks = []
    t = rng.uniform(0.2, 0.6)
    while t * fs < n_samples - 5:
        peaks.append(int(t * fs))
        jitter = rng.normal(0, 0.03 * rr)
        if arrhythmic and rng.random() < 0.25:
            jitter += rng.choice([-1, 1]) * rng.uniform(0.12, 0.28) * rr
        t += max(0.25, rr + jitter)
    return np.asarray(peaks, dtype=int)


def _synthesize_record(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    subgroup: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = int(cfg.duration_s * cfg.fs)
    C = cfg.n_channels
    hr = float(rng.uniform(*cfg.hr_bpm))
    arrhythmic = bool(rng.random() < cfg.arrhythmia_prob)
    peaks = _place_beats(n_samples, cfg.fs, hr, rng, arrhythmic)
    qrs = _qrs_template(cfg.fs)

    sig = np.zeros((C, n_samples), dtype=float)
    t = np.arange(n_samples) / cfg.fs
    for c in range(C):
        wander = 0.08 * np.sin(2 * np.pi * rng.uniform(0.05, 0.25) * t + rng.uniform(0, 6))
        gain = rng.uniform(0.7, 1.3) * (1.0 if c == 0 else rng.uniform(0.5, 1.1))
        polarity = 1.0 if c < 2 else float(rng.choice([-1.0, 1.0]))
        for p in peaks:
            a, b = p - len(qrs) // 2, p - len(qrs) // 2 + len(qrs)
            if a < 0 or b > n_samples:
                continue
            sig[c, a:b] += polarity * gain * qrs
        sig[c] += wander

    event_lab = np.zeros(n_samples, dtype=np.int8)
    if arrhythmic and len(peaks) > 4:
        start = int(peaks[len(peaks) // 3])
        end = min(n_samples, start + int(2.5 * cfg.fs))
        event_lab[start:end] = 1
        band = slice(start, end)
        for c in range(C):
            sig[c, band] += 0.35 * np.sin(2 * np.pi * 3.0 * t[band])
            sig[c, band] *= rng.uniform(0.6, 0.85)

    noise_boost = float(cfg.subgroup_noise_boost.get(subgroup, 1.0))
    base_noise = float(rng.uniform(*cfg.noise_std)) * noise_boost
    channel_noise = base_noise * rng.uniform(0.6, 1.6, size=C)
    for c in range(C):
        sig[c] += rng.normal(0, channel_noise[c], size=n_samples)
        if rng.random() < 0.08 * noise_boost:
            d0 = rng.integers(0, max(1, n_samples - int(0.4 * cfg.fs)))
            d1 = d0 + int(rng.uniform(0.15, 0.5) * cfg.fs)
            sig[c, d0:d1] *= rng.uniform(0.0, 0.15)

    from .quality import signal_quality_index

    sqi = np.array([signal_quality_index(sig[c], cfg.fs) for c in range(C)], dtype=float)
    return sig, peaks, event_lab, sqi


def generate_dataset(config: Optional[SyntheticConfig] = None) -> SyntheticDataset:
    """Generate a full synthetic multi-channel dataset."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    n_samples = int(cfg.duration_s * cfg.fs)
    signals = np.zeros((cfg.n_records, cfg.n_channels, n_samples), dtype=float)
    peaks_list: list = []
    labels_list: list = []
    sqi = np.zeros((cfg.n_records, cfg.n_channels), dtype=float)
    subgroups = np.empty(cfg.n_records, dtype=object)

    names = list(cfg.subgroup_names)
    for i in range(cfg.n_records):
        sg = names[i % len(names)]
        subgroups[i] = sg
        sig, peaks, lab, s = _synthesize_record(cfg, rng, sg)
        signals[i] = sig
        peaks_list.append(peaks)
        labels_list.append(lab)
        sqi[i] = s

    return SyntheticDataset(
        signals=signals,
        event_indices=peaks_list,
        event_labels=labels_list,
        subgroups=subgroups,
        sqi=sqi,
        fs=cfg.fs,
        config=cfg,
        meta={"generator": "clinical_signal_copilot.synthetic", "synthetic": True},
    )


def extract_windows(
    dataset: SyntheticDataset,
    window_s: float = 0.8,
    hop_s: float = 0.4,
    label_mode: str = "peak",
    max_neg_per_rec: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build windows with both positive and negative examples.

    label_mode:
      - 'peak': positives centered on QRS peaks; negatives midway between peaks
        (or sliding windows with no peak in the central third).
      - 'arrhythmia': sliding windows labeled by arrhythmia-segment overlap.
    """
    W = int(window_s * dataset.fs)
    H = max(1, int(hop_s * dataset.fs))
    half = W // 2
    rng = np.random.default_rng(seed)
    xs, ys, sgs, rids = [], [], [], []

    for i in range(dataset.n_records):
        T = dataset.n_samples
        peaks = np.asarray(dataset.event_indices[i], dtype=int)
        lab = dataset.event_labels[i]
        sig = dataset.signals[i]

        if label_mode == "arrhythmia":
            for start in range(0, T - W + 1, H):
                end = start + W
                y = int(lab[start:end].mean() > 0.3)
                xs.append(sig[:, start:end])
                ys.append(y)
                sgs.append(dataset.subgroups[i])
                rids.append(i)
            continue

        # --- peak mode: centered positives + inter-beat negatives ---
        pos_centers = []
        for p in peaks:
            if p - half >= 0 and p - half + W <= T:
                pos_centers.append(p)
                xs.append(sig[:, p - half : p - half + W])
                ys.append(1)
                sgs.append(dataset.subgroups[i])
                rids.append(i)

        neg_centers = []
        if len(peaks) >= 2:
            for a, b in zip(peaks[:-1], peaks[1:]):
                mid = (a + b) // 2
                if mid - half >= 0 and mid - half + W <= T:
                    # keep only if no peak inside window
                    if not np.any(np.abs(peaks - mid) < half * 0.9):
                        neg_centers.append(mid)
        # Extra random negatives away from peaks
        n_target = len(pos_centers) if max_neg_per_rec is None else max_neg_per_rec
        attempts = 0
        while len(neg_centers) < max(n_target, 1) and attempts < n_target * 8 + 10:
            attempts += 1
            c = int(rng.integers(half, max(half + 1, T - half)))
            if len(peaks) and np.min(np.abs(peaks - c)) < half:
                continue
            neg_centers.append(c)
        if max_neg_per_rec is not None:
            neg_centers = neg_centers[:max_neg_per_rec]
        else:
            neg_centers = neg_centers[: max(len(pos_centers), 1)]

        for c in neg_centers:
            xs.append(sig[:, c - half : c - half + W])
            ys.append(0)
            sgs.append(dataset.subgroups[i])
            rids.append(i)

    if not xs:
        raise RuntimeError("No windows extracted; check duration/window settings.")
    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int8)
    sg = np.asarray(sgs, dtype=object)
    rid = np.asarray(rids, dtype=int)
    return X, y, sg, rid
