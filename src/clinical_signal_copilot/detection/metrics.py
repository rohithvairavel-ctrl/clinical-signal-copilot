"""Event detection metrics with temporal tolerance matching (PhysioNet-style)."""

from __future__ import annotations

import numpy as np


def match_events(
    true_peaks: np.ndarray,
    pred_peaks: np.ndarray,
    tolerance: int,
) -> tuple[int, int, int]:
    """Greedy match predicted peaks to reference within ±tolerance samples.

    Returns (TP, FP, FN).
    """
    true_peaks = np.asarray(true_peaks, dtype=int)
    pred_peaks = np.asarray(pred_peaks, dtype=int)
    if true_peaks.size == 0 and pred_peaks.size == 0:
        return 0, 0, 0
    if true_peaks.size == 0:
        return 0, int(pred_peaks.size), 0
    if pred_peaks.size == 0:
        return 0, 0, int(true_peaks.size)

    used = np.zeros(true_peaks.size, dtype=bool)
    tp = 0
    for p in np.sort(pred_peaks):
        dists = np.abs(true_peaks - p)
        dists[used] = tolerance + 1
        j = int(np.argmin(dists))
        if dists[j] <= tolerance:
            used[j] = True
            tp += 1
    fp = int(pred_peaks.size - tp)
    fn = int(true_peaks.size - tp)
    return tp, fp, fn


def detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: int | None = None,
    true_peaks: list | None = None,
    pred_peaks: list | None = None,
) -> dict:
    """Compute Sens / PPV / F1.

    If peak lists + tolerance are provided, use event matching; else treat as
    binary window classification.
    """
    if true_peaks is not None and pred_peaks is not None and tolerance is not None:
        tp = fp = fn = 0
        for t, p in zip(true_peaks, pred_peaks):
            a, b, c = match_events(t, p, tolerance)
            tp += a
            fp += b
            fn += c
    else:
        y_true = np.asarray(y_true).astype(int).ravel()
        y_pred = np.asarray(y_pred).astype(int).ravel()
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sens = tp / (tp + fn + 1e-12)
    ppv = tp / (tp + fp + 1e-12)
    f1 = 2 * sens * ppv / (sens + ppv + 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(sens),
        "ppv": float(ppv),
        "f1": float(f1),
    }
