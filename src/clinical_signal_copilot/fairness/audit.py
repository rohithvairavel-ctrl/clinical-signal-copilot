"""Fairness gap reporting + sample-reweight mitigation sketch.

Protected attribute is fully synthetic (subgroup A/B), intended only to
illustrate disparity diagnostics — not real demographics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detection.metrics import detection_metrics


def fairness_gaps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroups: np.ndarray,
    metric_keys: tuple[str, ...] = ("sensitivity", "ppv", "f1"),
) -> dict:
    """Per-subgroup metrics and max-minus-min gaps."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    subgroups = np.asarray(subgroups)
    groups = sorted(set(subgroups.tolist()))
    per = {}
    for g in groups:
        m = subgroups == g
        per[str(g)] = detection_metrics(y_true[m], y_pred[m])
        per[str(g)]["n"] = int(m.sum())
    gaps = {}
    for k in metric_keys:
        vals = [per[str(g)][k] for g in groups]
        gaps[k] = float(max(vals) - min(vals)) if vals else float("nan")
    return {"groups": per, "gaps": gaps, "group_names": [str(g) for g in groups]}


def reweight_mitigation(
    subgroups: np.ndarray,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse-frequency sample weights (optionally balanced within label).

    Sketch mitigation: upweight underrepresented subgroup×label cells.
    """
    subgroups = np.asarray(subgroups)
    n = len(subgroups)
    if y is None:
        keys = subgroups
    else:
        y = np.asarray(y).ravel()
        keys = np.array([f"{s}|{yi}" for s, yi in zip(subgroups, y)], dtype=object)
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    w = counts[inv].astype(float)
    weights = (n / w) / len(counts)
    return weights.astype(float)


@dataclass
class FairnessAuditor:
    """Convenience wrapper around gap computation + mitigation weights."""

    def audit(self, y_true, y_pred, subgroups) -> dict:
        return fairness_gaps(y_true, y_pred, subgroups)

    def mitigation_weights(self, subgroups, y=None) -> np.ndarray:
        return reweight_mitigation(subgroups, y)
