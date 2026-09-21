"""Split conformal classification for binary event detection.

Produces set-valued predictions {0}, {1}, or {0,1} with finite-sample
marginal coverage guarantees under exchangeability (research demo only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalConfig:
    alpha: float = 0.1  # target miscoverage
    seed: int = 42


def _nonconformity(proba_pos: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Hinge-style score: 1 - P(true class)."""
    p1 = np.asarray(proba_pos, dtype=float)
    y = np.asarray(y).astype(int).ravel()
    p_true = np.where(y == 1, p1, 1.0 - p1)
    return 1.0 - p_true


class SplitConformalClassifier:
    """Calibrate a threshold on held-out scores; predict label sets."""

    def __init__(self, config: ConformalConfig | None = None) -> None:
        self.config = config or ConformalConfig()
        self.qhat_: float | None = None
        self.cal_scores_: np.ndarray | None = None

    def calibrate(self, proba_pos: np.ndarray, y_cal: np.ndarray) -> "SplitConformalClassifier":
        scores = _nonconformity(proba_pos, y_cal)
        self.cal_scores_ = scores
        n = len(scores)
        alpha = self.config.alpha
        # Finite-sample corrected quantile
        level = np.ceil((n + 1) * (1 - alpha)) / n
        level = float(np.clip(level, 0.0, 1.0))
        self.qhat_ = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_sets(self, proba_pos: np.ndarray) -> list[set]:
        if self.qhat_ is None:
            raise RuntimeError("Call calibrate() first.")
        p1 = np.asarray(proba_pos, dtype=float).ravel()
        sets = []
        for p in p1:
            s: set[int] = set()
            # Include label if its nonconformity <= qhat
            if (1.0 - (1.0 - p)) <= self.qhat_ + 1e-12:  # score for y=0: 1-(1-p)=p
                # nonconformity for predicting class 0 uses p_true=1-p ⇒ score=p
                pass
            score0 = p  # 1 - P(y=0) = 1 - (1-p) = p
            score1 = 1.0 - p
            if score0 <= self.qhat_ + 1e-12:
                s.add(0)
            if score1 <= self.qhat_ + 1e-12:
                s.add(1)
            if not s:
                # Never return empty: include argmax
                s.add(1 if p >= 0.5 else 0)
            sets.append(s)
        return sets

    def predict_point(self, proba_pos: np.ndarray) -> np.ndarray:
        """Point prediction = singleton if possible else argmax."""
        sets = self.predict_sets(proba_pos)
        p1 = np.asarray(proba_pos, dtype=float).ravel()
        out = []
        for s, p in zip(sets, p1):
            if len(s) == 1:
                out.append(next(iter(s)))
            else:
                out.append(1 if p >= 0.5 else 0)
        return np.asarray(out, dtype=np.int8)


def coverage_report(
    sets: list[set],
    y_true: np.ndarray,
    alpha: float,
) -> dict:
    """Empirical coverage, average set size, singleton rate."""
    y = np.asarray(y_true).astype(int).ravel()
    covered = np.array([int(yi in s) for yi, s in zip(y, sets)], dtype=float)
    sizes = np.array([len(s) for s in sets], dtype=float)
    return {
        "target_coverage": float(1 - alpha),
        "empirical_coverage": float(covered.mean()) if len(covered) else float("nan"),
        "avg_set_size": float(sizes.mean()) if len(sizes) else float("nan"),
        "singleton_rate": float(np.mean(sizes == 1)) if len(sizes) else float("nan"),
        "n": int(len(y)),
        "q_diagnostic_ok": bool(
            len(covered) == 0
            or covered.mean() >= (1 - alpha) - 0.08  # soft check for small-n demos
        ),
    }
