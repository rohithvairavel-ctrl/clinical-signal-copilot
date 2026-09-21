"""Supervised detection head on SSL or raw features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..ssl.encoder import featurize_windows


@dataclass
class DetectorConfig:
    model: str = "logreg"  # logreg | mlp
    C: float = 1.0
    max_iter: int = 400
    seed: int = 42
    class_weight: Optional[str] = "balanced"


class EventDetector:
    """Binary event / peak-window classifier."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.model_: Pipeline | None = None
        self.use_external_features_: bool = False

    def _build(self) -> Pipeline:
        cfg = self.config
        if cfg.model == "mlp":
            clf = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=cfg.max_iter,
                random_state=cfg.seed,
                early_stopping=True,
                validation_fraction=0.15,
            )
        else:
            clf = LogisticRegression(
                C=cfg.C,
                max_iter=cfg.max_iter,
                class_weight=cfg.class_weight,
                random_state=cfg.seed,
                solver="lbfgs",
            )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> "EventDetector":
        """Fit on windows (N,C,T) or precomputed embeddings."""
        self.model_ = self._build()
        if embeddings is not None:
            self.use_external_features_ = True
            feats = np.asarray(embeddings, dtype=float)
        else:
            self.use_external_features_ = False
            feats = featurize_windows(X)
        self.model_.fit(feats, np.asarray(y).ravel())
        return self

    def predict_proba(self, X: np.ndarray, embeddings: np.ndarray | None = None) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Detector not fitted.")
        if embeddings is not None:
            feats = np.asarray(embeddings, dtype=float)
        else:
            feats = featurize_windows(X)
        proba = self.model_.predict_proba(feats)
        # Probability of positive class
        classes = list(self.model_.named_steps["clf"].classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return proba[:, -1]

    def predict(self, X: np.ndarray, embeddings: np.ndarray | None = None, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X, embeddings) >= threshold).astype(np.int8)
