"""Conformal prediction for set-valued detection with coverage diagnostics."""

from .predictor import ConformalConfig, SplitConformalClassifier, coverage_report

__all__ = ["ConformalConfig", "SplitConformalClassifier", "coverage_report"]
