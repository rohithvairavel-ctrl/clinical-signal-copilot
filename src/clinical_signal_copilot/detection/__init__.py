"""Supervised event detection head and evaluation metrics."""

from .supervised import EventDetector, DetectorConfig
from .metrics import detection_metrics, match_events

__all__ = ["EventDetector", "DetectorConfig", "detection_metrics", "match_events"]
