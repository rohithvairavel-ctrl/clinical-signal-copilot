"""Synthetic multi-channel physiological signal generation and SQI."""

from .synthetic import SyntheticConfig, SyntheticDataset, generate_dataset
from .quality import signal_quality_index, window_sqi

__all__ = [
    "SyntheticConfig",
    "SyntheticDataset",
    "generate_dataset",
    "signal_quality_index",
    "window_sqi",
]
