"""Shared pytest fixtures — seeded for determinism."""

from __future__ import annotations

import numpy as np
import pytest

from clinical_signal_copilot.signals.synthetic import SyntheticConfig, generate_dataset, extract_windows


@pytest.fixture(scope="session")
def seed() -> int:
    return 42


@pytest.fixture(scope="session")
def small_dataset(seed):
    cfg = SyntheticConfig(
        n_records=10,
        n_channels=3,
        duration_s=6.0,
        fs=125.0,
        seed=seed,
    )
    return generate_dataset(cfg)


@pytest.fixture(scope="session")
def windows(small_dataset):
    X, y, sg, rid = extract_windows(small_dataset, window_s=0.8, hop_s=0.4, label_mode="peak")
    return X, y, sg, rid
