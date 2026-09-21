import numpy as np

from clinical_signal_copilot.signals.quality import signal_quality_index, window_sqi
from clinical_signal_copilot.signals.synthetic import SyntheticConfig, generate_dataset, extract_windows


def test_generate_dataset_shapes(small_dataset):
    ds = small_dataset
    assert ds.signals.ndim == 3
    assert ds.n_records == 10
    assert ds.n_channels == 3
    assert ds.sqi.shape == (10, 3)
    assert set(ds.subgroups.tolist()) <= {"A", "B"}
    assert all(len(p) > 0 for p in ds.event_indices)


def test_reproducibility():
    a = generate_dataset(SyntheticConfig(n_records=4, duration_s=4.0, seed=7))
    b = generate_dataset(SyntheticConfig(n_records=4, duration_s=4.0, seed=7))
    np.testing.assert_array_equal(a.signals, b.signals)


def test_sqi_flatline_and_range():
    assert signal_quality_index(np.zeros(100)) == 0.0
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0, 20, 200)) + 0.05 * rng.normal(size=200)
    s = signal_quality_index(x)
    assert 0.0 <= s <= 1.0
    w = window_sqi(np.stack([x, x * 0.5]))
    assert w.shape == (2,)


def test_extract_windows(windows):
    X, y, sg, rid = windows
    assert X.ndim == 3
    assert len(y) == len(sg) == len(rid) == X.shape[0]
    assert set(np.unique(y)).issubset({0, 1})
