import numpy as np

from clinical_signal_copilot.conformal.predictor import (
    ConformalConfig,
    SplitConformalClassifier,
    coverage_report,
)


def test_split_conformal_coverage():
    rng = np.random.default_rng(0)
    n_cal, n_test = 200, 200
    # Well-calibrated synthetic scores
    y_cal = rng.integers(0, 2, size=n_cal)
    proba_cal = np.clip(0.2 + 0.6 * y_cal + rng.normal(0, 0.1, n_cal), 0.01, 0.99)
    y_test = rng.integers(0, 2, size=n_test)
    proba_test = np.clip(0.2 + 0.6 * y_test + rng.normal(0, 0.1, n_test), 0.01, 0.99)

    conf = SplitConformalClassifier(ConformalConfig(alpha=0.1, seed=0))
    conf.calibrate(proba_cal, y_cal)
    sets = conf.predict_sets(proba_test)
    assert all(len(s) >= 1 for s in sets)
    report = coverage_report(sets, y_test, alpha=0.1)
    assert report["n"] == n_test
    # With decent scores, coverage should be near target
    assert report["empirical_coverage"] >= 0.8
    assert report["avg_set_size"] >= 1.0
