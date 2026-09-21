import numpy as np

from clinical_signal_copilot.detection.metrics import detection_metrics, match_events
from clinical_signal_copilot.detection.supervised import DetectorConfig, EventDetector


def test_match_events_tolerance():
    true = np.array([100, 200, 300])
    pred = np.array([102, 250, 301])
    tp, fp, fn = match_events(true, pred, tolerance=5)
    assert tp == 2 and fp == 1 and fn == 1


def test_detection_metrics_binary():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    m = detection_metrics(y_true, y_pred)
    assert set(m) >= {"sensitivity", "ppv", "f1", "tp", "fp", "fn"}
    assert 0 <= m["f1"] <= 1


def test_event_detector_fit_predict(windows):
    X, y, _, _ = windows
    # Ensure both classes present
    if y.sum() == 0 or y.sum() == len(y):
        y = y.copy()
        y[0] = 1
        y[1] = 0
    det = EventDetector(DetectorConfig(seed=0, max_iter=200))
    det.fit(X, y)
    proba = det.predict_proba(X)
    pred = det.predict(X)
    assert proba.shape == (len(y),)
    assert pred.shape == (len(y),)
    assert set(np.unique(pred)).issubset({0, 1})
