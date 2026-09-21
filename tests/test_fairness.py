import numpy as np

from clinical_signal_copilot.fairness.audit import FairnessAuditor, fairness_gaps, reweight_mitigation


def test_fairness_gaps_and_weights():
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    sg = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    out = fairness_gaps(y_true, y_pred, sg)
    assert "gaps" in out and "groups" in out
    assert set(out["group_names"]) == {"A", "B"}
    w = reweight_mitigation(sg, y_true)
    assert w.shape == (8,)
    assert np.all(w > 0)
    aud = FairnessAuditor()
    assert aud.audit(y_true, y_pred, sg)["gaps"]
