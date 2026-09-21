"""End-to-end research pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from .signals.synthetic import SyntheticConfig, generate_dataset, extract_windows
from .signals.quality import window_sqi
from .ssl.contrastive import ContrastiveConfig, ContrastivePretrainer
from .ssl.encoder import featurize_windows
from .detection.supervised import DetectorConfig, EventDetector
from .detection.metrics import detection_metrics
from .conformal.predictor import ConformalConfig, SplitConformalClassifier, coverage_report
from .fairness.audit import FairnessAuditor, reweight_mitigation
from .report.copilot import CopilotReport, render_markdown, render_html


@dataclass
class PipelineConfig:
    synthetic: SyntheticConfig | None = None
    contrastive: ContrastiveConfig | None = None
    detector: DetectorConfig | None = None
    conformal: ConformalConfig | None = None
    window_s: float = 0.8
    hop_s: float = 0.4
    label_mode: str = "peak"
    test_size: float = 0.25
    cal_size: float = 0.25
    seed: int = 42


def _split_masks(n: int, test_size: float, cal_size: float, seed: int, y=None):
    idx = np.arange(n)
    strat = y if y is not None and len(np.unique(y)) > 1 else None
    trainval, test = train_test_split(
        idx, test_size=test_size, random_state=seed, shuffle=True, stratify=strat
    )
    strat2 = y[trainval] if strat is not None else None
    train, cal = train_test_split(
        trainval, test_size=cal_size, random_state=seed + 1, shuffle=True, stratify=strat2
    )
    return train, cal, test


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, Any]:
    """Run full SSL → detect → conformal → fairness → report pipeline."""
    cfg = config or PipelineConfig()
    syn_cfg = cfg.synthetic or SyntheticConfig(seed=cfg.seed)
    ds = generate_dataset(syn_cfg)
    X, y, sg, rid = extract_windows(ds, cfg.window_s, cfg.hop_s, cfg.label_mode, seed=cfg.seed)

    train, cal, test = _split_masks(len(y), cfg.test_size, cfg.cal_size, cfg.seed, y=y)
    # --- Case A: SSL vs supervised-only ---
    ssl_cfg = cfg.contrastive or ContrastiveConfig(seed=cfg.seed, epochs=6, max_windows=384)
    pretrainer = ContrastivePretrainer(ssl_cfg)
    # Unlabeled = all train windows (SSL ignores labels)
    pretrainer.fit(X[train])
    emb_train = pretrainer.transform(X[train])
    emb_cal = pretrainer.transform(X[cal])
    emb_test = pretrainer.transform(X[test])

    det_ssl = EventDetector(cfg.detector or DetectorConfig(seed=cfg.seed))
    det_ssl.fit(X[train], y[train], embeddings=emb_train)

    det_sup = EventDetector(cfg.detector or DetectorConfig(seed=cfg.seed))
    det_sup.fit(X[train], y[train], embeddings=None)

    proba_ssl = det_ssl.predict_proba(X[test], embeddings=emb_test)
    pred_ssl = (proba_ssl >= 0.5).astype(np.int8)
    proba_sup = det_sup.predict_proba(X[test], embeddings=None)
    pred_sup = (proba_sup >= 0.5).astype(np.int8)

    metrics_ssl = detection_metrics(y[test], pred_ssl)
    metrics_sup = detection_metrics(y[test], pred_sup)

    # --- Case B: conformal coverage ---
    conf = SplitConformalClassifier(cfg.conformal or ConformalConfig(alpha=0.1, seed=cfg.seed))
    proba_cal = det_ssl.predict_proba(X[cal], embeddings=emb_cal)
    conf.calibrate(proba_cal, y[cal])
    sets = conf.predict_sets(proba_ssl)
    cov = coverage_report(sets, y[test], conf.config.alpha)

    # --- Case C: fairness ---
    auditor = FairnessAuditor()
    fair_before = auditor.audit(y[test], pred_ssl, sg[test])
    # Mitigation sketch: retrain with inverse-frequency weights on train
    weights = reweight_mitigation(sg[train], y[train])
    # sklearn logreg supports sample_weight
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feats_train = emb_train
    feats_test = emb_test
    mit = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=400, class_weight=None, random_state=cfg.seed, solver="lbfgs"
                ),
            ),
        ]
    )
    mit.fit(feats_train, y[train], clf__sample_weight=weights)
    pred_mit = mit.predict(feats_test).astype(np.int8)
    fair_after = auditor.audit(y[test], pred_mit, sg[test])

    # SQI summary on test windows
    sqi_vals = np.array([float(window_sqi(X[i]).mean()) for i in test])
    sqi_summary = {
        "mean_sqi": round(float(sqi_vals.mean()), 4),
        "min_sqi": round(float(sqi_vals.min()), 4),
        "frac_low_sqi_<0.4": round(float(np.mean(sqi_vals < 0.4)), 4),
        "n_test_windows": int(len(test)),
    }

    case_studies = {
        "A — SSL vs supervised-only": {
            "ssl_f1": round(metrics_ssl["f1"], 4),
            "supervised_only_f1": round(metrics_sup["f1"], 4),
            "ssl_sensitivity": round(metrics_ssl["sensitivity"], 4),
            "supervised_only_sensitivity": round(metrics_sup["sensitivity"], 4),
            "delta_f1_ssl_minus_sup": round(metrics_ssl["f1"] - metrics_sup["f1"], 4),
        },
        "B — Conformal coverage": {
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in cov.items()},
            "qhat": round(float(conf.qhat_), 4) if conf.qhat_ is not None else None,
        },
        "C — Fairness gap illustration": {
            "gaps_before": {k: round(v, 4) for k, v in fair_before["gaps"].items()},
            "gaps_after_reweight": {k: round(v, 4) for k, v in fair_after["gaps"].items()},
            "groups_before": fair_before["group_names"],
        },
    }

    report = CopilotReport(
        detection={
            "ssl": metrics_ssl,
            "supervised_only": metrics_sup,
        },
        conformal=cov,
        fairness=fair_before,
        sqi_summary=sqi_summary,
        case_studies=case_studies,
        notes=[
            "All signals are fully synthetic (PhysioNet-style morphology, no real recordings).",
            "SSL uses SimCLR/CLOCS-style temporal+channel views with NT-Xent (numpy encoder).",
            "Conformal guarantees assume exchangeability — violated under distribution shift.",
            "Fairness mitigation is a sketch (sample reweighting), not a clinical fairness solution.",
        ],
    )

    return {
        "dataset": ds,
        "X": X,
        "y": y,
        "subgroups": sg,
        "splits": {"train": train, "cal": cal, "test": test},
        "pretrainer": pretrainer,
        "detector_ssl": det_ssl,
        "detector_sup": det_sup,
        "conformal": conf,
        "metrics_ssl": metrics_ssl,
        "metrics_sup": metrics_sup,
        "coverage": cov,
        "fairness_before": fair_before,
        "fairness_after": fair_after,
        "case_studies": case_studies,
        "report": report,
        "report_markdown": render_markdown(report),
        "report_html": render_html(report),
        "ssl_history": list(pretrainer.history_),
        "raw_feature_dim": int(featurize_windows(X[:2]).shape[1]),
    }
