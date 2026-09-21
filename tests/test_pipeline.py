from clinical_signal_copilot.pipeline import PipelineConfig, run_pipeline
from clinical_signal_copilot.signals.synthetic import SyntheticConfig
from clinical_signal_copilot.ssl.contrastive import ContrastiveConfig


def test_run_pipeline_quick():
    result = run_pipeline(
        PipelineConfig(
            synthetic=SyntheticConfig(n_records=10, duration_s=6.0, fs=125.0, seed=42),
            contrastive=ContrastiveConfig(epochs=2, max_windows=80, batch_size=16, seed=42),
            seed=42,
        )
    )
    assert "case_studies" in result
    assert "A — SSL vs supervised-only" in result["case_studies"]
    assert "B — Conformal coverage" in result["case_studies"]
    assert "C — Fairness gap illustration" in result["case_studies"]
    assert "NOT A MEDICAL DEVICE" in result["report_markdown"]
    assert result["metrics_ssl"]["f1"] >= 0.0
    assert 0.0 <= result["coverage"]["empirical_coverage"] <= 1.0
