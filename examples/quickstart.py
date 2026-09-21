#!/usr/bin/env python3
"""Quickstart: generate synthetic signals and print a tiny report."""

from clinical_signal_copilot import DISCLAIMER
from clinical_signal_copilot.pipeline import PipelineConfig, run_pipeline
from clinical_signal_copilot.signals.synthetic import SyntheticConfig
from clinical_signal_copilot.ssl.contrastive import ContrastiveConfig

print(DISCLAIMER)
result = run_pipeline(
    PipelineConfig(
        synthetic=SyntheticConfig(n_records=12, duration_s=8.0, seed=0),
        contrastive=ContrastiveConfig(epochs=2, max_windows=96, batch_size=32, seed=0),
        seed=0,
    )
)
print(result["report_markdown"][:1500])
print("...")
print("Case studies:", result["case_studies"])
