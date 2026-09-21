"""CLI entrypoint for case studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import DISCLAIMER, __version__
from .pipeline import PipelineConfig, run_pipeline
from .signals.synthetic import SyntheticConfig
from .ssl.contrastive import ContrastiveConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clinical Signal Copilot — research case studies (NOT a medical device)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-records", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="artifacts")
    parser.add_argument("--quick", action="store_true", help="Tiny run for smoke tests")
    args = parser.parse_args(argv)

    print("=" * 72)
    print(DISCLAIMER)
    print("=" * 72)
    print(f"clinical-signal-copilot v{__version__}")

    if args.quick:
        syn = SyntheticConfig(n_records=12, duration_s=8.0, seed=args.seed)
        ssl = ContrastiveConfig(epochs=2, max_windows=128, batch_size=32, seed=args.seed)
    else:
        syn = SyntheticConfig(n_records=args.n_records, seed=args.seed)
        ssl = ContrastiveConfig(epochs=args.epochs, seed=args.seed)

    result = run_pipeline(PipelineConfig(synthetic=syn, contrastive=ssl, seed=args.seed))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(result["report_markdown"], encoding="utf-8")
    (out / "report.html").write_text(result["report_html"], encoding="utf-8")
    summary = {
        "case_studies": result["case_studies"],
        "ssl_loss_history": result["ssl_history"],
        "disclaimer": DISCLAIMER,
    }
    (out / "case_studies.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(result["case_studies"], indent=2))
    print(f"Wrote {out / 'report.md'}, {out / 'report.html'}, {out / 'case_studies.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
