"""Markdown / HTML decision-aid report with uncertainty, SQI, fairness caveats.

*** NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE. ***
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import Any


DISCLAIMER = (
    "NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE, DIAGNOSIS, TRIAGE, OR TREATMENT. "
    "This is a research / education decision-aid demo on fully synthetic signals. "
    "Do not use on real patients or clinical workflows."
)


@dataclass
class CopilotReport:
    title: str = "Clinical Signal Copilot — Research Report"
    detection: dict = field(default_factory=dict)
    conformal: dict = field(default_factory=dict)
    fairness: dict = field(default_factory=dict)
    sqi_summary: dict = field(default_factory=dict)
    case_studies: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def render_markdown(report: CopilotReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"> **DISCLAIMER:** {DISCLAIMER}",
        "",
        "## Signal quality (SQI)",
        "",
    ]
    if report.sqi_summary:
        for k, v in report.sqi_summary.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("_No SQI summary provided._")
    lines += ["", "## Detection metrics", ""]
    if report.detection:
        for k, v in report.detection.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.4f}")
            else:
                lines.append(f"- **{k}**: {v}")
    lines += ["", "## Conformal risk sets", ""]
    if report.conformal:
        for k, v in report.conformal.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.4f}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")
        lines.append(
            "_Interpretation:_ set size > 1 means the model abstains to a risk set; "
            "do not treat singleton predictions as clinical ground truth."
        )
    lines += ["", "## Subgroup fairness", ""]
    if report.fairness:
        gaps = report.fairness.get("gaps", {})
        lines.append("### Metric gaps (max − min across synthetic subgroups)")
        for k, v in gaps.items():
            lines.append(f"- **{k} gap**: {v:.4f}")
        groups = report.fairness.get("groups", {})
        for g, m in groups.items():
            lines.append(f"### Group `{g}` (n={m.get('n', '?')})")
            for kk in ("sensitivity", "ppv", "f1"):
                if kk in m:
                    lines.append(f"- {kk}: {m[kk]:.4f}")
        lines.append("")
        lines.append(
            "**Caveat:** subgroups are synthetic attributes for demo disparity "
            "diagnostics only — not real race/sex/age or clinical cohorts."
        )
    if report.case_studies:
        lines += ["", "## Case studies", ""]
        for name, body in report.case_studies.items():
            lines.append(f"### {name}")
            if isinstance(body, dict):
                for k, v in body.items():
                    lines.append(f"- **{k}**: {v}")
            else:
                lines.append(str(body))
            lines.append("")
    if report.notes:
        lines += ["", "## Notes", ""]
        for n in report.notes:
            lines.append(f"- {n}")
    lines += [
        "",
        "---",
        f"*{DISCLAIMER}*",
        "",
    ]
    return "\n".join(lines)


def render_html(report: CopilotReport) -> str:
    md_like = render_markdown(report)
    # Minimal HTML wrapper (no external deps required)
    body = escape(md_like).replace("\n", "<br>\n")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{escape(report.title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; }}
.banner {{ background: #7f1d1d; color: #fff; padding: 1rem; border-radius: 8px; font-weight: 600; }}
pre {{ white-space: pre-wrap; background: #f8fafc; padding: 1rem; border-radius: 8px; }}
</style>
</head>
<body>
<div class="banner">{escape(DISCLAIMER)}</div>
<pre>{body}</pre>
</body>
</html>
"""
