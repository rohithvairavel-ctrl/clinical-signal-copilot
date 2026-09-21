"""Clinical Signal Copilot — research-only biomedical signal DS+AIML demo.

*** NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE. RESEARCH / EDUCATION ONLY. ***

Pipeline: synthetic multi-channel physio signals → SSL representations →
event detection → conformal risk sets → subgroup fairness → uncertainty report.
"""

from __future__ import annotations

__version__ = "0.1.0"
__disclaimer__ = (
    "NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE, DIAGNOSIS, OR TREATMENT. "
    "Research and education demo only. All signals are fully synthetic."
)

DISCLAIMER = __disclaimer__
