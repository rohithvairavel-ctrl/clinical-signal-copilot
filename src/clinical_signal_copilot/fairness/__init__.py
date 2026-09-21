"""Subgroup fairness auditing and simple mitigation sketch."""

from .audit import FairnessAuditor, fairness_gaps, reweight_mitigation

__all__ = ["FairnessAuditor", "fairness_gaps", "reweight_mitigation"]
