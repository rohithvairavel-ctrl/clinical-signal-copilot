# Clinical Signal Copilot — Research Report

> **DISCLAIMER:** NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE, DIAGNOSIS, TRIAGE, OR TREATMENT. This is a research / education decision-aid demo on fully synthetic signals. Do not use on real patients or clinical workflows.

## Signal quality (SQI)

- **mean_sqi**: 0.7073
- **min_sqi**: 0.5296
- **frac_low_sqi_<0.4**: 0.0
- **n_test_windows**: 43

## Detection metrics

- **ssl**: {'tp': 19, 'fp': 6, 'fn': 7, 'sensitivity': 0.7307692307692027, 'ppv': 0.7599999999999697, 'f1': 0.7450980392151573}
- **supervised_only**: {'tp': 16, 'fp': 6, 'fn': 10, 'sensitivity': 0.6153846153845918, 'ppv': 0.7272727272726943, 'f1': 0.6666666666661424}

## Conformal risk sets

- **target_coverage**: 0.9000
- **empirical_coverage**: 0.9535
- **avg_set_size**: 1.6512
- **singleton_rate**: 0.3488
- **n**: 43
- **q_diagnostic_ok**: True

_Interpretation:_ set size > 1 means the model abstains to a risk set; do not treat singleton predictions as clinical ground truth.

## Subgroup fairness

### Metric gaps (max − min across synthetic subgroups)
- **sensitivity gap**: 0.1905
- **ppv gap**: 0.0192
- **f1 gap**: 0.1077
### Group `A` (n=19)
- sensitivity: 0.8333
- ppv: 0.7692
- f1: 0.8000
### Group `B` (n=24)
- sensitivity: 0.6429
- ppv: 0.7500
- f1: 0.6923

**Caveat:** subgroups are synthetic attributes for demo disparity diagnostics only — not real race/sex/age or clinical cohorts.

## Case studies

### A — SSL vs supervised-only
- **ssl_f1**: 0.7451
- **supervised_only_f1**: 0.6667
- **ssl_sensitivity**: 0.7308
- **supervised_only_sensitivity**: 0.6154
- **delta_f1_ssl_minus_sup**: 0.0784

### B — Conformal coverage
- **target_coverage**: 0.9
- **empirical_coverage**: 0.9535
- **avg_set_size**: 1.6512
- **singleton_rate**: 0.3488
- **n**: 43
- **q_diagnostic_ok**: True
- **qhat**: 0.8844

### C — Fairness gap illustration
- **gaps_before**: {'sensitivity': 0.1905, 'ppv': 0.0192, 'f1': 0.1077}
- **gaps_after_reweight**: {'sensitivity': 0.119, 'ppv': 0.0, 'f1': 0.0593}
- **groups_before**: ['A', 'B']


## Notes

- All signals are fully synthetic (PhysioNet-style morphology, no real recordings).
- SSL uses SimCLR/CLOCS-style temporal+channel views with NT-Xent (numpy encoder).
- Conformal guarantees assume exchangeability — violated under distribution shift.
- Fairness mitigation is a sketch (sample reweighting), not a clinical fairness solution.

---
*NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE, DIAGNOSIS, TRIAGE, OR TREATMENT. This is a research / education decision-aid demo on fully synthetic signals. Do not use on real patients or clinical workflows.*
