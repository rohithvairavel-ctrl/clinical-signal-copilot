# Interview one-pager — clinical-signal-copilot

**30-second pitch:** Research-only biomedical signal stack: synthetic multi-channel physio → self-supervised representations → event detection → conformal risk sets → subgroup fairness diagnostics → uncertainty-aware copilot report. **Not a medical device.**

**Repo:** https://github.com/rohithvairavel-ctrl/clinical-signal-copilot

---

## Problem

Clinical ML fails in interviews when people only show accuracy. Real discussion needs **signal quality**, **representation learning**, **uncertainty sets**, and **subgroup gaps** — plus a hard ethical boundary.

## System (what I built)

1. **Synthetic multi-channel signals** — ECG-like morphology, SQI, arrhythmia/QRS-style labels, synthetic subgroups A/B (no real PHI / no PhysioNet downloads)  
2. **SSL** — CLOCS/SimCLR-style temporal + channel views; NT-Xent; lean numpy encoder  
3. **Detection head** — Sens / PPV / F1 with tolerance-aware peak matching  
4. **Split conformal** — set-valued predictions + coverage / set-size diagnostics  
5. **Fairness sketch** — gap audit + inverse-frequency reweight illustration  
6. **Copilot report / optional Streamlit** — uncertainty, SQI, caveats bannered as non-clinical

## Proof points

- Case A: SSL features lift F1 vs supervised-only (demo Δ on the order of ~0.08 in quick run)  
- Case B: conformal empirical coverage near target with reported average set size  
- Case C: synthetic subgroup gaps shrink after reweight (pedagogical, not causal fairness)  
- Tests: 16 seeded tests at ship

## Metrics I own

| Layer | Metric | Talking point |
|-------|--------|----------------|
| Detection | Sens / PPV / F1 @ tolerance | Threshold + matching window matter |
| SSL | Downstream ΔF1 | Pretrain is a means, not the product |
| Conformal | Coverage, avg set size, singleton rate | Abstention / set prediction > fake confidence |
| Fairness | Gap in Sens/F1 by subgroup | Diagnostics before “fair model” claims |
| SQI | Heuristic quality flags | Garbage-in awareness |

## Threats to validity

- Synthetic ≠ ICU/device data; no site or annotation shift  
- Lean numpy SSL ≠ large temporal CNN/Transformer  
- Conformal needs exchangeability — clinical shift breaks it  
- Fairness reweight is a sketch, not equalized odds / causal fairness  
- SQI heuristics are not validated clinical algorithms  

## 5-minute talk track

1. **(20s)** Open with the disclaimer: research demo, not SaMD, synthetic only  
2. **(60s)** Pipeline: signals → SSL → detect → conformal → fairness → report  
3. **(90s)** Deep dive conformal: why point predictions are dishonest; coverage vs set size tradeoff  
4. **(60s)** SSL vs supervised-only ablation  
5. **(60s)** Ethics + what production would require (multi-site validation, human factors, regulatory)

## Likely questions → answers

- **Why synthetic?** Licensing + PHI; still lets us test the *methodology* cleanly.  
- **Why conformal over softmax confidence?** Finite-sample coverage targets under exchangeability; softmax is not calibrated risk.  
- **Is this deployable?** No — explicitly not. It’s an interview system design artifact.

## What this is NOT

Not FDA/MDR SaMD, not diagnosis/triage/treatment, not a claim on real ECG benchmarks.
