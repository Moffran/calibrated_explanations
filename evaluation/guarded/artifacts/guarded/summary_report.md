# Guarded Explanation Evaluation — Summary Report

**5 PASS / 0 FAIL** across 5 scenario(s).
Total runtime: 1785.2s

## Results by Scenario

| Scenario | Status | Runtime (s) |
|---|---|---|
| Scenario A — Domain plausibility (synthetic constraint) | ✓ PASS | 1533.0 |
| Scenario B — OOD detection quality | ✓ PASS | 53.5 |
| Scenario C — Regression invariants | ✓ PASS | 21.1 |
| Scenario D — Real dataset correctness | ✓ PASS | 162.9 |
| Scenario E — Edge case behavior | ✓ PASS | 14.7 |

## Per-Scenario Reports

Each scenario writes its own `report.md` under `artifacts/guarded/scenario_*/`.
See those files for metric details and interpretation.

## Metrics Quick Reference

| Metric | Scenario | Claim | Healthy | Red flag |
|---|---|---|---|---|
| `violation_rate` (guarded < standard) | A | Detection | Guarded lower | Guarded ≥ standard |
| `auroc` | B | Detection | > 0.80 for moderate+ shift | < 0.60 for extreme shift |
| `fpr_at_significance` | B | Calibration | ≈ significance | >> significance |
| `n_invariant_violations` | C | Correctness | 0 always | Any > 0 = bug |
| `audit_field_completeness` | D | Correctness | True always | Any False = bug |
| `fraction_instances_fully_filtered` | D | Usability | < 0.05 at α=0.10 | > 0.10 |
| Edge case PASS/FAIL | E | Correctness | All PASS | Any unexpected FAIL |
