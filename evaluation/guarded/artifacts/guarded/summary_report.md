# Guarded Explanation Evaluation Summary

This file is an engineering dashboard, not a paper summary.
For the paper, use Scenario A and Scenario B only.
Treat Scenarios C, D, and E as implementation checks.

1 PASS / 4 FAIL across 5 scenario(s).
Total runtime: 3215.2s

## Results by Scenario

| Scenario | Status | Runtime (s) |
|---|---|---|
| Scenario A: Domain plausibility (synthetic constraint) | ✗ FAIL | 9.2 |
| Scenario B: OOD detection quality | ✗ FAIL | 2568.2 |
| Scenario C: Regression invariants | ✗ FAIL | 305.1 |
| Scenario D: Real dataset correctness | ✗ FAIL | 315.7 |
| Scenario E: Edge case behavior | ✓ PASS | 17.0 |

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

## Failed Scenarios

- Scenario A: Domain plausibility (synthetic constraint) (exit code 3228369023)
- Scenario B: OOD detection quality (exit code 1)
- Scenario C: Regression invariants (exit code 1)
- Scenario D: Real dataset correctness (exit code 1)
