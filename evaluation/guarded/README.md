# Guarded Explanation Evaluation Suite

This directory contains two different kinds of evaluation:

1. paper-facing experiments that support the main guarded explanation claims
2. engineering checks that harden the implementation but should not be used as main paper evidence

Semantics note:
- These evaluations operate on the guarded interval semantics.
- A candidate counted as `intervals_removed_guard` or `intervals_conforming` reflects the guard-rule decision for that interval candidate, not whole-interval certification.

The main paper should use Scenario A and Scenario B only.

## Paper-facing subset

### Scenario A: controlled rule usefulness

Script: `scenario_a_guarded_vs_standard.py`

Question:
Does guarded CE emit rules that better respect a known domain constraint than standard CE?

Why this belongs in the paper:
Scenario A is the only setting in this suite with a direct notion of rule plausibility. It therefore tests the main usefulness claim of guarded explanations under the emitted guarded rule format.

Use in the paper:
- Primary metric: `violation_rate`
- Optional tradeoff metric: factual-mode `rule_count`

Do not use in the paper:
- `stability_jaccard`
- prediction-agreement metrics
- interval-overlap metrics
- alternative-mode rule-count headlines

Paper interpretation:
- Lower guarded `violation_rate` is the main result.
- Factual `rule_count` can be used once as a cost and coverage tradeoff.
- This scenario validates the emitted guarded rule format, not whole-interval safety and not a pure representative-only semantics claim.

### Scenario B: direct guard detection quality

Script: `scenario_b_ood_detection_quality.py`

Question:
Do the guard scores separate in-distribution from out-of-distribution points under controlled distribution shift?

Why this belongs in the paper:
Scenario B is the direct test of the guard itself. It supports the claim that the guard is not filtering arbitrarily.

Use in the paper:
- Primary metric: `auroc`
- Optional calibration-style diagnostic: interval-level `fpr_at_significance` computed from raw audit-row p-values on in-distribution rows

Metric contract:
- AUROC uses Fisher-combined per-instance p-values.
- The interval-level rejection rate is not computed from combined instance scores.
- The paper should use the default slice with `normalize_guard=True` and `n_neighbors=5`.

Do not use in the paper:
- non-default normalization as a headline result
- `n_neighbors=1` stress behaviour as a headline result
- any theoretical wording that treats Fisher-combined instance scores as conformal p-values

Paper interpretation:
- AUROC is the main detection result.
- The interval-level rejection rate is a threshold diagnostic, not the main result.
- The scenario matters because it tests whether the guard is selective for a reason, rather than merely removing intervals.

## Engineering validation only

### Scenario C: regression invariant check

Script: `scenario_c_regression.py`

Use:
- appendix sanity check only, if the paper keeps a regression-support claim

Keep:
- `n_invariant_violations`
- a brief statement that the regression path is separate code and was checked explicitly

Do not use:
- the OOD-responsiveness secondary diagnostic as main evidence

Interpretation:
- This scenario is primarily a correctness gate.
- Any non-zero invariant violation is a bug, not a weak result.
- A clean pass supports only a narrow implementation claim, not a broad regression-quality claim.

### Scenario D: real-data API and usability sweep

Script: `scenario_d_real_datasets.py`

Use:
- engineering validation only by default
- optional appendix note if the paper needs a minimal multiclass or real-data sanity check

Keep only if needed:
- no exceptions
- low `fraction_instances_fully_filtered` under default settings

Do not use:
- `audit_field_completeness` as a scientific result
- broad scenario D dashboards in the paper

Interpretation:
- Scenario D checks whether the guarded API survives realistic dataset shapes: multiclass outputs, high-dimensional inputs, and small calibration sets.
- It is useful for demonstrating robustness of the implementation, not superiority of guarded explanations on real tasks.

### Scenario E: edge-case hardening

Script: `scenario_e_edge_cases.py`

Use:
- engineering regression suite only

Do not use:
- PASS or FAIL edge-case tables in the paper

Interpretation:
- Scenario E documents design boundaries and catches brittle implementation failures.
- A PASS may still describe a limitation; it only means the behavior is known and acceptable.

## Running the suite

Run from `evaluation/guarded/` so that `common_guarded` is importable.

Quick checks:

```bash
cd evaluation/guarded
python scenario_a_guarded_vs_standard.py --quick
python scenario_b_ood_detection_quality.py --quick
python scenario_c_regression.py --quick
python scenario_d_real_datasets.py --quick
python scenario_e_edge_cases.py --quick
```

Full runs:

```bash
cd evaluation/guarded
python scenario_a_guarded_vs_standard.py
python scenario_b_ood_detection_quality.py
python scenario_c_regression.py
python scenario_d_real_datasets.py
python scenario_e_edge_cases.py
```

Master runner:

```bash
cd evaluation/guarded
python run_all_guarded.py --quick
python run_all_guarded.py --scenarios a,b --quick
python run_all_guarded.py --scenarios all
```

Artifacts are written to `evaluation/guarded/artifacts/`.

## Paper-use rules

- Do not use `--quick` artifacts in the paper.
- Do not include a suite dashboard in the paper.
- Do not include a metric unless it answers a specific paper question.
- Report uncertainty at seed level or seed-by-model level.
- Keep the paper evaluation text plain and short.
