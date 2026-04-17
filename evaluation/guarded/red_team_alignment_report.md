# Guarded Paper Red-Team Alignment Report

Date: 2026-04-16

Scope: alignment between paper claims in [evaluation/guarded/main.tex](evaluation/guarded/main.tex), guarded core implementation, and evaluation evidence in [evaluation/guarded](evaluation/guarded).

## Findings (Ordered by Severity)

### High: Suite-level artifact status currently undermines paper evidence chain

Evidence:

- [evaluation/guarded/artifacts/guarded/summary_report.md](evaluation/guarded/artifacts/guarded/summary_report.md#L8) reports 1 PASS / 4 FAIL.
- The same summary explicitly labels itself engineering dashboard output.

Risk:

- If referenced directly, this creates a credibility conflict with positive paper claims.

Required action:

- Treat suite summary as engineering CI signal only.
- Use scenario-specific full-run paper artifacts for claims in main text.

### High: Scenario A metric wording previously implied representative-only scoring

Evidence:

- Guarded plausibility in code is rule-format aware via [evaluation/guarded/scenario_a_guarded_vs_standard.py](evaluation/guarded/scenario_a_guarded_vs_standard.py#L305).
- Main-text wording was updated to reflect emitted-rule condition scoring in [evaluation/guarded/main.tex](evaluation/guarded/main.tex#L494).

Risk:

- Reviewer could claim method-report mismatch if representative-only phrasing is used.

Required action:

- Keep rule-format-aware metric contract in manuscript and scenario README.

### Medium: Scenario B paper-facing metric contract is correct but easy to overstate

Evidence:

- AUROC uses Fisher-combined per-instance scores in [evaluation/guarded/scenario_b_ood_detection_quality.py](evaluation/guarded/scenario_b_ood_detection_quality.py#L42).
- README distinguishes AUROC from interval-level FPR interpretation in [evaluation/guarded/README.md](evaluation/guarded/README.md#L55).

Risk:

- Reviewer pushback if combined Fisher scores are framed as conformal p-values.

Required action:

- Keep explicit language that Fisher-combined values are anomaly scores for ranking, not conformal validity statements.

### Medium: Guarded semantics are scoped and should remain narrowly claimed

Evidence:

- Single representative probe per bin in [src/calibrated_explanations/core/explain/_guarded_explain.py](src/calibrated_explanations/core/explain/_guarded_explain.py#L803).
- ADR scope limits in [docs/improvement/adrs/ADR-032-guarded-explanation-semantics.md](docs/improvement/adrs/ADR-032-guarded-explanation-semantics.md).

Risk:

- Overclaim risk if paper implies whole-interval certification or semantic identity with standard factual CE.

Required action:

- Keep non-claim wording explicit: no whole-interval certification, no causal/actionability guarantee.

### Low: Fast explainer incompatibility needs clear visibility in paper and docs

Evidence:

- Hard fail in guarded core for fast mode at [src/calibrated_explanations/core/explain/_guarded_explain.py](src/calibrated_explanations/core/explain/_guarded_explain.py#L78).

Risk:

- Reproducibility friction if readers attempt fast mode for guarded runs.

Required action:

- Add one sentence in reproducibility appendix or implementation note clarifying guarded entrypoints require non-fast explainers.

## Claim-to-Implementation Status

| Claim Area | Status | Notes |
| --- | --- | --- |
| Guarded perturbation filtering via KNN conformal test | Supported | Implemented in guard + orchestrator path. |
| Reuse of CE calibrated prediction semantics after filtering | Supported | Batched prediction uses CE backend for retained candidates. |
| Interval-level guarded audit trail | Supported | Exposed by guarded explanation audit payloads. |
| Whole-interval safety/certification | Not claimed (correct) | Must remain out of scope per ADR-032. |
| Equivalence with standard factual semantics | Not claimed (correct) | Must remain explicitly not equivalent. |

## COPA Relevance Assessment

Strengths:

- Clear methodological contribution: distribution-aware filtering before rule assembly.
- Strong implementation traceability through audit payloads and explicit scenario contracts.
- Focused experimental framing in Scenario A and B.

Weaknesses:

- Mixed checked-in artifact state can confuse evidence quality if not carefully scoped.
- Limited external baseline diversity in paper-facing experiments.
- Statistical robustness presentation remains narrow unless full-run uncertainty reporting is consistently foregrounded.

Current venue-fit risk: Medium.

Interpretation:

- The technical idea is relevant and defensible for COPA.
- Acceptance risk is mostly evidence-packaging and claim-precision risk, not core-method incoherence.

## Minimum Next Implementation Steps

1. Ensure paper tables/figures are generated from full-run Scenario A/B outputs only.
2. Keep manuscript language tied to current metric contracts and ADR-032 scope boundaries.
3. Add a short reproducibility note on non-fast requirement for guarded entrypoints.
4. Keep Scenario C/D/E explicitly labeled engineering validation in paper and supplemental text.
