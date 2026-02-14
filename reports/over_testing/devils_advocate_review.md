# DEVILS-ADVOCATE REVIEW (Updated 2026-02-13)

## Independent Risk Review

The refreshed data quality is good (`contexts_detected=1785`), so we can now trust overlap directionally. The major risk is no longer context validity; it is over-aggressive pruning too close to the 90% gate.

## Cross-Proposal Challenges

### Pruner

Strengths:
- Correctly identifies one all-zero file and a set of extremely low-value candidates.

Risks:
- Some low-mean files still contain a small number of unique lines that may guard niche branches.
- File-level pruning alone can remove disproportionately valuable edge tests.

Required mitigation:
- Remove in mini-batches (2-3 files), verify full suite/coverage each step.

### Deadcode-Hunter

Strengths:
- Conservative call: no high-confidence dead src removals.

Risk:
- False-positive interpretation of protocol hooks (e.g., `_missing_`) must remain blocked.

Required mitigation:
- No source deletions without dynamic reachability proof.

### Test-Creator

Strengths:
- Focuses on high-yield, deterministic, public-path branches.

Risk:
- Overestimating uplift from large modules (`plotting.py`, `explanations/*`) per test.

Required mitigation:
- Keep additions compact and verify real uplift with each merge.

### Anti-Pattern Auditor

Strengths:
- Clean report (0 blockers).

Risk:
- Clean scanners can hide latent weak assertions if rules are too narrow.

Required mitigation:
- Keep periodic manual sampling of low-value files during pruning.

### Process Architect

Strengths:
- Correctly flags tooling mismatches (`select_zero_unique_files.py`, runtime=0 score signal).

Risk:
- If left unfixed, automated recommendations remain partially misleading.

Required mitigation:
- Script fixes before scaling automated prune selection.

## Consolidated Risk Ratings

| Planned change | Risk | Reason |
| --- | --- | --- |
| Remove one all-zero file (`test_exec_core_reject_module.py`) | Low | No unique contribution in fresh data |
| Remove extremely low-value files (mean < 0.1) | Medium | Small unique lines may still protect branch edge cases |
| Remove source code as dead | High | No high-confidence dead-code set identified |
| Continue iterative test backfill for headroom | Low | Proven effective in current repo state |

## Recommended Execution Order

1. Remove only the all-zero file.
2. Run full suite/coverage.
3. Remove 2-3 low-value files from next candidate list.
4. Re-verify; when near 90.1, add high-quality backfill tests.
5. Repeat.

## No-Go List

- No large one-shot deletion wave from low-value lists while coverage is near 90%.
- No source dead-code removal based solely on static scan patterns.
- No estimator-driven mass removal until runtime and remove-list format issues are fixed.

## Net Assessment

The method is ready for continued pruning, but only with tight remove-verify-backfill cycles. The largest remaining operational risk is tooling-driven false confidence, not data freshness.

---

## DEVILS-ADVOCATE ADDENDUM (2026-02-13, Code-Focused CQ-001)

### Scope Reviewed

- Proposal: `reports/over_testing/code_quality_auditor_proposal.md`
- Inputs cross-checked:
  - `reports/anti-pattern-analysis/private_method_analysis.csv`
  - `reports/anti-pattern-analysis/test_anti_pattern_report.csv`
  - `reports/anti-pattern-analysis/code_hotspots.md`
  - current gate results from code-focused cycle (ADR-002/import/docstring/coverage)

### Findings (ordered by risk)

1. **High risk if misinterpreted:** proposed source "Pattern 3" removals are not safe.
   - `_reconstruct_*` symbols in `src/calibrated_explanations/cache/cache.py` are referenced dynamically via `getattr(mod, "...")` in pickle reduce paths.
   - `RejectPolicy._missing_` in `src/calibrated_explanations/explanations/reject.py` is Enum protocol behavior.
   - Verdict: keep out of deletion scope until dynamic behavior is explicitly replaced and revalidated.

2. **Medium risk:** `feature_task` refactor can silently drift tuple shape/ordering.
   - This function is high-complexity and central to explanation computation.
   - Existing tests hit key paths, but extraction changes can still alter edge behavior (especially categorical branch bookkeeping and masks).
   - Verdict: proceed only as bounded extraction (no signature change, no numeric-branch rewrite in this batch).

3. **Low risk:** dead-private test helper removals.
   - `_make_binary_explainer` and `ContainerStub._get_explainer` appear unused and non-reflective.
   - Verdict: safe to remove first as independent micro-change.

### Consolidated Risk Ratings for CQ-001

| Change | Risk | Decision |
| --- | --- | --- |
| Remove `_make_binary_explainer` | Low | GO |
| Remove `ContainerStub._get_explainer` | Low | GO |
| Extract early-return helper in `feature_task` | Medium | GO (bounded) |
| Extract categorical helper in `feature_task` | Medium | GO (bounded) |
| Delete Pattern-3 source symbols in `cache.py` / `reject.py` | High | NO-GO |

### Required Mitigations

1. Keep CQ-001 as two commits/phases:
   - Phase 1: dead-private test helper removals only.
   - Phase 2: `feature_task` extraction only.
2. Run targeted tests before full suite on each phase.
3. Preserve exact return tuple contract and public signature for `feature_task`.
4. Re-run gate pack + coverage gates after each phase.

### Recommended Execution Order

1. Apply Phase 1 dead-private test helper removals.
2. Run targeted tests + gate pack + full coverage gates.
3. Apply Phase 2 bounded `feature_task` extraction.
4. Re-run targeted tests + gate pack + full coverage gates.
5. Refresh anti-pattern artifacts and record results in remedy ledger.

### No-Go List (for this batch)

- No deletion of dynamically referenced `_reconstruct_*` functions.
- No removal or renaming of Enum `_missing_` hooks.
- No numeric-branch restructuring inside `feature_task` in CQ-001.
