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
