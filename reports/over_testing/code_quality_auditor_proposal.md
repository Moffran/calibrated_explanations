# Code-Quality Auditor Proposal

Date: 2026-02-13
Cycle: Extended code-focused cycle
Scope: Concrete remediation batch draft (not yet applied)

## Batch CQ-001 (Concrete)

Goal: remove truly dead private helpers and execute one contained hotspot refactor with strong verification.

### A) Dead-private removals (low risk)

1. Remove unused helper function:
   - Target: `tests/unit/explanations/test_conjunction_hardening.py:14`
   - Symbol: `_make_binary_explainer`
   - Evidence: symbol definition found, no call sites in `tests/unit` (searched via `rg`).
   - Expected impact: no behavioral change; smaller test surface.

2. Remove unused private stub method:
   - Target: `tests/unit/explanations/test_explanation_more.py:42`
   - Symbol: `ContainerStub._get_explainer`
   - Evidence: `ContainerStub.get_explainer` is used; `_get_explainer` has no call sites.
   - Expected impact: none, test readability improves.

### B) Hotspot refactor target (medium risk)

Target:
- `src/calibrated_explanations/core/explain/feature_task.py:117`
- Function: `feature_task`
- Reason: high hotspot score and high branching in core explanation path.

Refactor shape (single batch, no behavior change):

1. Extract early-return path helper:
   - Proposed helper: `_build_empty_feature_result(...)`
   - Covers duplicated early-return logic for ignored/empty feature index paths.

2. Extract categorical branch helper:
   - Proposed helper: `_process_categorical_feature(...)`
   - Moves categorical aggregation + weighting logic into a focused unit.

3. Keep numeric branch in `feature_task` for Batch CQ-001:
   - Do not split numeric branch yet in this batch.
   - Rationale: constrain risk and keep first hotspot refactor bounded.

4. Keep public API stable:
   - No signature changes for `feature_task`.
   - Return tuple shape must remain exactly unchanged.

## Risk Assessment

1. Dead-private removals: Low
   - Risk: accidental hidden usage in tests via reflection.
   - Mitigation: run targeted tests and full unit subset after deletion.

2. `feature_task` extraction: Medium
   - Risk: subtle output shape or ordering drift in return tuple.
   - Mitigation: strict targeted tests around `feature_task`, then full gates.

3. False-positive dead-private source symbols: High if removed
   - Symbols in `src/calibrated_explanations/cache/cache.py` and `src/calibrated_explanations/explanations/reject.py` reported as Pattern 3 are dynamically referenced:
     - pickle reconstruct hooks loaded via `getattr(mod, \"_reconstruct_...\")`
     - Enum hook `_missing_` invoked by Enum machinery
   - Decision: do **not** remove these in CQ-001.

## Verification Steps

Run in this order:

1. Targeted tests for touched logic:
   - `pytest -q --no-cov tests/unit/core/test_calibrated_explainer_additional.py tests/unit/core/test_assign_weight_scalar.py tests/unit/core/test_explain_helpers_and_plugins.py tests/unit/explanations/test_explanation_more.py tests/unit/explanations/test_conjunction_hardening.py`

2. Code-quality gate pack:
   - `python scripts/quality/check_adr002_compliance.py`
   - `python scripts/quality/check_import_graph.py`
   - `python scripts/quality/check_docstring_coverage.py`

3. Deprecation-sensitive subset:
   - PowerShell: `$env:CE_DEPRECATIONS='error'; pytest tests/unit -m "not viz" -q --maxfail=1 --no-cov`

4. Coverage gates:
   - `pytest --cov-fail-under=90 -q`
   - `python scripts/quality/check_coverage_gates.py`

5. Refresh analysis artifacts:
   - `python scripts/anti-pattern-analysis/analyze_private_methods.py src tests --output reports/anti-pattern-analysis/private_method_analysis.csv`
   - `python scripts/anti-pattern-analysis/detect_test_anti_patterns.py --output reports/anti-pattern-analysis/test_anti_pattern_report.csv`

## Expected Outcome

1. Eliminate two dead private test helpers with negligible regression risk.
2. Reduce complexity concentration in `feature_task` while preserving behavior.
3. Keep all current package/module coverage and code-quality gates green.
