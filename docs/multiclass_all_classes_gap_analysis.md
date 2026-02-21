# Multiclass all-classes plotting quality + gap analysis

## Scope reviewed
- Multiclass factual orchestration path in `ExplanationOrchestrator.invoke_factual`.
- `MultiClassCalibratedExplanations` plotting integration with `_plot_probabilistic_dict`.
- Test coverage around entrypoint aliasing for all-classes multiclass explanations.

## Findings
1. **Feature flag discoverability/linkage gap (high impact)**
   - The multiclass all-classes branch is guarded by `multi_lables_explanation` (misspelled) only.
   - No `all_classes` or correctly spelled `multi_labels_explanation` lookup existed, making the feature hard to discover and effectively unlinked from natural API usage.

2. **Behavioral validation gap (medium impact)**
   - No focused unit test asserted that public aliases route into the all-classes multiclass branch and return `MultiClassCalibratedExplanations`.

3. **Operational risk (medium impact)**
   - Because the branch is legacy-backed and conditionally imported, regressions can silently disable the all-class path without obvious failures unless branch-entry tests exist.

## Remedy implemented in this patch
- Added alias resolution for:
  - `all_classes`
  - `multi_labels_explanation`
  - `multi_lables_explanation` (backward-compat legacy alias)
- Added unit tests verifying all aliases hit the multiclass all-classes path and construct `MultiClassCalibratedExplanations`.

## Suggested follow-up plan
1. **API hardening**
   - Document `all_classes=True` as the canonical public flag.
   - Keep typo alias for one release, then deprecate with warning.

2. **Plot-path regression tests**
   - Add targeted tests for `MultiClassCalibratedExplanations.plot_factual()` and `plot_alternative()` to verify `_plot_probabilistic_dict` / `_plot_alternative_dict` dispatch under multiclass all-classes output.

3. **Nomenclature cleanup**
   - Normalize internal variable names from `multi_lables_*` to `multi_labels_*` throughout code comments/docs.

4. **User-facing guidance**
   - Add a short section in the CE-first docs showing exact invocation examples for all-classes multiclass explanations and plotting.
