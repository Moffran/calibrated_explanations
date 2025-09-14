# Proposed Test Reorganization Plan

Goal: Move to clear `tests/unit`, `tests/integration`, `tests/e2e` structure and group by logical area (core, viz, api, helpers). Keep changes minimal: create directories and propose moves/renames; perform moves in separate PRs.

Mapping (current file -> proposed new location + rename suggestion)

- tests/test_helper.py -> tests/unit/core/test_helpers.py
  - Rationale: unit tests for utility helpers.

- tests/test_utils_helper_unit.py -> tests/unit/core/test_utils_helper.py
  - Rationale: similarly core utils.

- tests/test_validation_unit.py -> tests/unit/core/test_validation.py
- tests/test_validation_runtime.py -> tests/integration/core/test_validation_runtime.py
  - Rationale: runtime tests can be slightly larger integration-style.

- tests/test_params_canonicalization.py -> tests/unit/api/test_params_canonicalization.py
- tests/test_api_config_unit.py -> tests/unit/api/test_api_config.py
- tests/test_api_config_builder.py -> tests/unit/api/test_api_builder.py
- tests/test_api_snapshot.py -> tests/integration/api/test_api_snapshot.py
  - Rationale: snapshot comparisons are integration-ish and need stable env.

- tests/test_deprecation_import.py -> tests/unit/core/test_deprecation.py
- tests/test_alias_warnings.py -> tests/unit/core/test_alias_warnings.py

- tests/test_exceptions.py -> tests/unit/core/test_exceptions.py
- tests/test_config_validation_utils.py -> tests/unit/core/test_config_validation.py

- tests/test_preprocessor_wiring.py -> tests/integration/core/test_preprocessor_wiring.py
  - Rationale: Preprocessor wiring interacts with transformers and should be integration-scoped.

- tests/test_wrap_regression.py -> tests/integration/core/test_wrap_regression.py
- tests/test_wrap_classification.py -> tests/integration/core/test_wrap_classification.py
- tests/test_regression.py -> tests/integration/core/test_regression.py
- tests/test_classification.py -> tests/integration/core/test_classification.py
  - Rationale: these are heavier integration-style tests using real datasets.

- tests/test_calibration_helpers.py -> tests/unit/core/test_calibration_helpers.py
- tests/test_calibration_helpers.py (if duplicates) -> consolidate

- tests/test_golden_fixture.py -> tests/integration/core/test_golden_fixtures.py
- tests/test_golden_explanations.py -> tests/integration/core/test_golden_explanations.py
- tests/golden_explanation_v1.json -> tests/data/golden/
- tests/data/golden/*.json -> keep in `tests/data/golden/`

- tests/test_prediction_helpers.py -> tests/unit/core/test_prediction_helpers.py
- tests/test_plots_helpers.py -> tests/unit/viz/test_plots_helpers.py
- tests/test_plots_more.py -> tests/integration/viz/test_plots_more.py
- tests/test_plots_integration.py -> tests/integration/viz/test_plots_integration.py
- tests/test_plot_config.py -> tests/unit/viz/test_plot_config.py
- tests/test_plotspec_mvp.py -> tests/unit/viz/test_plotspec_mvp.py
- tests/test_plotspec_serialization.py -> tests/unit/viz/test_plotspec_serialization.py
- tests/test_viz_builders.py -> tests/unit/viz/test_viz_builders.py
- tests/test_viz_serializers_unit.py -> tests/unit/viz/test_viz_serializers.py
- tests/test_viz_* -> group under `tests/unit/viz` or `tests/integration/viz` based on scope

- tests/test_serialization_and_quick.py -> tests/unit/core/test_serialization_quick.py
- tests/test_transform_to_numeric.py -> tests/unit/core/test_transform_to_numeric.py

- tests/test_perf_foundations.py and tests/test_perf_factory.py -> tests/unit/core/test_perf.py OR tests/integration/core/test_perf (depending on external deps). Prefer unit with mocks.
- tests/test_perturbation.py -> tests/unit/core/test_perturbation.py

- tests/test_plugin_registry.py and tests/test_plugins_registry.py -> consolidate to tests/unit/core/test_plugin_registry.py
- tests/plugins/example_plugin.py -> keep in tests/plugins/ for integration plugin discovery tests

- tests/test_validation.py -> tests/integration/core/test_validation_integration.py

- tests/test_helper.py/test_helper units -> group

- tests/test_explanation_parity.py -> tests/integration/core/test_explanation_parity.py
- tests/test_golden_explanations.py -> tests/integration/core/test_golden_explanations.py
- tests/test_golden_fixture.py -> tests/integration/core/test_golden_fixture.py

- tests/test_domain_model_adapters.py -> tests/unit/core/test_domain_model_adapters.py

- tests/test_perf_* tests -> group under tests/unit/core or tests/integration/perf

- tests/test_fast_units.py -> tests/unit/core/test_fast_units.py (keep small fast-checks)

- tests/test_viz_* -> see viz group above

- tests/test_framework.py -> tests/unit/core/test_framework_utils.py or tests/unit/core/test_framework.py
  - Rationale: contains utilities and small helpers

- tests/test_plot_config.py -> tests/unit/viz/test_plot_config.py

Notes and next steps

- Approach: Do not move files in a single large PR. Instead:
  1. Create target directories in a small PR and add `__init__.py` if desired (not necessary for pytest).
  2. Move a small group of related, low-risk files (e.g., `tests/unit/api/*`) in a follow-up PR and run CI.
  3. Consolidate duplicate tests/helpers as you move files.

- Renaming: When renaming, prefer `test_<area>_<behavior>.py` and keep test functions named as `should_<behavior>_when_<condition>` where practical.

- Automation: We can create a script to move files and update imports; propose to run in a branch, run full tests, and open a PR.

- Keep `tests/README.md` and `tests/_fixtures.py` at repo root `tests/` so other tooling can find them easily.

If you'd like, I can now:

- Move a small batch (e.g., `tests/test_api_*` -> `tests/unit/api/`) and update imports.
- Or generate a shell/PowerShell script listing `git mv` commands to perform the reorg.
