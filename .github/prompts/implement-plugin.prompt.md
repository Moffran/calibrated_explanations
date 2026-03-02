# /implement-plugin

Scaffold a new CE plugin following the ADR-first, TDD workflow.

## Before you start

Read the relevant ADRs in `docs/improvement/adrs/`:
- **ADR-006** – Plugin trust model and registry rules.
- **ADR-013** – Interval calibrator plugin strategy (if calibrator plugin).
- **ADR-014** – Visualization plugin architecture (if plot plugin).
- **ADR-015 / ADR-026** – Explanation plugin integration and semantics (if explanation plugin).

## Steps Copilot must follow

1. **Identify plugin type** – calibrator, plot, or explanation.
2. **Define the Protocol** – extend the matching `typing.Protocol` in `src/calibrated_explanations/plugins/`.
3. **Register the plugin** – add an entry to the plugin registry; respect the trust model (ADR-006).
4. **Write failing tests first (TDD red)** – follow `tests/README.md`; place tests in `tests/unit/plugins/test_<plugin_name>.py` or the nearest existing file.
5. **Implement the plugin (TDD green)** – create `src/calibrated_explanations/plugins/<plugin_name>.py` with Numpy-style docstrings (`from __future__ import annotations`).
6. **Refactor** – ensure no circular imports; use lazy imports for heavy libs.
7. **Emit a fallback warning if appropriate** – follow the Fallback Visibility Policy in `copilot-instructions.md` §7.
8. **Update `CHANGELOG.md`** and, if an ADR is affected, update its status section.

## Inputs (optional)

- `plugin_type=calibrator|plot|explanation`
- `plugin_name=<snake_case_name>`
- `target_adr=<ADR-NNN>` (cite the ADR that governs this plugin)

## Checklist on completion

- [ ] Protocol extended and registry entry added.
- [ ] Unit tests written and passing (`make test`).
- [ ] Docstrings present (Numpy style).
- [ ] No new eager imports of matplotlib/pandas in `__init__.py`.
- [ ] `CHANGELOG.md` updated.
