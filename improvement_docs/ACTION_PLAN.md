# Action Plan — Outstanding Work Aligned to Releases

Created: August 16, 2025
Last Updated: September 19, 2025
Scope: Not‑yet‑achieved items organized by target release. See `CHANGELOG.md` for completed work; see `improvement_docs/DEVELOPMENT_PLAN.md` for strategy.

## CI Snapshot
- Reference run (2025‑09‑11): 150 passed, 1 xpassed, 22 warnings in ~371s. Not re‑run in this pass.

## v0.6.x (stabilization — ongoing patch series)
- Keep behavior stable; no new feature defaults.
- Housekeeping for groundwork already merged (docs polish, small test covers).

Release gate
- No user‑visible changes beyond docs/tests.

## v0.7.0 (performance + docs + PlotSpec routing)

- Visualization (ADR‑014)
  - Convert at least one classification/probabilistic plot to PlotSpec with parity tests vs legacy `_plots.py`.
  - Add opt‑in routing from `Explanation.plot()` to PlotSpec; keep legacy path as default.
  - Unify style mapping between legacy plots and the adapter; document any intentional differences.

- Performance (ADR‑003/ADR‑004)
  - Keep caching/parallel behind flags; add one end‑to‑end example and short docs.
  - Wire micro‑bench checker in CI using `scripts/micro_bench_perf.py` and `scripts/check_perf_micro.py` with thresholds.

- Docs
  - Refresh API/gallery; ensure linkcheck remains green; add PlotSpec usage page updates.

Release gate
- Parity tests pass for the converted plot; feature flag toggles PlotSpec routing safely.
- CI perf guard enforced; example demonstrates safe, opt‑in perf usage.
- Docs refreshed and linkcheck passes.

## v0.8.0 (extensibility + viz by default)

- Interval Calibrator Plugins (ADR‑013)
  - Define classification/regression calibrator protocols that compose VennAbers/IntervalRegressor.
  - Extend plugin registry with interval plugin helpers (register/find/trusted) and metadata validation.
  - Provide a minimal interval plugin and contract‑level tests.

- Explanation Plugins (ADR‑015)
  - Finalize JSON/domain batch output contracts for `finalize(...)`/`finalize_fast(...)`.
  - Implement predict/finalize bridges; add a minimal example plugin and an integration test.
  - Document plugin risk/trust and usage.

- Visualization defaulting (ADR‑014)
  - Route a subset of plots through PlotSpec by default; legacy path remains available.

Release gate
- Interval/explanation plugin examples pass contract and integration tests on fixtures.
- PlotSpec becomes default for selected plots; golden parity holds.
- Trust model and usage clearly documented.

## Dependencies & Risks
- Plugins: discovery remains explicit/in‑process; trusted‑only finders are the recommended default.
- Visual parity: minor stylistic drift acceptable if documented; avoid semantic changes to bar/interval semantics.
