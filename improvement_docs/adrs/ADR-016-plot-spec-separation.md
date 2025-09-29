# ADR 016: Plot spec separation and uncertainty parity

Status: Proposed

Context
-------
The plotting code in v0.5.1 multiplexed multiple plotting modes (factual vs
alternative, probabilistic vs regression) into a small set of functions. This
led to complex branching, implicit color-role swaps, and brittle visual
semantics. Tests did not capture intended visual invariants (e.g. parity
between uncertainty on/off), allowing regressions.

Decision
--------
1. Introduce explicit plot kinds: `factual_probabilistic`,
   `alternative_probabilistic`, `factual_regression`, `alternative_regression`.
   Each plot kind has a focused builder and renderer.
   `triangular` is a special plot that draws a special kind of alternative plot for probabilistic explanations. It is invoked by the style parameter `style=triangular`, following the legacy logic from v0.5.1.
   Additional plots can be added later.
2. Make semantics explicit in `PlotSpec` metadata:
   - header: `dual` (default factual probabilistic), `single` (default factual regression), `none` (default alternatives, and triangular), `...` (placeholder for future headers, e.g. true multiclass)
   - body: `weight` (default factual and fast), `predict` (default alternative, and triangular), `...` (placeholder for future plots)
   - style: `regular` (default factual, alternative, and fast), `triangular` (triangular), `...` (placeholder for future styles).
   - uncertainty: `true`/`false` (whether uncertainty is drawn, applicable to factual plots, default=false).
   - rank_by: `ensured` (default alternative), `feature_weight` (default factual and fast), `uncertainty`, `...` (placeholder for future ranking metrics).
   - rank_weight : float [-1, 1] (Used with the `ensured` ranking metric, the weight of the uncertainty in the ranking, default=0.5).
3. Adopt default legacy behaviour: if uncertainty=False, solid bars are drawn from 0 to the feature weight value; if uncertainty=True, solid bars are drawn to the uncertainty-area, and translucent uncertainty overlays replace the solid bars and cover the uncertainty area. If the uncertainty interval crosses zero, the solid bar is suppressed to avoid visual confusion. This matches the existing behaviour in v0.5.1. Only the legacy visual mode is supported by `BarItem` and `BarHPanelSpec`.
4. Add an adapter test hook `export_drawn_primitives` that returns a
   backend-agnostic description of drawn primitives (solids/overlays/header) for robust testing.
5. Add adapter primitive tests to assert legacy visual mode and parity between uncertainty on/off for each plot kind.
6. Add additional adapter options, allowing dynamic configuration of color roles and uncertainty visual style.

Consequences
------------
- Pros:
  - Clear separation reduces bugs and simplifies adapter implementations.
  - Tests can assert drawing semantics reliably via primitives export.
  - Legacy behaviour can be reproduced when needed.
- Cons:
  - Small API surface growth (PlotSpec metadata and adapter options).
  - Migration needed for consumers who depended on implicit behaviour.

Notes
-----
This ADR is a practical decision to stabilize visuals and tests.
Renderers are split into small functions per plot kind and the
builders updated to return consistent `PlotSpec` metadata.
