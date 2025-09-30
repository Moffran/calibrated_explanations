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

Amendment: Global plots and PlotSpec precisions (updated 2025-09-30)
-----------------------------------------------------------------
This ADR is amended to explicitly include global plots and to add
precise PlotSpec, axis-id and primitives conventions required to ensure
full parity with the legacy v0.5.1 plotting behaviour. The edits
below codify the decisions required to make adapters and tests
deterministic and reproducible across binary and multiclass use-cases.

1) Global plots
   - New canonical kinds `global_probabilistic` and `global_regression`
     are included (map to legacy `_plot_global`). These kinds MUST
     preserve legacy semantics: scatter of uncertainty vs probability
     (or prediction), color/marker mapping by `y_test` when provided,
     and the regularized branch that draws the probability triangle.
   - The triangle for regularized global plots is a full-figure
     background (axis id `triangle`) rather than a header subplot.

2) PlotSpec additions and clarifications
   - Required fields: PlotSpec MUST include the following fields where
     applicable:
     - `kind` (see canonical kinds in this ADR)
     - `mode`: 'classification' | 'regression'
     - `threshold`: None | scalar | [low, high]
     - `class_index`: None | int (when applicable)
     - `feature_order`: list[int] (indices used for plotting, deterministic order)
     - `feature_names`: optional list[str] aligned to dataset columns
     - `y_minmax`: [float, float] (fallback used when predict low/high is infinite)
     - `confidence`: optional numeric used for regression header text
     - `save_behavior`: { path: str, title: str, default_exts: [str], save_global: bool }
     - `labels`: optional dict mapping axis_id -> human-readable label
     - `color_role_map`: map per plot-kind of { positive: role_name, negative: role_name }
     - `legacy_color_mode`: optional bool (when True adapters attempt exact legacy hex reproduction)

   - Header semantics: `PlotSpec.header` is an ordered list of header
     axis identifiers (e.g., `['header.0','header.1']` for legacy
     two-class). For legacy convenience, aliases `header.neg` ->
     `header.0` and `header.pos` -> `header.1` are recognized. Builders
     for multiclass MUST populate header entries for each class in the
     canonical class order.

   - Body/style/rank fields described earlier are retained; the above
     fields are additive and required for deterministic adapter output.

3) Uncertainty visual parity (precise legacy rules)
   - If `uncertainty == False`: draw solid bars from zero to the
     feature weight (opaque bar).
   - If `uncertainty == True`: compute interval [wl, wh] for each
     feature.
     - If the interval does not cross zero: draw a translucent overlay
       between wl and wh (alpha ~= 0.2) and also draw the opaque solid
       bar from 0 to the point estimate (legacy behaviour).
     - If the interval crosses zero (wl < 0 < wh): do **not** draw an
       opaque solid bar. Instead:
         * classification mode: draw two translucent overlays split at
           zero: [wl, 0] colored by the negative role (alpha ~= 0.2) and
           [0, wh] colored by the positive role (alpha ~= 0.2).
         * regression mode: draw a single translucent overlay spanning
           [wl, wh] (alpha ~= 0.2).
   - Adapters MUST implement these exact semantics for legacy-parity
     mode.

4) Colors and the legacy color algorithm
   - PlotSpec includes `color_role_map` which assigns semantic roles
     for positive/negative visual states per plot kind. Example defaults
     for legacy parity:
       * `factual_probabilistic`: positive -> 'positive_fill' (red), negative -> 'negative_fill' (blue)
       * `factual_regression`: positive -> 'negative_fill' (blue), negative -> 'positive_fill' (red)  (legacy sign inversion)
   - Tests SHOULD validate color role and alpha semantics by default.
     A `legacy_color_mode` boolean in PlotSpec can be set to True to
     request exact legacy hex color reproduction (the canonical
     algorithm is implemented in `coloring.py`). Adapters MAY support
     exact-hex reproduction when legacy_color_mode is True but MUST
     export role-based colors otherwise.

5) Adapter primitives contract (export_drawn_primitives)
   - Adapters SHOULD implement `export_drawn_primitives()` that returns
     a JSON-serializable dict with keys `plot_spec` and `primitives`.
   - Axis id conventions (canonical):
     - header axes: `header.<index>` (e.g., header.0, header.1, ...);
       convenience aliases header.neg/header.pos map to header.0/1
       respectively for legacy two-class plots.
     - main plotting axis: `main`
     - twin axis used for instance value labels: `main.values`
     - triangle full-figure background: `triangle`
     - filesystem/save primitives: `filesystem`
   - Each primitive MUST include:
     - `axis_id`: one of the axis ids above
     - `type`: primitive type (e.g., 'fill_betweenx','scatter','quiver','triangle_background','save_fig','ytick_labels')
     - `coords`: data-space coordinates (numbers, ranges)
     - `visual`: { color_role: str, color_hex?: str, alpha?: float, marker?: str, size?: number }
     - `meta`: optional human-readable metadata (e.g., feature_index, label text)
   - Coordinates MUST be in data-space (not pixels). Primitives MUST
     include human-readable axis label strings (either in PlotSpec.labels
     or in primitive.meta labels).

6) Global plots specifics
   - For `global_probabilistic` regularized branch, adapters MUST emit a
     `triangle_background` primitive on axis `triangle` and quiver/scatter
     primitives on `main` as in legacy `_plot_global` behaviour.
   - For non-regularized branch adapters MUST emit scatter primitives on
     `main` and, when `save_behavior.save_global` is True, MUST also
     emit `save_fig` primitives on axis `filesystem` for each configured
     extension. The adapter MUST normalize the filesystem path (do not
     rely on client trailing separators) and join path/title via a safe
     path-join operation.

7) Labels and test behavior
   - PlotSpec may include exact `labels` (axis_id -> string). If
     provided, adapters MUST use those labels in primitive meta. If not
     provided, adapters MUST generate canonical labels using the
     templates described in this ADR (thresholded text, confidence
     strings, class labels, etc.). Tests should prefer exact equality
     when PlotSpec supplies labels and otherwise assert presence of the
     numeric values (threshold/confidence) in generated labels.

8) Feature ordering
   - PlotSpec.feature_order MUST be a list of integer column indices
     matching the indices used by the Explanation (features_to_plot in
     legacy code). PlotSpec.feature_names may be supplied for readable
     tick labels.

9) Miscellaneous recommendations
   - Adapters are encouraged to emit a `legend` primitive listing label
     entries and to provide a `colorbar` primitive only when requested
     by the builder (legacy code had a commented colorbar for regression).
   - Default visual constants (marker sizes, default alphas) used by
     the legacy implementation should be documented in `coloring.py`
     and reused for parity tests.

If conflicts exist between this ADR and other plotting ADRs, this ADR
takes precedence for plotting behaviour and the primitive schema.
