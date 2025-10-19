# ADR-025: Legacy Plot Rendering Semantics

Status: Draft
Date: 2025-02-14
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None
Related: ADR-014, ADR-016, ADR-021, ADR-024

## Context

The default Calibrated Explanations renderer relies on Matplotlib-based helpers
in `legacy.plotting` to communicate probabilities, calibrated intervals, and rule
impacts. These helpers implicitly define the “Calibrated Explanations look and
feel,” including axis structure, colour choices, fill behaviour, and text
labels. As we build a plotspec layer and a new renderer, the existing semantics
must be preserved through formal documentation; otherwise tests cannot verify
that the new implementation reproduces the same layout and styling guarantees.
This ADR records the full rendering logic for each helper so downstream
implementations and automated checks can validate parity.

## Decision

### `_plot_probabilistic`

* **Layout:** constructs a `(10, num_to_show * 0.5 + 2)` inch figure and splits
  it into three stacked subfigures with height ratios `[1, 1, num_to_show + 2]`.
  The first two rows host one-row probability gauges (negative vs. positive
  events) and the bottom subfigure plots feature contributions.【F:src/calibrated_explanations/legacy/plotting.py†L67-L77】
* **Probability gauges:** the negative axis visualises `1 - p` via three
  `fill_betweenx` bands: a solid bar at the point prediction, an opaque band for
  the full interval `[0, 1 - ph]`, and a translucent highlight for the calibrated
  interval `[1 - pl, 1 - ph]`. The positive axis mirrors this with `p`, `[0, pl]`,
  and `[pl, ph]` fills. Axes are clamped to `[0, 1]`; the negative gauge shows
  tick marks at 0.2 increments while the positive gauge hides x ticks.【F:src/calibrated_explanations/legacy/plotting.py†L79-L96】
* **Gauge labelling:** the y-labels depend on explanation metadata. Thresholded
  regressors show threshold comparisons (scalar or interval). When class labels
  are absent the helper falls back to numeric classes (`P(y=0)`/`P(y=1)` in
  binary). Multiclass paths prefer mapped labels from
  `explanation.get_class_labels()` but fall back to predicted class indices.
  Non-thresholded binary classification uses explicit class labels with
  negated/predicted forms. The negative gauge’s x-axis is labelled "Probability"
  while the positive gauge omits it.【F:src/calibrated_explanations/legacy/plotting.py†L98-L135】
* **Base contribution strip:** when features are present the helper draws a
  horizontal zero baseline across the y-span and extends it by half-unit margins
  on both ends using additional `fill_betweenx` calls. In interval mode the
  generalised weight band is shown as a grey translucent region spanning the
  global interval `(predict["low"] - predict["predict"], predict["high"] - predict["predict"])`.【F:src/calibrated_explanations/legacy/plotting.py†L137-L151】
* **Feature bars:** each feature index `j` yields a vertical strip with width
  ±0.2 around its slot. Non-interval mode draws a solid bar from 0 to the signed
  weight, coloured red for positive and blue for negative contributions. Interval
  mode reorders the provided uncertainty bounds to `(min, max)`, splits the fill
  into deterministic (solid) and stochastic (alpha=0.2) regions, and neutralises
  fills when the uncertainty crosses zero; classification interval plots split
  negative and positive translucent bands to emphasise the crossing.【F:src/calibrated_explanations/legacy/plotting.py†L152-L179】
* **Axes and annotations:** the primary y-axis lists feature labels and spans
  `(-0.5, num_to_show - 0.5)`. A twinned y-axis annotates raw instance values for
  the same indices. The x-axis is labelled "Feature weights" and remains symmetric
  around zero by default (no explicit limits).【F:src/calibrated_explanations/legacy/plotting.py†L180-L191】
* **Output hooks:** after drawing, the helper saves to every extension in
  `save_ext` using `path + title + ext` and optionally calls `fig.show()`.
  One-sided explanations reject interval mode with a `Warning` exception; plots
  must reproduce this guard.【F:src/calibrated_explanations/legacy/plotting.py†L70-L71】【F:src/calibrated_explanations/legacy/plotting.py†L192-L195】

### `_plot_regression`

* **Layout:** uses a two-row figure with height ratios `[1, num_to_show + 2]` to
  separate the interval gauge (top) from feature contributions (bottom). Figure
  size matches the probabilistic variant.【F:src/calibrated_explanations/legacy/plotting.py†L224-L231】
* **Interval gauge:** plots the calibrated range `[pl, ph]` as a translucent red
  band and overlays the median `p` as a solid red bar. The x-span is clamped to
  the min/max of the interval versus the training extrema stored on
  `explanation.y_minmax`. The sole y-tick is labelled "Median prediction", and
  the x-axis caption embeds `get_confidence()` from the explanation bundle.【F:src/calibrated_explanations/legacy/plotting.py†L233-L250】
* **Feature contributions:** identical positioning to `_plot_probabilistic`
  except the sign colouring flips: positive weights are blue and negative weights
  red. Interval mode mirrors the probabilistic logic (reordered bounds, alpha
  overlay) and tracks running min/max to set symmetric x-limits around all
  contributions. Without intervals the helper still records the extrema to scale
  the axis.【F:src/calibrated_explanations/legacy/plotting.py†L252-L307】
* **Axes and annotations:** the y-axis caption becomes "Rules" (since regression
  explanations often represent rule antecedents). A twin axis again surfaces raw
  instance values. Interval validation and save/show behaviour match the
  probabilistic helper.【F:src/calibrated_explanations/legacy/plotting.py†L299-L315】

### `_plot_triangular`

* **Mode-dependent backdrop:** draws either the probability simplex (classification
  or thresholded regression) using `__plot_proba_triangle()` or a blank Cartesian
  plot for unthresholded regression. The simplex helper traces three bounding
  lines parameterised by rational curves to emulate the triangular diagram used
  in original CE docs.【F:src/calibrated_explanations/legacy/plotting.py†L345-L365】【F:src/calibrated_explanations/legacy/plotting.py†L404-L412】
* **Axis scaling:** regression mode widens both axes to include a 10% margin
  beyond the extrema of the rule predictions and uncertainties; when the span is
  effectively zero it emits a warning about identical uncertainties. Probabilistic
  mode keeps `[0, 1]` bounds for probability and uncertainty.【F:src/calibrated_explanations/legacy/plotting.py†L351-L365】
* **Vectors and scatter:** renders grey quiver arrows from the original point
  `(proba, uncertainty)` to the first `num_to_show` alternative rules, then plots
  all rule points as grey dots (marker size 50) and the original point as a red
  dot. Labels "Alternative Explanations" and the axis captions switch between
  "Probability" and "Prediction" depending on mode.【F:src/calibrated_explanations/legacy/plotting.py†L367-L393】
* **Legend and persistence:** always adds a legend distinguishing the original
  prediction from alternatives, writes files for each requested extension, and
  optionally calls `plt.show()`.【F:src/calibrated_explanations/legacy/plotting.py†L395-L401】

### `_plot_alternative`

* **Layout:** single `(10, num_to_show * 0.5)` inch plot showing stacked bars for
  the baseline prediction and alternative rule intervals. X positions match the
  feature ordering so the y-axis enumerates alternative rules.【F:src/calibrated_explanations/legacy/plotting.py†L437-L513】
* **Original interval fill:** computes the Venn-Abers interval `[p_l, p_h]` for
  the base prediction. If the interval lives entirely on one side of 0.5 (or the
  explanation mode includes "regression") the helper fills the entire band with a
  colour derived from `__get_fill_color`; regression mode overlays a red median
  line. If the interval straddles 0.5 it splits the band into two fills anchored
  at 0.5 with colours corresponding to the lower and upper halves, producing the
  familiar CE red/blue split.【F:src/calibrated_explanations/legacy/plotting.py†L440-L479】
* **Feature-specific fills:** for each alternative rule the helper repeats the
  interval logic at the rule level. Regression intervals become translucent red
  bands with a solid median line. Classification intervals that stay on one side
  of 0.5 receive a solid fill; crossing intervals are split at 0.5 with separate
  colours for the lower and upper portions.【F:src/calibrated_explanations/legacy/plotting.py†L481-L505】
* **Axis annotations:** left y-axis labelled "Alternative rules" with tick labels
  from `column_names` when available; the twin y-axis shows raw instance values
  for the same indices. The x-axis caption depends on context: thresholded
  regression spells out the probability of the event, regression shows the
  calibrated confidence interval text, and classification references either the
  positive class or the predicted class label depending on metadata availability.
  All threshold and classification branches clamp the x-axis to `[0, 1]` with ten
  evenly spaced ticks.【F:src/calibrated_explanations/legacy/plotting.py†L507-L555】
* **Tight layout & persistence:** wraps `fig.tight_layout()` in
  `contextlib.suppress` to avoid exceptions, then saves each extension and
  optionally displays the figure.【F:src/calibrated_explanations/legacy/plotting.py†L557-L562】
* **Colour generation:** helper `__color_brew(2)` constructs two RGB colours by
  sweeping hues and reversing the order. `__get_fill_color` selects the winning
  class (>= 0.5 probability) and blends it against white using an alpha computed
  from the predicted probability (optionally overridden by a `reduction`
  parameter). This reproduces the soft red/blue palette used across CE plots and
  underpins every fill in `_plot_alternative`.【F:src/calibrated_explanations/legacy/plotting.py†L746-L785】

### `_plot_global`

* **Prediction sources:** chooses between `explainer.predict` (non-probabilistic)
  and `explainer.predict_proba` (probabilistic or thresholded) based on the
  presence of `predict_proba` on the learner and the optional `threshold`. It then
  constructs uncertainty as `high - low`. This branching determines whether the
  plot is in simplex mode or Cartesian mode.【F:src/calibrated_explanations/legacy/plotting.py†L589-L605】
* **Backdrops:** probabilistic flows reuse `_plot_proba_triangle()` to render the
  simplex background in a new figure, matching the triangular view from the
  alternative plot. Non-probabilistic flows create a standard Matplotlib axes and
  compute 10% margins around the historical calibration targets (`explainer.y_cal`)
  and uncertainty range. Degenerate spans trigger the same warning about identical
  uncertainties.【F:src/calibrated_explanations/legacy/plotting.py†L600-L621】
* **Scatter logic:**
  * Without `y`, the helper plots predictions as grey points (marker size 50).
    Multiclass probabilities collapse to the predicted class component.
  * With `y` and non-probabilistic predictions, points are coloured via the
    `viridis` colormap relative to target values. The helper assumes an axes handle
    exists (hence the dedicated subplot path).
  * With `y` and probabilistic outputs, the helper builds per-class scatter series
    using either a blue/red pair (binary) or the `tab10` colormap with repeated
    marker shapes for additional classes. Probabilities and uncertainties are
    indexed by the ground-truth labels to display class-conditional calibration.【F:src/calibrated_explanations/legacy/plotting.py†L631-L712】
* **Labelling:** axis captions pivot on mode. Non-probabilistic plots label both
  axes explicitly (“Predictions” / “Uncertainty”). Thresholded regression records
  the probability event text, and multiclass runs switch between predicted and
  actual class references depending on whether `y` is provided. Binary
  probabilistic plots label the x-axis `Probability of Y = 1`. Axes limits follow
  the computed margins.【F:src/calibrated_explanations/legacy/plotting.py†L713-L732】
* **Interactivity:** unlike other helpers `_plot_global` always calls
  `plt.show()` at the end (there is no save loop). Consumers should reproduce this
  to match interactive behaviour, including the headless guard when `show=False`
  is passed through `kwargs`.【F:src/calibrated_explanations/legacy/plotting.py†L584-L605】【F:src/calibrated_explanations/legacy/plotting.py†L731-L732】

### Probability simplex helpers

* `__plot_proba_triangle()` (internal to `_plot_triangular`) draws three lines:
  `(x/(1+x), x)`, `(x, (1-x)/x)`, and offsets around 0.5 to approximate the
  triangular feasible region for binary probabilities and uncertainties. The
  top-level `_plot_proba_triangle()` used by `_plot_global` repeats the same
  curves but starts by instantiating a new figure. This duplication reflects the
  legacy renderer’s habit of redrawing the simplex when entering global mode.【F:src/calibrated_explanations/legacy/plotting.py†L404-L412】【F:src/calibrated_explanations/legacy/plotting.py†L735-L743】

## Consequences

* Renderer reimplementations can now adopt deterministic snapshot tests that
  assert on axis counts, label text, colour choices, and marker usage, ensuring
  fidelity with the legacy visuals.
* Any change to the default colour palette, layout, or labelling must be
  accompanied by an ADR update and corresponding golden-image refresh in the new
  renderer.
* Documented semantics clarify why two simplex helpers exist, preventing future
  “cleanups” that would otherwise break `_plot_global`’s implicit behaviour.
