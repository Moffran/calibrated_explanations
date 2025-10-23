# ADR-024: Legacy Plot Input Contracts

Status: Draft
Date: 2025-10-18
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None
Related: ADR-014, ADR-016, ADR-020, ADR-021

## Context

Calibrated Explanations currently exposes five legacy plotting entry points
through `calibrated_explanations.legacy.plotting`: `_plot_probabilistic`,
`_plot_regression`, `_plot_triangular`, `_plot_alternative`, and `_plot_global`.
Each plotting helper accepts a bespoke collection of arguments pulled from the
explanation runtime. The upcoming plot specification (plotspec) system needs a
precise contract describing which inputs are required, their types, and how the
legacy renderer interprets them. Without a canonical reference, new plotspec
producers risk omitting required values (e.g., calibrated intervals, feature
weights, class labels) or mis-shaping payloads (for example, confusing
probability dictionaries with raw arrays). This ADR captures the exact input
structure, optional behaviours, and error guards enforced by the legacy
plotting layer so that plotspec definitions can be constructed faithfully, and
supplies enough colour/figure metadata to recreate the historical imagery.

## Decision

The legacy plotting helpers consume structured inputs that must be replicated by
any plotspec generator. The contracts below describe every positional argument,
its expected structure, and any derived values the renderer pulls from the
`CalibratedExplanation`/`CalibratedExplainer` context.

### Shared plotting mechanics
* All helpers defer to `__require_matplotlib()` once rendering is requested so
  optional dependencies stay lazy. Calls with `show=False` **and** missing save
  metadata (`path`/`title`) short-circuit before the guard, enabling headless
  execution without the viz extra. `_plot_global` reads `show` from `kwargs`;
  the remaining helpers take it positionally.【F:src/calibrated_explanations/legacy/plotting.py†L24-L65】【F:src/calibrated_explanations/legacy/plotting.py†L192-L195】【F:src/calibrated_explanations/legacy/plotting.py†L331-L339】【F:src/calibrated_explanations/legacy/plotting.py†L584-L605】
* `save_ext` defaults to `("svg", "pdf", "png")` (lists in code) and is
  iterated verbatim when saving. Filenames are produced via `path + title + ext`
  without inserting separators, so builders must ensure any directory separator
  or dot is already embedded in `path` or `title`.【F:src/calibrated_explanations/legacy/plotting.py†L63-L65】【F:src/calibrated_explanations/legacy/plotting.py†L193-L195】【F:src/calibrated_explanations/legacy/plotting.py†L398-L401】【F:src/calibrated_explanations/legacy/plotting.py†L559-L562】
### Instance plots
The instance-oriented helpers (`_plot_probabilistic`, `_plot_regression`,
`_plot_triangular`, `_plot_alternative`) render per-explanation artefacts.
Unless stated otherwise, they expect a single explanation object, a matching
instance vector, feature selections, and the display controls described below.
#### Shared instance plot contract
* **Figure sizing**: Each helper scales the figure height with `num_to_show`
  using `figsize=(10, num_to_show * 0.5 + offset)`. Mismatched feature counts
  therefore distort the vertical layout and must be prevented by the caller.【F:src/calibrated_explanations/legacy/plotting.py†L63-L68】【F:src/calibrated_explanations/legacy/plotting.py†L214-L225】【F:src/calibrated_explanations/legacy/plotting.py†L398-L401】
* **Feature selection**: `features_to_plot` controls both plotting order and
  axis labelling. The helpers iterate `enumerate(features_to_plot)` so order is
  preserved verbatim.【F:src/calibrated_explanations/legacy/plotting.py†L152-L191】
* **Instance values**: `instance` must align index-wise with
  `features_to_plot`. The vector is reused to annotate twin axes or alternative
  overlays depending on the helper.【F:src/calibrated_explanations/legacy/plotting.py†L187-L191】【F:src/calibrated_explanations/legacy/plotting.py†L440-L505】
* **Column labels**: Optional `column_names` sequences map feature indices to
  display names. `None` leaves numeric ticks in place.【F:src/calibrated_explanations/legacy/plotting.py†L180-L186】【F:src/calibrated_explanations/legacy/plotting.py†L440-L505】
* **Display controls**: `title`, `path`, `show`, and `save_ext` share identical
  semantics across helpers, including headless no-op behaviour when `show=False`
  and save metadata is incomplete.【F:src/calibrated_explanations/legacy/plotting.py†L57-L65】【F:src/calibrated_explanations/legacy/plotting.py†L192-L195】【F:src/calibrated_explanations/legacy/plotting.py†L331-L399】
* **Interval gating**: Interval rendering requires `interval=True`, a non-null
  `idx`, and `explanation.is_one_sided()` to be false. The helpers also demand
  interval-shaped payloads for feature weights or predictions when the flag is
  active.【F:src/calibrated_explanations/legacy/plotting.py†L65-L72】【F:src/calibrated_explanations/legacy/plotting.py†L156-L229】
#### `_plot_probabilistic`
* **Explanation handle**: Must expose `is_one_sided()`, `is_thresholded()`,
  `get_class_labels()`, `prediction`, `y_minmax`, `y_threshold`, `is_multiclass`,
  `_get_explainer().is_multiclass()`, and `get_mode()` for axis labelling,
  palette selection, and interval guards.【F:src/calibrated_explanations/legacy/plotting.py†L70-L133】【F:src/calibrated_explanations/legacy/plotting.py†L171-L178】
* **Prediction summary**: `predict` is a mapping with `"predict"`, `"low"`, and
  `"high"` keys. Infinite bounds are replaced with `explanation.y_minmax` limits
  during rendering.【F:src/calibrated_explanations/legacy/plotting.py†L81-L83】
* **Feature weights**: Accepts either a 1-D array (standard mode) or a mapping
  with `"predict"`, `"low"`, and `"high"` arrays (interval mode). The renderer
  shades red/blue lobes based on sign changes and clamps mixed spans to avoid
  solid fills.【F:src/calibrated_explanations/legacy/plotting.py†L152-L178】
* **Display budget**: `num_to_show` controls axis ranges and figure sizing, so
  it must match the available feature data. Zero suppresses feature bars while
  retaining probability gauges.【F:src/calibrated_explanations/legacy/plotting.py†L63-L191】
#### `_plot_regression`
* **Prediction summary**: Mirrors `_plot_probabilistic` with `"predict"`,
  `"low"`, and `"high"` entries and swaps infinite bounds for
  `explanation.y_minmax`.【F:src/calibrated_explanations/legacy/plotting.py†L233-L239】
* **Feature weights**: Shares the vector/dictionary behaviour but shades positive
  spans blue and negative spans red to reproduce the regression palette.【F:src/calibrated_explanations/legacy/plotting.py†L269-L296】
* **Header metadata**: Requires access to
  `explanation.calibrated_explanations.get_confidence()` and `y_minmax` to build
  the interval summaries shown above the bars. Interval widths subtract the
  median prediction before shading.【F:src/calibrated_explanations/legacy/plotting.py†L227-L295】
#### `_plot_triangular`
* **Instance summary**: Needs scalar `proba`/`uncertainty` for the primary
  instance and array-like `rule_proba`/`rule_uncertainty` for alternatives. The
  arrays must be at least `num_to_show` long because the helper slices them
  during arrow rendering.【F:src/calibrated_explanations/legacy/plotting.py†L367-L383】
* **Explanation handle**: Must expose `get_mode()`, `is_thresholded()`, and
  `y_minmax` so regression mode can reuse prediction bounds and classification
  mode can render the probability simplex provided by
  `__plot_proba_triangle()`.【F:src/calibrated_explanations/legacy/plotting.py†L345-L365】
#### `_plot_alternative`
* **Prediction payloads**: `predict` and `feature_predict` reuse the
  `_plot_probabilistic` structure, including the interval dictionaries and
  infinite-bound substitution. Per-feature arrays must align with
  `features_to_plot`.【F:src/calibrated_explanations/legacy/plotting.py†L440-L505】
* **Explanation handle**: Requires `get_mode()`, `is_thresholded()`, `y_minmax`,
  `calibrated_explanations.get_confidence()`, `_get_explainer().is_multiclass()`,
  `prediction`, and `get_class_labels()` so the helper can label axes and choose
  probability palettes for both regression and classification flows.【F:src/calibrated_explanations/legacy/plotting.py†L451-L555】
* **Colour handling**: Relies on `__color_brew()` and `__get_fill_color()` to
  blend palette colours against white based on the Venn-Abers mass. Payloads must
  therefore capture predictions and bounds to recreate the fills.【F:src/calibrated_explanations/legacy/plotting.py†L441-L521】【F:src/calibrated_explanations/legacy/plotting.py†L747-L785】
### Batch plots
`_plot_global` is the sole batch helper and operates on collections of
instances rather than a single explanation.
#### `_plot_global`
* **Explainer**: Must expose `.learner`, `.predict()`, `.predict_proba()`,
  `.is_multiclass()`, `.class_labels`, and `.y_cal`. The helper inspects
  `dir(explainer.learner)` to determine probabilistic support and falls back to
  raw predictions otherwise. Scalar thresholds are required for non-probabilistic
  regression calls.【F:src/calibrated_explanations/legacy/plotting.py†L589-L666】
* **Input data**: Array-like `x` is forwarded to the explainer; optional `y`
  provides colouring/label grouping and must align with the prediction shape.
  Classification mode expects integer or label arrays for probability lookups.【F:src/calibrated_explanations/legacy/plotting.py†L631-L712】
* **Threshold**: Optional scalar or tuple forwarded to `predict_proba`. In
  non-probabilistic flows the value must be scalar to satisfy the runtime
  assertion, so plotspecs should enforce the same restriction.【F:src/calibrated_explanations/legacy/plotting.py†L659-L666】
* **Display controls**: Accepts `show` via `kwargs` with the same headless
  no-op semantics described earlier. Any plotspec generator must expose the flag
  to keep CI runs silent without the viz extra.【F:src/calibrated_explanations/legacy/plotting.py†L584-L605】
* **Uncertainty construction**: Builds uncertainty as `high - low` from the
  prediction tuple, so routines must emit identically shaped arrays. Scatter
  plots expect those arrays to align with the first dimension of `x`/`y`, and
  regression axes expand by 10% of the observed range, mirroring the triangular
  plot padding.【F:src/calibrated_explanations/legacy/plotting.py†L592-L641】【F:src/calibrated_explanations/legacy/plotting.py†L631-L706】
### Helper colour derivation
* `_plot_proba_triangle()` draws the probability simplex background leveraged by
  both `_plot_triangular` and `_plot_global` whenever probabilistic rendering is
  active.【F:src/calibrated_explanations/legacy/plotting.py†L320-L401】【F:src/calibrated_explanations/legacy/plotting.py†L590-L634】
* `__color_brew(n)` returns integer RGB triplets tuned to the legacy palette;
  `_plot_alternative` only ever requests two slots and then blends against white
  via `__get_fill_color()` to achieve semi-transparent fills.【F:src/calibrated_explanations/legacy/plotting.py†L747-L785】【F:src/calibrated_explanations/legacy/plotting.py†L441-L521】
## Consequences
* Plotspec authors now have a canonical reference for every data payload the
  legacy renderer consumes, allowing them to design serializable plot
  definitions that capture the same information.
* Input validation in new systems can mirror the documented assertions (e.g.,
  `idx` presence for interval plots, scalar threshold enforcement in global
  regression) to maintain parity with legacy expectations.
* Future ADRs may extend this document if additional plotting helpers are
  restored from history or if new inputs (such as custom colour palettes) become
  first-class configuration items.
