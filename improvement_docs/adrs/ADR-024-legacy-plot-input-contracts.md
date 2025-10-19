# ADR-024: Legacy Plot Input Contracts

Status: Draft
Date: 2025-02-14
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None
Related: ADR-014, ADR-016, ADR-021

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
plotting layer so that plotspec definitions can be constructed faithfully.

## Decision

The legacy plotting helpers consume structured inputs that must be replicated by
any plotspec generator. The contracts below describe every positional argument,
its expected structure, and any derived values the renderer pulls from the
`CalibratedExplanation`/`CalibratedExplainer` context.

### `_plot_probabilistic`

* **Explanation handle** (`explanation`): must expose `is_one_sided()`,
  `is_thresholded()`, `get_class_labels()`, `prediction`, `y_minmax`,
  `y_threshold`, `is_multiclass`, and `_get_explainer().is_multiclass()`
  attributes used to tailor labels and guard one-sided interval plots.
  `calibrated_explanations.get_confidence()` is not referenced here, but the
  object must provide `get_mode()` for interval colouring.【F:src/calibrated_explanations/legacy/plotting.py†L70-L133】【F:src/calibrated_explanations/legacy/plotting.py†L171-L178】
* **Instance values** (`instance`): indexable collection used to annotate the
  twin y-axis with per-feature values. The indices must align with
  `features_to_plot` ordering.【F:src/calibrated_explanations/legacy/plotting.py†L187-L191】
* **Prediction summary** (`predict`): mapping with keys `"predict"`, `"low"`,
  and `"high"`. `"low"` may equal `-np.inf` and `"high"` may equal `np.inf`; the
  renderer substitutes `explanation.y_minmax` bounds in those cases.【F:src/calibrated_explanations/legacy/plotting.py†L81-L83】
* **Feature weights** (`feature_weights`): either a 1-D array of signed weights
  (non-interval mode) or a mapping with `"predict"`, `"low"`, and `"high"` keys
  holding per-feature arrays (interval mode). When interval data is supplied the
  renderer expects finite lower/upper arrays aligned with
  `features_to_plot` indices.【F:src/calibrated_explanations/legacy/plotting.py†L156-L178】
* **Feature selection** (`features_to_plot`): ordered iterable of integer
  indices controlling both the plotting order and the per-axis labels.【F:src/calibrated_explanations/legacy/plotting.py†L152-L190】
* **Display budget** (`num_to_show`): integer count of features to display. The
  renderer builds axis ranges and figure sizes directly from this value, so it
  must match the length of `features_to_plot` and available data in
  `feature_weights`/`instance`. Zero suppresses feature plotting while still
  emitting probability gauges.【F:src/calibrated_explanations/legacy/plotting.py†L67-L191】
* **Column labels** (`column_names`): optional sequence that maps feature
  indices to display names. When `None`, y-axis ticks are left numeric.【F:src/calibrated_explanations/legacy/plotting.py†L180-L186】
* **Output controls** (`title`, `path`, `show`, `save_ext`): `title` and `path`
  are string prefixes concatenated for saving. `save_ext` defaults to
  `["svg", "pdf", "png"]`. When `show` is false and either matplotlib is
  unavailable or saving metadata is incomplete (`path`/`title` missing), the call
  becomes a no-op, which plotspec implementations must replicate to keep headless
  flows silent.【F:src/calibrated_explanations/legacy/plotting.py†L57-L65】【F:src/calibrated_explanations/legacy/plotting.py†L192-L195】
* **Interval toggle** (`interval`, `idx`): when `interval` is truthy the caller
  must pass `feature_weights` as a dictionary (see above), and also provide a
  non-`None` `idx` (legacy callers use it to pick interval slices even though the
  renderer only asserts its presence). Interval mode additionally requires
  `explanation.is_one_sided()` to be false; calling plotspecs must mirror this
  guard to avoid unsupported combinations.【F:src/calibrated_explanations/legacy/plotting.py†L65-L71】【F:src/calibrated_explanations/legacy/plotting.py†L156-L167】

### `_plot_regression`

* Shares the core structure with `_plot_probabilistic`: identical handling of
  `instance`, `features_to_plot`, `column_names`, display controls, and interval
  gating. The same no-op logic applies when `show` is false and saving is not
  requested.【F:src/calibrated_explanations/legacy/plotting.py†L214-L226】【F:src/calibrated_explanations/legacy/plotting.py†L312-L315】
* `predict` again requires `"predict"`, `"low"`, and `"high"` entries and uses
  `explanation.y_minmax` when bounds are infinite.【F:src/calibrated_explanations/legacy/plotting.py†L236-L238】
* `feature_weights` follows the same bifurcation: vector in standard mode or
  dictionary for interval mode.【F:src/calibrated_explanations/legacy/plotting.py†L269-L292】
* The regression-specific header axis references
  `explanation.calibrated_explanations.get_confidence()` and the global interval
  span, so the explanation object must expose the nested
  `calibrated_explanations` container with `get_confidence()` and `y_minmax`
  metadata.【F:src/calibrated_explanations/legacy/plotting.py†L233-L250】

### `_plot_triangular`

* Requires a probability/uncertainty pair for the original instance (`proba`,
  `uncertainty` scalars) and arrays for alternative rules (`rule_proba`,
  `rule_uncertainty`). The arrays must be indexable and at least
  `num_to_show` long because the renderer slices to that length during arrow
  plotting.【F:src/calibrated_explanations/legacy/plotting.py†L367-L383】
* `explanation` must provide `get_mode()`, `is_thresholded()`, and the numerical
  range metadata (`y_minmax`) because regression mode reuses prediction extents
  when constructing axes. The helper also consults `explanation.get_mode()` to
  decide whether to render the probability simplex template produced by
  `__plot_proba_triangle()`.【F:src/calibrated_explanations/legacy/plotting.py†L345-L365】
* `title`, `path`, `show`, and `save_ext` follow the same semantics as other
  plots. Plotspecs must preserve the default extensions and headless no-op
  behaviour.【F:src/calibrated_explanations/legacy/plotting.py†L331-L399】

### `_plot_alternative`

* Accepts the same explanation handle, instance vector, `features_to_plot`,
  `column_names`, and display controls as `_plot_probabilistic`. The contract for
  `predict` and `feature_predict` matches `_plot_probabilistic`’s interval-aware
  dictionaries, including `-np.inf`/`np.inf` substitution and per-feature arrays
  keyed by `"predict"`, `"low"`, and `"high"` in interval mode.【F:src/calibrated_explanations/legacy/plotting.py†L440-L505】
* The explanation object must expose `get_mode()`, `is_thresholded()`,
  `y_minmax`, `calibrated_explanations.get_confidence()`,
  `_get_explainer().is_multiclass()`, `prediction`, and `get_class_labels()` for
  labelling the x-axis. Threshold metadata determines which probability labels
  are rendered; regression mode requires calibrated confidence metadata; and
  classification uses either raw class IDs or mapped labels depending on
  availability.【F:src/calibrated_explanations/legacy/plotting.py†L451-L555】

### `_plot_global`

* **Explainer** (`explainer`): must expose `.learner`, `.predict()`,
  `.predict_proba()`, `.is_multiclass()`, `.class_labels`, and `.y_cal`. The
  helper inspects `dir(explainer.learner)` to decide whether probabilistic APIs
  are available and falls back to raw predictions otherwise. Scalar thresholds
  are required for non-probabilistic regression calls.【F:src/calibrated_explanations/legacy/plotting.py†L589-L666】
* **Input data** (`x`): array-like passed straight into the explainer prediction
  routines. Optional `y` allows colouring/label grouping; when provided it must be
  broadcast-compatible with the prediction array, and classification mode expects
  integer or label arrays used for indexing into probabilities.【F:src/calibrated_explanations/legacy/plotting.py†L631-L712】
* **Threshold** (`threshold`): optional scalar or tuple forwarded to
  `predict_proba`. When provided in non-probabilistic flows it must be scalar to
  satisfy the assertion at call time. Plotspecs must mirror this validation to
  avoid mixing per-instance thresholds with the global plot.【F:src/calibrated_explanations/legacy/plotting.py†L659-L666】
* **kwargs**: forwarded to the underlying prediction routine and inspected for a
  `show` flag controlling headless operation. Default `show=True`; when
  `show=False` and matplotlib is unavailable, no work is performed. Any plotspec
  generator must expose a comparable switch so CI runs without viz extras can
  skip renderer setup.【F:src/calibrated_explanations/legacy/plotting.py†L584-L605】
* The helper always constructs uncertainty as `high - low` from the returned
  tuple, so prediction routines must emit two arrays of identical shape. The
  scatter plotting logic expects the prediction/probability arrays to share the
  same leading dimension as `x`/`y`.【F:src/calibrated_explanations/legacy/plotting.py†L592-L641】【F:src/calibrated_explanations/legacy/plotting.py†L701-L711】

### Shared optional dependency handling

All helpers respect a shared pattern: when `show` is false and saving metadata is
incomplete, the call may exit early without requiring matplotlib. When plotting
is requested the helper invokes `__require_matplotlib()` to raise a
user-friendly error if the optional dependency is missing. Plotspec generators
must maintain this behaviour to avoid import-time failures in core-only
installations.【F:src/calibrated_explanations/legacy/plotting.py†L57-L65】【F:src/calibrated_explanations/legacy/plotting.py†L332-L337】【F:src/calibrated_explanations/legacy/plotting.py†L584-L605】

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
