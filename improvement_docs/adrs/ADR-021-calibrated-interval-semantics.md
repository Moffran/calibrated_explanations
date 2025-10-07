# ADR-021: Calibrated Interval Semantics

Status: Draft
Date: 2025-10-07
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None
Related: ADR-006, ADR-008, ADR-013, ADR-015

## Context

The release plan for v0.8.0 exposed a documentation gap around how
`CalibratedExplainer` produces calibrated predictions and intervals across
three distinct usage patterns:

1. **Probabilistic classification** (binary and multiclass) uses Venn-Abers
   calibration exclusively.
2. **Percentile-driven regression** produces conformal intervals based on the
   `low_high_percentiles` argument (default `(5, 95)`).
3. **Thresholded regression** returns calibrated probabilities for threshold
   events (`y \leq t` or `t_0 < y \leq t_1`) and is the only regression path that
   relies on both conformal predictive systems (CPS) *and* Venn-Abers.

When task 2 in the v0.8.0 gap analysis asked to “Ensure explain* APIs emit
CE-formatted intervals when percentile arguments are provided,” the request was
interpreted as “retain percentile metadata for CE intervals.” The misreading
stems from the lack of a single reference explaining how each pathway
constructs predictions, how interval metadata is serialized, and which
calibration engine supplies the numbers. This ADR documents the current
behaviour so future work can preserve the intended semantics and avoid drifting
between the three paths.

## Decision

### 1. Classification: Venn-Abers across predictions and explanations

* `CalibratedExplainer` resolves the interval learner through the interval
  plugin registry. For the in-tree defaults the plugin returns a Venn-Abers
  calibrator regardless of binary or multiclass setup, and the instance is
  cached on the explainer.【F:src/calibrated_explanations/core/calibration_helpers.py†L42-L56】
* `CalibratedExplainer._predict` delegates to `VennAbers.predict_proba` in both
  multiclass and binary branches. The helper always requests
  `output_interval=True` so the calibrated prediction, lower bound, and upper
  bound for each class are produced simultaneously.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1372-L1401】
* During explanation generation the first pass through
  `prediction_helpers.explain_predict_step` stores the calibrated probability
  vector under `prediction["__full_probabilities__"]` alongside the class-level
  interval for each queried instance. Subsequent perturbations—emitted by the
  FAST explanation path, which layers a dedicated plugin on top of the legacy
  interval calibrator—reuse the same Venn-Abers calibrator, ensuring
  CE-formatted outputs contain both per-feature impacts and the original
  probability interval metadata.【F:src/calibrated_explanations/core/prediction_helpers.py†L158-L218】
* `CalibratedExplanations` caches the per-instance predictions, intervals, and
  class labels supplied by the Venn-Abers calibrator, making the probability
  cube accessible for downstream helpers (JSON export, `get_confidence`,
  threshold comparisons).【F:src/calibrated_explanations/explanations/explanations.py†L32-L105】

### 2. Regression without thresholds: CPS percentile intervals

* The same plugin resolution yields an `IntervalRegressor` for regression
  modes. Initialization fits a `ConformalPredictiveSystem` (CPS) over the
  calibration residuals, optionally conditioned on Mondrian bins and difficulty
  estimates, and keeps the structure attached to the explainer.【F:src/calibrated_explanations/_interval_regressor.py†L20-L64】
* When `threshold` is `None`, `_predict` validates the requested percentiles,
  converts one- or two-sided bounds into CPS inputs, and delegates to
  `IntervalRegressor.predict_uncertainty`. That method calls the CPS to produce
  `(median, low, high)` triples and returns them in the format expected by the
  explanation pipeline.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1402-L1455】【F:src/calibrated_explanations/_interval_regressor.py†L121-L166】
* `prediction_helpers.initialize_explanation` records the active percentile
  pair on each `CalibratedExplanations` container so downstream consumers know
  the bounds are conformal percentiles rather than probability events. If no
  threshold is supplied, the collection’s `low_high_percentiles` attribute is
  populated for later serialization.【F:src/calibrated_explanations/core/prediction_helpers.py†L82-L110】
* Explanations therefore expose percentile intervals that faithfully mirror the
  CPS output. Perturbation passes, supplied by the FAST explanation
  extension rather than the legacy interval calibrator itself, reuse the same
  calibrator, so the percentile semantics remain consistent even when features
  are altered.

### 3. Thresholded regression: CPS probabilities calibrated by Venn-Abers

* Thresholded regression keeps the `IntervalRegressor` but activates its
  probabilistic path. The explainer first validates and normalizes the
  threshold argument (scalar, tuple, or per-instance sequence) to ensure it can
  be broadcast across perturbations.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1457-L1466】【F:src/calibrated_explanations/core/prediction_helpers.py†L195-L220】
* `IntervalRegressor.predict_probability` splits calibration residuals in two:
  one half fits the CPS (identical to the percentile flow) while the other half
  is reused on demand to fit a Venn-Abers classifier whose targets encode the
  threshold event (`y \leq t` or `t_0 < y \leq t_1`). The CPS supplies calibrated
  probabilities for the event, which become the pseudo-scores passed into
  Venn-Abers; the resulting predictor returns the probability interval along
  with the calibrated mean.【F:src/calibrated_explanations/_interval_regressor.py†L67-L205】【F:src/calibrated_explanations/_interval_regressor.py†L203-L290】
* Because the Venn-Abers step is recomputed whenever the threshold changes,
  `predict_probability` caches the latest threshold under
  `self.current_y_threshold` so the helper can expose a `predict_proba` method
  matching the classification API. This lets the explanation runtime treat the
  output as a probability interval even though the base task is regression.
* Explanation containers preserve the threshold that produced the calibrated
  probabilities (`CalibratedExplanations.y_threshold`) and avoid populating
  percentile metadata, differentiating the payload from standard regression
  intervals. Downstream utilities therefore know to interpret the interval as a
  calibrated probability mass rather than a percentile band.【F:src/calibrated_explanations/explanations/explanations.py†L24-L74】

### 4. Shared guarantees across modes

* Interval learners are resolved once per explainer instance and cached to
  ensure all perturbations for a given explanation share identical calibration
  state. FAST explanations reuse the same mechanics but resolve a dedicated
  plugin identifier so the probabilistic semantics remain intact.【F:src/calibrated_explanations/core/calibration_helpers.py†L42-L69】
* The explanation pipeline always records the raw prediction (`predict`), the
  lower and upper interval bounds (`low`, `high`), and—when available—the full
  class probability cube. This invariant means CE-formatted outputs remain
  stable regardless of the calibration backbone used for the active mode.【F:src/calibrated_explanations/core/prediction_helpers.py†L158-L218】

### 5. Implications for interval plugins

ADR-013 defines the registry contracts that interval plugins must satisfy. This
ADR constrains *how* those plugins behave when implementing the protocols.

* Any plugin advertising **classification support** must deliver the exact
  Venn-Abers semantics documented above: per-class probability vectors and
  interval triples that correspond to calibrated Venn-Abers outputs. Plugins may
  swap the internal implementation but must document the replacement and prove
  that the resulting probabilities respect the same monotonicity and
  normalisation guarantees.
* Plugins providing **percentile regression** must expose conformal intervals
  compatible with the CPS contract. Implementations that substitute alternative
  conformal engines must still accept the percentile inputs defined in
  `CalibratedExplainer`, populate `low_high_percentiles` metadata, and return the
  `(median, low, high)` triple so downstream consumers cannot tell the difference
  from the legacy CPS path.
* Plugins supporting **thresholded regression** must produce calibrated
  probability intervals for threshold events. Implementations may derive those
  probabilities through alternative scoring functions, but they must still
  present the event as a classification task and emit the same `predict`, `low`,
  and `high` probability cube described above. Any deviation (e.g., returning a
  percentile-style band) violates the core semantics and requires a new ADR.
* Plugin documentation MUST reference this ADR to explain how their calibrator
  satisfies the legacy semantics, and new implementations should ship targeted
  tests showing the same payload fields (`low_high_percentiles` or
  `y_threshold`) are populated with the expected meaning.

## Consequences

Positive:

* Contributors have a single reference describing how classification,
  percentile regression, and thresholded regression compose CPS and Venn-Abers.
* Documentation clarifies why percentile metadata is optional (it only applies
  to regression without thresholds) and why thresholded regression emits
  probability-style intervals.
* Downstream features such as telemetry, JSON export, and `get_confidence`
  retain consistent semantics because their expectations are now documented.

Negative / Risks:

* The ADR reflects current behaviour; any future change to CPS/Venn-Abers
  integration requires updating this document to avoid renewed ambiguity.
* No runtime code was altered, so existing misconceptions in older materials
  remain until linked or revised.

## Status & Follow-up

* Adopt this ADR as the canonical explanation of calibrated interval semantics.
* Link developer documentation and release planning templates to this ADR when
  referencing interval behaviour or requesting changes that depend on it.
* Future enhancements (e.g., alternative probabilistic calibrators) must state
  explicitly whether they replace CPS, Venn-Abers, or both within each mode.
