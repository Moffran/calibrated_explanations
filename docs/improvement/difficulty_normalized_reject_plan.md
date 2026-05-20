# Difficulty-Normalized Reject: Current State and Remaining Gap

## Purpose

This note documents the current repository behavior for difficulty-aware interval calibration and reject-option conformal classification, and pinpoints the exact remaining gap before implementing difficulty-normalized reject nonconformity scores.

This document is intentionally implementation-accurate and does not propose runtime changes in this step. It is grounded by:

- ADR-013, which requires interval calibration plugins to preserve the existing Venn-Abers and IntervalRegressor semantics while receiving read-only context.
- ADR-020, which requires preserving the CE-first public API and existing wrapper/explainer lifecycle surface.
- ADR-029, which requires reject to remain opt-in, policy-driven, and backward compatible when reject is not selected.

## Short Answer

`difficulty_estimator` already affects calibrated probabilities because it is accepted by `CalibratedExplainer`, propagated through the interval plugin context, and used inside `VennAbers` before Venn-Abers calibration.

It does not currently normalize reject nonconformity scores by difficulty. The reject path consumes already-calibrated probabilities from the interval learner and then computes reject nonconformity scores only from probability-derived margin, hinge, or interval-width terms. No sigma or difficulty term is applied in `core/reject/orchestrator.py`.

## 1. Where `difficulty_estimator` enters the wrapper/calibration API

At the wrapper level, `WrapCalibratedExplainer.calibrate(...)` does not declare a dedicated `difficulty_estimator` parameter, but it forwards arbitrary `**kwargs` into `CalibratedExplainer(...)`. That means difficulty enters the wrapper API through `WrapCalibratedExplainer.calibrate(..., difficulty_estimator=...)` as part of the forwarded calibration kwargs.

Relevant code:

- `src/calibrated_explanations/core/wrap_explainer.py`
  - `WrapCalibratedExplainer.calibrate(...)` forwards `**kwargs` to `CalibratedExplainer(...)` in all three branches.
  - `WrapCalibratedExplainer.set_difficulty_estimator(...)` also forwards post-calibration updates to the underlying explainer.

At the explainer level, `CalibratedExplainer.__init__(..., difficulty_estimator=None, **kwargs)` accepts it explicitly.

Relevant code:

- `src/calibrated_explanations/core/calibrated_explainer.py`
  - constructor signature includes `difficulty_estimator=None`

## 2. Where it is stored on `CalibratedExplainer`

`CalibratedExplainer.__init__` calls:

```python
self.set_difficulty_estimator(difficulty_estimator, initialize=False)
```

`CalibratedExplainer.set_difficulty_estimator(...)` validates the object and stores it on:

```python
self.difficulty_estimator = difficulty_estimator
```

That same method also invalidates cached interval plugin metadata and clears the active interval learner so reinitialization sees the updated estimator.

Relevant code:

- `src/calibrated_explanations/core/calibrated_explainer.py`
  - constructor initialization path
  - `set_difficulty_estimator(...)`

## 3. Where it is passed into interval calibration plugins

The interval plugin context is built in `PredictionOrchestrator.build_interval_context(...)`. That method constructs:

```python
difficulty = {"estimator": self.explainer.difficulty_estimator}
```

and also writes the estimator into plugin metadata:

```python
enriched_metadata["difficulty_estimator"] = self.explainer.difficulty_estimator
```

The resulting `IntervalCalibratorContext` exposes a read-only `difficulty` mapping, as required by ADR-013.

The built-in legacy interval plugin then reads it in `LegacyIntervalCalibratorPlugin.create(...)`:

```python
difficulty = context.difficulty.get("estimator")
```

and passes it into `VennAbers(...)` for classification:

```python
calibrator = VennAbers(
    x_cal,
    y_cal,
    learner,
    bins,
    difficulty_estimator=difficulty,
    predict_function=predict_function,
)
```

Relevant code:

- `src/calibrated_explanations/core/prediction/orchestrator.py`
  - `build_interval_context(...)`
- `src/calibrated_explanations/plugins/intervals.py`
  - `IntervalCalibratorContext`
- `src/calibrated_explanations/plugins/builtins.py`
  - `LegacyIntervalCalibratorPlugin.create(...)`

## 4. Where `VennAbers` uses it

`VennAbers.__init__(..., difficulty_estimator=None, ...)` stores the estimator on:

```python
self.de = difficulty_estimator
```

Then `VennAbers.__predict_proba_with_difficulty(...)` applies it before Venn-Abers calibration. The method first gets raw model probabilities, then if `self.de is not None` computes:

```python
difficulty = self.de.apply(x)
```

and uses that difficulty in probability scaling before the calibrated Venn-Abers model consumes the values.

This confirms the current implementation already supports difficulty-aware probability calibration. The effect is upstream of reject and happens at interval-calibrator probability generation time.

Relevant code:

- `src/calibrated_explanations/calibration/venn_abers.py`
  - constructor stores `self.de`
  - `__predict_proba_with_difficulty(...)` applies `self.de.apply(x)` before VA inference

## 5. Where the reject framework obtains probabilities from the interval learner

The reject framework does not compute probabilities directly from the base learner. It reads them from the active interval learner in `RejectOrchestrator`.

Calibration-time reject probabilities are obtained in `initialize_reject_learner(...)`:

- Regression:
  - `self.explainer.interval_learner.predict_probability(x_cal, y_threshold=threshold, bins=bins_cal)`
- Multiclass classification:
  - `self.explainer.interval_learner.predict_proba(x_cal, bins=bins_cal)`
  - then binarizes around the predicted class
- Binary classification:
  - `self.explainer.interval_learner.predict_proba(x_cal, bins=bins_cal)`

Test-time reject probabilities are obtained in `_compute_prediction_set(...)` using the same interval learner:

- Regression:
  - `self.explainer.interval_learner.predict_probability(x, y_threshold=threshold, bins=bins)`
- Multiclass classification:
  - `self.explainer.interval_learner.predict_proba(x, bins=bins)`
  - then binarizes around the predicted class
- Binary classification:
  - `self.explainer.interval_learner.predict_proba(x, bins=bins)`

So reject is downstream of interval calibration and consumes the interval learner as its probability source.

Relevant code:

- `src/calibrated_explanations/core/reject/orchestrator.py`
  - `initialize_reject_learner(...)`
  - `_compute_prediction_set(...)`

## 6. Where calibration and test reject nonconformity scores are computed

The reject nonconformity score helpers live in `src/calibrated_explanations/core/reject/orchestrator.py`.

Calibration scores are computed in:

- `_ncf_scores_cal(proba, classes, labels, ncf, w, default_kind)`

Test scores are computed in:

- `_ncf_scores_test(proba, ncf, w, default_kind)`

The score ingredients are:

- `_default_score_cal(...)`
  - `hinge(...)` for binary/regression default mode
  - `_margin_score(...)` for multiclass default mode
- `_default_score_test(...)`
  - same task-dependent default logic at test time
- `_interval_width_score(proba)`
  - interval width term used by `ensured`
- `_legacy_base_ncf(proba, ncf)`
  - legacy entropy or margin helpers

For the current public modes:

- `ncf="default"`
  - calibration: returns only the default score
  - test: returns only the default score
- `ncf="ensured"`
  - calibration: `(1.0 - w) * interval_width + w * default_score`
  - test: `(1.0 - w) * interval_width + w * default_score`

Relevant code:

- `src/calibrated_explanations/core/reject/orchestrator.py`
  - `_default_score_cal(...)`
  - `_default_score_test(...)`
  - `_interval_width_score(...)`
  - `_ncf_scores_cal(...)`
  - `_ncf_scores_test(...)`

## 7. Does the current implementation normalize reject nonconformity scores by difficulty?

No.

Evidence:

1. The reject NCF helpers accept only `proba`, class/label arrays, `ncf`, `w`, and `default_kind`.
   - No helper takes `difficulty_estimator`, `sigma`, or any per-instance difficulty array.

2. `_ncf_scores_cal(...)` computes either:
   - default hinge or margin score, or
   - `(1.0 - w) * interval_width + w * default_score`
   There is no division by sigma, no multiplication by inverse difficulty, and no alternative normalization step.

3. `_ncf_scores_test(...)` has the same structure at test time.

4. `RejectOrchestrator.initialize_reject_learner(...)` and `_compute_prediction_set(...)` fetch interval-learner probabilities and immediately pass them into `_ncf_scores_cal(...)` and `_ncf_scores_test(...)`.
   - The reject path never calls `self.explainer.difficulty_estimator.apply(...)`.
   - It also does not call `interval_registry.get_sigma_test(...)`.

5. `IntervalRegistry.get_sigma_test(...)` exists, but reject does not use it.

Conclusion:

The current implementation is difficulty-aware only indirectly through the probability outputs already produced by the interval learner, especially Venn-Abers. It is not difficulty-normalized at the reject nonconformity score level.

## 8. What should be tested before adding new code

Before changing runtime behavior, the existing behavior should be pinned with tests and measurement so the eventual experiment can be evaluated against a known baseline.

Recommended test and measurement checklist:

1. Wrapper and public API pass-through
   - Verify `WrapCalibratedExplainer.calibrate(..., difficulty_estimator=...)` still forwards cleanly to `CalibratedExplainer`.
   - Verify `WrapCalibratedExplainer.set_difficulty_estimator(...)` still delegates without changing the public CE-first lifecycle.

2. Explainer storage and invalidation
   - Verify `CalibratedExplainer.set_difficulty_estimator(...)` updates `self.difficulty_estimator` and invalidates cached interval calibrator metadata.

3. Plugin context propagation
   - Verify `PredictionOrchestrator.build_interval_context(...)` exposes the current estimator in both `context.difficulty["estimator"]` and metadata.
   - Verify the built-in legacy interval plugin passes that estimator into `VennAbers`.

4. Existing difficulty effect on interval probabilities
   - Pin that `VennAbers` changes its probability path when a difficulty estimator is supplied.
   - This is the key test proving difficulty already affects interval calibration.

5. Current reject-score formulas
   - Pin that `_ncf_scores_cal(...)` and `_ncf_scores_test(...)` depend only on probability-derived quantities and not on sigma.
   - For `ensured`, pin the exact blend of interval width and default score.

6. Reject probability source
   - Pin that reject uses `interval_learner.predict_proba(...)` or `interval_learner.predict_probability(...)`, not the raw learner.

7. Baseline evaluation runs
   - Re-run the existing reject evaluation suite described in `evaluation/reject/README.md`, especially the current regression threshold baseline and NCF grid scenarios.
   - Preserve the current null result that threshold reject does not select by uncertainty before introducing a difficulty-normalized variant.

8. No-regression API behavior
   - Verify ADR-029 behavior remains unchanged when no reject policy is selected.
   - Verify ADR-020 wrapper and explainer signatures remain stable if an experimental path is later added.

## 9. What should be implemented only after the existing behavior is measured

Only after the current baseline is measured should the repository add an explicitly experimental difficulty-normalized reject path.

That implementation should be scoped as follows:

1. Keep the current public API stable
   - Do not replace the existing public `default` or `ensured` reject NCF modes until the experiment is validated.
   - Do not casually expand the public reject API.

2. Add a strictly opt-in experimental path
   - The experimental logic should live in the reject orchestration layer, because that is where reject NCFs are currently computed.
   - The likely implementation point is `src/calibrated_explanations/core/reject/orchestrator.py`, not `VennAbers`.

3. Normalize reject NCFs using explicit difficulty inputs
   - Use the current estimator already stored on `CalibratedExplainer` or a sigma array derived from it.
   - Apply the normalization consistently to both calibration and test NCF computation.
   - Preserve the current conformal calibration flow and reject learner contracts.

4. Preserve plugin architecture
   - Keep interval plugin behavior unchanged unless the experiment specifically requires plugin-side metadata or calibration artifacts.
   - The gap is in reject scoring, not in interval plugin transport.

5. Add a standalone evaluation scenario under `evaluation/reject/`
   - The existing evaluation README already states that difficulty-normalized regression reject is deferred pending the RT-2 sigma-normalisation-only fix.
   - The experimental implementation should therefore land together with a dedicated evaluation scenario that measures whether the new scoring actually improves selective behavior.

## Exact Remaining Gap

The remaining gap is not API plumbing and not interval-plugin transport.

The exact missing piece is:

> A reject-orchestrator-level nonconformity computation that incorporates per-instance difficulty, for both calibration and test scores, while keeping the existing CE-first API, reject policy contracts, and interval plugin architecture unchanged by default.

In practical terms, the repository already has:

- wrapper-to-explainer difficulty pass-through,
- explainer storage and invalidation,
- plugin-context propagation,
- Venn-Abers difficulty-aware probability scaling,
- reject scoring over interval-learner probabilities.

It does not yet have:

- sigma-aware reject score normalization in `_ncf_scores_cal(...)`,
- sigma-aware reject score normalization in `_ncf_scores_test(...)`,
- an experimental opt-in control surface for that behavior,
- a dedicated evaluation scenario proving the experimental path is worth promoting.

## File Map

- `src/calibrated_explanations/core/wrap_explainer.py`
  - wrapper calibration entrypoint and post-calibration setter delegation
- `src/calibrated_explanations/core/calibrated_explainer.py`
  - accepts and stores `difficulty_estimator`
- `src/calibrated_explanations/core/prediction/orchestrator.py`
  - builds the interval plugin context carrying difficulty
- `src/calibrated_explanations/core/prediction/interval_registry.py`
  - exposes sigma helper, currently unused by reject
- `src/calibrated_explanations/plugins/builtins.py`
  - built-in legacy interval plugin passes difficulty into `VennAbers`
- `src/calibrated_explanations/calibration/venn_abers.py`
  - applies difficulty before Venn-Abers calibration
- `src/calibrated_explanations/core/reject/orchestrator.py`
  - computes reject NCFs and prediction sets; this is where the remaining gap lives
- `evaluation/reject/README.md`
  - already documents that difficulty-normalized reject remains deferred pending the RT-2 fix
