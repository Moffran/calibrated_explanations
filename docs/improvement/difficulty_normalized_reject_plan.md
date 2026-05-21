# Difficulty-Normalized Reject: Current State and Final Documentation

## Purpose

This note documents the repository behavior for difficulty-aware interval
calibration and reject-option conformal classification, the implemented
experimental difficulty-normalized reject strategy, and the documentation state
after the Scenario 8-11 evaluation pass.

This document is intentionally implementation-accurate and does not propose runtime changes in this step. It is grounded by:

- ADR-013, which requires interval calibration plugins to preserve the existing Venn-Abers and IntervalRegressor semantics while receiving read-only context.
- ADR-020, which requires preserving the CE-first public API and existing wrapper/explainer lifecycle surface.
- ADR-029, which requires reject to remain opt-in, policy-driven, and backward compatible when reject is not selected.

## Short Answer

`difficulty_estimator` already affects calibrated probabilities because it is accepted by `CalibratedExplainer`, propagated through the interval plugin context, and used inside `VennAbers` before Venn-Abers calibration.

The repository now also includes an explicitly experimental direct reject-score
normalization path selected through:

```python
strategy="experimental.difficulty_normalized"
```

This strategy normalizes reject nonconformity scores before conformal p-values
and prediction sets are computed. The public NCF modes remain `default` and
`ensured`; the experimental strategy is not a public NCF promotion.

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

## 7. Does the implementation normalize reject nonconformity scores by difficulty?

Yes, but only on the explicit experimental path.

Baseline evidence:

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

Baseline conclusion:

The current implementation is difficulty-aware only indirectly through the probability outputs already produced by the interval learner, especially Venn-Abers. It is not difficulty-normalized at the reject nonconformity score level.

Experimental implementation:

- `RejectOrchestrator` registers `experimental.difficulty_normalized`.
- The strategy applies difficulty to calibration and test reject scores before
  conformal p-values and prediction sets are computed.
- The strategy rejects regression mode and remains classification-only.
- Provenance checks warn or raise when estimator metadata indicates calibration
  label/residual leakage without cross-fitting.
- A second diagnostic strategy,
  `experimental.ambiguity_normalized_novelty_penalized`, adds an evaluation-only
  novelty penalty for ambiguity-vs-novelty experiments.

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

## 9. Implemented experimental path and remaining promotion gates

The experimental path has been implemented after measuring the existing
baseline. Promotion to a public NCF mode remains intentionally deferred.

The implemented scope is:

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

5. Add standalone evaluation scenarios under `evaluation/reject/`
   - Scenario 8 measures the existing indirect VA difficulty effect.
   - Scenario 9 evaluates direct difficulty-normalized reject scoring.
   - Scenario 10 evaluates the ambiguity-normalized novelty-penalized variant.
   - Scenario 11 evaluates matched operating-point selection before any public
     promotion decision.

## Exact Remaining Promotion Gap

The remaining gap is not API plumbing, interval-plugin transport, or a missing
experimental implementation.

The exact remaining promotion question is:

> Whether difficulty-normalized reject scoring should be promoted from an
> experimental strategy identifier to a stable public NCF contract.

The repository already has:

- wrapper-to-explainer difficulty pass-through,
- explainer storage and invalidation,
- plugin-context propagation,
- Venn-Abers difficulty-aware probability scaling,
- reject scoring over interval-learner probabilities,
- experimental direct difficulty-normalized reject scoring,
- provenance warnings/strict validation for risky estimator fitting metadata,
- evaluation artifacts for Scenarios 8-11.

It does not yet have:

- promotion evidence strong enough to add `difficulty_normalized` as a public NCF,
- a resolved ambiguity-vs-novelty separation strategy,
- a resolved policy for combining VA difficulty scaling with direct reject
  normalization,
- conditional/Mondrian validity guidance for the strategy,
- finite-sample characterization across small calibration regimes.

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

## Final documentation summary (2026-05-21)

This repository now documents difficulty-normalized reject-option conformal
classification as an explicitly experimental strategy layered on top of the
existing CE reject contracts.

Final user-facing documentation lives in:

- `docs/practitioner/advanced/reject-policy.md`
- `docs/researcher/advanced/difficulty_normalized_reject.md`
- `evaluation/reject/README.md`

### What reject-option conformal classification does in CE

For classification, CE builds conformal prediction sets and maps set geometry to
reject outcomes:

- Singleton prediction set: accepted.
- Empty prediction set: novelty reject.
- Multi-label prediction set: ambiguity reject.

This mapping is reflected in reject metadata (`reject_rate`, `ambiguity_rate`,
`novelty_rate`, `prediction_set_size`, masks).

### Existing and new difficulty behavior

- Existing behavior (already implemented):
  `difficulty_estimator -> VennAbers probability scaling -> reject scoring`
- New experimental behavior:
  `difficulty_estimator -> direct reject-score normalization -> conformal p-values`

### Why normalization belongs before conformal p-values

Difficulty normalization is part of nonconformity definition. Applying it only as
a post-hoc threshold changes the decision cutoff but not the conformal score
distribution. Therefore, it does not represent difficulty-aware conformal scoring.

## Scenario 8-11 evidence snapshot

Source artifacts:

- `evaluation/reject/artifacts/scenario_8_difficulty_reject_ablation.md`
- `evaluation/reject/artifacts/scenario_9_difficulty_normalized_ncf.md`
- `evaluation/reject/artifacts/scenario_10_ambiguity_novelty_reject.md`
- `evaluation/reject/artifacts/scenario_11_operating_point_selection.md`

### Scenario 8 (indirect VA difficulty effect)

- Difficulty through VA alone tightened rejection strongly (lower accept rate,
  higher rejected-error capture).
- Accepted accuracy dropped materially in this setup.
- Interpretation: current indirect path acts mainly as a stricter reject gate.

### Scenario 9 (difficulty-normalized reject NCF)

- Primary A-vs-C contrast supports direct difficulty normalization as a stronger
  difficulty-aligned reject selector.
- Matched reject-rate analysis reported accepted-accuracy gains for arm C.
- Diagnostic arms with both VA difficulty and direct normalization indicate
  potential double-counting risk.

### Scenario 10 (ambiguity-normalized novelty-penalized variant)

- Novelty lift versus C was small.
- Ambiguity did not decrease in aggregate in this run.
- C remained the recommended simpler experimental baseline.

### Scenario 11 (matched operating-point selection)

- Confidence values were selected closest to target reject rates 0.10, 0.20,
  0.30, and 0.40 instead of averaging across the confidence sweep.
- A-vs-C accepted-accuracy deltas by target were +0.0012, -0.0029, -0.0070,
  and -0.0089.
- A-vs-C mean difficulty-reject-AUC delta across targets was -0.0040.
- G-vs-C increased novelty/empty-set rates by +0.0084 on average and novelty
  reject AUC by +0.0845, but accepted accuracy changed by -0.0005.
- Recommendation: do not promote difficulty-normalized reject scoring to a
  public NCF yet; keep the novelty-aware strategy internal/experimental.

## Usage guidance

- Use `ncf="default"` for stable public behavior.
- Use `ncf="ensured"` when interval-width blending is desired.
- Use `strategy="experimental.difficulty_normalized"` for opt-in research and
  ablation runs.
- Use `strategy="experimental.ambiguity_normalized_novelty_penalized"` only for
  research diagnostics until promotion evidence is stronger.

## Validity caveats

- Fit/freeze the difficulty estimator before reject calibration.
- Do not fit on calibration labels/residuals unless cross-fitted.
- Distinguish empirical utility from formal coverage guarantees.
- Keep experimental strategy semantics separate from public NCF contracts
  (`default`, `ensured`) until promotion criteria are met.

## Contribution framing

- Development contribution:
  difficulty-aware reject routing integrated in CE while preserving CE-first API,
  reject policy contracts, and plugin architecture.
- Research contribution:
  experimental difficulty-normalized nonconformity for reject-option conformal
  classification, evaluated against baseline and novelty-aware variants.

## Open research questions

- How to separate ambiguity and novelty more effectively without harming accepted
  decision quality.
- How to avoid difficulty double-counting when VA difficulty scaling and direct
  reject normalization are both active.
- How to extend to conditional/Mondrian validity while preserving useful
  reject selectivity.
- How finite-sample behavior changes across calibration-set sizes and
  confidence regimes.
