# Difficulty-Normalized Reject for Conformal Classification

This page is the research-facing summary for CE's experimental
difficulty-normalized reject-option conformal classification work. It documents
what has been implemented and evaluated without promoting a new public
nonconformity function (NCF) beyond the stable `default` and `ensured` choices.

The companion practitioner guide is {doc}`../../practitioner/advanced/reject-policy`.

## What CE reject-option classification does

For classification calls with reject enabled, CE builds conformal prediction
sets from calibrated class probabilities. The prediction-set geometry determines
whether the instance can be routed automatically or should be rejected:

| Prediction set | CE route | Interpretation |
| --- | --- | --- |
| Singleton set (`|S(x)| = 1`) | accept | One class remains plausible at the selected confidence. |
| Empty set (`|S(x)| = 0`) | novelty reject | No class is supported strongly enough; treat as non-covered or novel. |
| Multi-label set (`|S(x)| >= 2`) | ambiguity reject | More than one class remains plausible; defer or review. |

These routes are exposed through reject metadata including `reject_rate`,
`ambiguity_rate`, `novelty_rate`, `prediction_set_size`, `ambiguity_mask`, and
`novelty_mask`. The aggregate relation is
`reject_rate = ambiguity_rate + novelty_rate`.

## Existing difficulty-aware Venn-Abers path

CE already has a difficulty-aware classification path before direct reject-score
normalization is selected:

1. `difficulty_estimator` is accepted by `CalibratedExplainer`.
2. The interval plugin context carries the estimator to the built-in legacy
   interval plugin.
3. The built-in plugin passes the estimator into `VennAbers`.
4. `VennAbers` applies difficulty through probability scaling before
   Venn-Abers calibration.
5. The reject framework then computes `default` or `ensured` reject scores from
   the calibrated probabilities.

This path is difficulty-aware indirectly. It changes the probabilities consumed
by reject, but the reject nonconformity formulas themselves remain the public
`default` or `ensured` formulas.

## New direct reject-score normalization

The experimental direct strategy is selected through the reject strategy
registry:

```python
strategy="experimental.difficulty_normalized"
```

It changes the reject score definition by normalizing calibration and test
nonconformity scores with per-instance difficulty before conformal p-values and
prediction sets are computed. This keeps the CE-first API and `RejectPolicy`
contracts unchanged while making the experimental behavior explicit and opt-in.

The novelty-aware research variant is:

```python
strategy="experimental.ambiguity_normalized_novelty_penalized"
```

That variant remains diagnostic. It combines ambiguity difficulty normalization
with an additional novelty penalty to explore whether empty-set novelty rejects
can be separated from multi-label ambiguity rejects more cleanly.

## Why normalization must happen before p-values

Difficulty normalization is part of the nonconformity score definition. The
conformal p-values and prediction sets are calibrated from the score
distribution, so the normalization must be applied to both calibration and test
scores before p-value computation.

Applying difficulty only as a post-hoc reject threshold is a different
operation. It moves a final decision cutoff but leaves the conformal scores,
p-values, and prediction sets unchanged. That may be a useful heuristic, but it
is not difficulty-normalized conformal scoring.

## When to use each mode

Use `ncf="default"` when you need the stable public baseline and want to preserve
current CE reject behavior.

Use `ncf="ensured"` when interval-width-aware scoring is desired and the blend
weight `w` is part of the experiment or operating-point tuning. This is still a
stable public NCF mode.

Use `strategy="experimental.difficulty_normalized"` for research and ablation
runs where difficulty should directly shape reject scoring. Keep VA difficulty
off for the cleanest primary contrast unless you are explicitly studying
double-counting.

Use `strategy="experimental.ambiguity_normalized_novelty_penalized"` only for
diagnostics of ambiguity-vs-novelty separation. Current Scenario 10 evidence did
not justify promoting it over the simpler difficulty-normalized strategy.

## Validity and methodology caveats

- Fit and freeze the difficulty estimator before reject calibration alphas are
  estimated.
- Avoid fitting the estimator on calibration labels or calibration residuals
  unless explicit cross-fitting is used.
- Treat empirical selectivity, accepted accuracy, and difficulty alignment as
  empirical usefulness metrics, not new formal coverage guarantees.
- Coverage claims remain tied to the conformal score pipeline and its
  exchangeability assumptions. Difficulty normalization changes that pipeline.
- Experimental strategies are not public NCF contract expansion; `default` and
  `ensured` remain the public-facing NCF modes.

## Evaluation summary: Scenarios 8-11

The evaluation artifacts live under `evaluation/reject/artifacts/`.

### Scenario 8: existing indirect difficulty effect

Scenario 8 measured the existing path
`difficulty_estimator -> VennAbers probability scaling -> reject NCF`.

For `default`, enabling VA difficulty changed accept rate by -22.2 percentage
points, rejected-error capture by +11.5 percentage points, and accepted accuracy
by -10.9 percentage points. For `ensured`, accept rate changed by -9.7
percentage points, rejected-error capture by +4.9 percentage points, and
accepted accuracy by -12.2 percentage points.

Interpretation: the indirect VA difficulty path acted mainly as a stricter reject
gate in this run. It captured more errors but accepted far fewer instances and
reduced accepted accuracy.

### Scenario 9: direct difficulty-normalized reject scores

Scenario 9 compared six arms. The primary research contrast was A vs C:
`builtin.default`, `ncf=default`, no VA difficulty, against
`experimental.difficulty_normalized`, `ncf=default`, no VA difficulty.

Direct normalization changed reject rate by +1.08 percentage points, the
rejected-minus-accepted difficulty gap by +0.3416, and
`difficulty_reject_auc` by +0.2012. Matched reject-rate bins showed a mean
accepted-accuracy delta of -0.0089 for C minus A, so the evidence favors
difficulty-aligned routing rather than a blanket accepted-accuracy improvement.

Arms with both VA difficulty and direct score normalization were diagnostic for
double-counting. The artifact recommends arm C as the next experimental baseline
because it gives the cleanest direct-normalization contrast without VA
double-count risk.

### Scenario 10: ambiguity-normalized novelty-penalized variant

Scenario 10 compared the built-in baseline, arm C, and the novelty-aware arm G.
Relative to C, G changed novelty rate by +0.0019, ambiguity rate by +0.0047, and
accepted accuracy by +0.0037. It also reduced novelty reject AUC by -0.0371.

Interpretation: the novelty-aware variant did not clearly improve
ambiguity-vs-novelty separation. Arm C remains the simpler recommended
experimental baseline.

### Scenario 11: matched operating-point selection

Scenario 11 selected confidence values closest to target reject rates
`0.10`, `0.20`, `0.30`, and `0.40` instead of averaging across the confidence
grid. This is the decision-gate scenario for public API promotion.

For A vs C, accepted-accuracy deltas by target reject rate were +0.0012,
-0.0029, -0.0070, and -0.0089. The best matched operating point was target
`0.10`, but the overall evidence was mixed. Across targets, C minus A mean
`difficulty_reject_auc` was -0.0040.

For C vs G, the novelty-aware variant increased novelty and empty-set rates by
+0.0084 on average, changed accepted accuracy by -0.0005, and increased
novelty reject AUC by +0.0845 while reducing ambiguity rate by -0.0114.

Interpretation: Scenario 11 does not justify public API promotion. Direct
difficulty normalization and the novelty-aware strategy should both remain
experimental.

## Contribution framing

Development contribution:
: CE now has difficulty-aware reject routing that preserves the CE-first
  lifecycle, `RejectPolicy` output contracts, and the existing plugin
  architecture.

Research contribution:
: CE now has an experimental difficulty-normalized nonconformity strategy for
  reject-option conformal classification, evaluated against the built-in
  difficulty-aware VA path and a novelty-aware diagnostic variant.

## Open questions

- Ambiguity vs novelty separation: can empty-set novelty and multi-label
  ambiguity be separated without harming accepted decision quality?
- Double-counting difficulty: when, if ever, should VA difficulty scaling and
  direct reject-score normalization be combined?
- Conditional validity and Mondrian variants: can subgroup-aware calibration
  improve reliability without sacrificing useful reject selectivity?
- Finite-sample behavior: how stable are the observed effects across small
  calibration sets, high confidence regimes, and heterogeneous datasets?

## Minimal CE-first experiment snippet

```python
from calibrated_explanations import RejectPolicySpec, WrapCalibratedExplainer

wrapper = WrapCalibratedExplainer(model)
wrapper.fit(x_train, y_train)
wrapper.calibrate(
    x_cal,
    y_cal,
    feature_names=feature_names,
    difficulty_estimator=difficulty_estimator,
)

result = wrapper.predict(
    x_test,
    reject_policy=RejectPolicySpec.flag(ncf="default"),
    confidence=0.95,
    strategy="experimental.difficulty_normalized",
)

print(result.metadata["reject_rate"])
print(result.metadata["ambiguity_rate"], result.metadata["novelty_rate"])
```

Entry-point tier: Tier 3
