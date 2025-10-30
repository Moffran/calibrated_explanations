---
sidebarclass: how-to
---

# Interpret calibrated explanations

Understanding the explanation objects is the core reason to use this library. This guide shows how to read factual and alternative explanations, relate them to the calibrated predictions and uncertainty intervals they produce, and tie everything back to the plots and telemetry that document provenance.

```{admonition} Terminology
:class: tip

- **Factual explanation** – describes why the model produced the calibrated prediction for the observed instance.
- **Alternative explanation** – proposes changes to feature values that would produce a different calibrated outcome.
- **Rule** – a feature-level contribution (weight + condition) inside each explanation.
```

## 1. Start with the explanation collections

Calibrated explanations return collections (`CalibratedExplanations`, `AlternativeExplanations`) that behave like sequences. The first step is to look at the summary for each item:

```python
factual_batch = explainer.explain_factual(X_test[:1])
factual = factual_batch[0]
alternative_batch = explainer.explore_alternatives(X_test[:1])
alternative = alternative_batch[0]

print(f"Mode: {factual.get_mode()}  Calibrated prediction: {factual.predict:.3f}")
print(f"Mode: {alternative.get_mode()}  Calibrated prediction: {alternative.predict:.3f}")
```

Key attributes shared by both modes:

| Attribute | Description |
| --------- | ----------- |
| `predict` | Calibrated probability (classification) or numeric estimate (regression). |
| `prediction_interval` | Tuple of `(low, high)` values; mirrors `uncertainty["lower_bound"]`/`["upper_bound"]`. |
| `rules` | Ordered list of feature-level rules, highest absolute weight first. |
| `telemetry` | Dictionary describing the interval, probability, and plot sources that produced the explanation. |

For regression with `threshold` values, the `predict` value expresses the probability that the outcome lies within the threshold bounds; the interval reflects percentile uncertainty around the calibrated estimate.

## 2. Examine the rule tables in detail

Rules translate the calibration result into actionable statements. Use the helper payloads to inspect them in a structured way.

### 2.1 Factual explanations

```python
factual_rules = factual.build_rules_payload()
for rule in factual_rules:
    print(
        f"{rule['feature']:>20} "
        f"| condition={rule['condition']['operator']} {rule['condition']['value']} "
        f"| weight={rule['weight']:+.3f} "
        f"| weight interval=({rule['uncertainty']['lower_bound']:.3f}, "
        f"{rule['uncertainty']['upper_bound']:.3f})"
    )
```

Read the table from top to bottom:

1. **Weight sign** – positive weights push the calibrated prediction towards the positive outcome; negative weights pull it away. Large magnitude implies greater influence.
2. **Condition** – confirms the feature value observed in the instance (for example, `glucose >= 140`). Use this to cross-check domain constraints.
3. **Prediction field** – shows the calibrated prediction contribution when the feature is perturbed; compare it to `predict` to understand marginal impact.
4. **Weight interval** – showcases the uncertainty around the contribution estimate; wide ranges imply the feature weight is poorly supported by calibration data.

`baseline_prediction` (when present) captures the calibrated output before feature-level adjustments and is useful when comparing against alternative scenarios.

### 2.2 Alternative explanations

Alternative explanations add scenario guidance on top of feature contributions. The payload uses the same helper:

```python
alternative_rules = alternative.build_rules_payload()
for item in alternative_rules:
    if item['kind'] == 'alternative':
        print('Suggested changes:', item['conditions'])
        print('Resulting calibrated prediction:', item['calibrated_prediction'])
        for feature_rule in item['feature_rules']:
            print(
                f"  {feature_rule['feature']}: weight={feature_rule['weight']:+.3f} "
                f"condition={feature_rule['condition']['operator']} {feature_rule['condition']['value']}"
            )
```

- `conditions` describe the feature adjustments recommended to reach the alternative outcome.
- `calibrated_prediction` shows the probability or value after applying those adjustments; compare it with the factual `calibrated_prediction` to judge impact.
- `feature_rules` mirrors the factual structure so you can see how each feature behaves under the alternative scenario.
- `feature_rules[n]["uncertainty"]` reports the weight interval under the alternative scenario; wide ranges suggest collecting more calibration data before acting.

Use these payloads to answer “which change makes the most difference?”—the best candidate is the scenario with the highest calibrated probability swing and acceptable feature adjustments. If uncertainty bounds remain wide, treat the recommendation with caution and look for additional calibration data.
## 3. Connect explanations to plots

PlotSpec plots visualise the same rule rankings and uncertainty intervals:

```python
fig = factual.plot(uncertainty=True, filter_top=6)
```

Interpret the chart as follows:

1. **Bar length** corresponds to the absolute weight shown in the rule table.
2. **Colour** indicates direction (green for positive impact, purple for negative by default).
3. **Shaded band** around each bar is the rule-level uncertainty interval.
4. **Vertical line** over the summary panel mirrors the calibrated prediction and its `(low, high)` bounds.

When reviewing alternative explanations, use `plot(style="alternative")` (or open the alternative tab in interactive renderers) to see how the suggested feature changes alter the bars and the calibrated outcome.

If the plot does not match the table, regenerate the explanation—the library guarantees they use the same data.

## 4. Tie everything back to telemetry

Telemetry proves which components produced the explanation:

```python
telemetry = getattr(factual_batch, "telemetry", {})
print(telemetry["interval_source"])
print(telemetry["plot_source"])
print(telemetry["uncertainty"])
```

Interpretation tips:

- `interval_source` / `proba_source` show the calibration algorithms in play. Namespaced identifiers such as `registry:interval.default` indicate trusted plugins; `legacy:` prefixes mean a fallback was used.
- `plot_source` surfaces the renderer (for example `plot_spec.default.factual`). Investigate if a legacy fallback appears unexpectedly.
- The nested `uncertainty` dictionary mirrors the structure already discussed and is the authoritative source for machine-readable exports.
- `preprocessor` provides ADR-009 provenance so you can reproduce how categorical encoders or scalers shaped the data.

## 5. Validate in notebooks and downstream systems

After you absorb the tables and plots:

1. Run the demo notebooks (`demo_under_the_hood.ipynb`, classification/regression demos) and compare their telemetry dumps to your production environment.
2. Embed the interpretation checks into dashboards—surface calibrated prediction vs. interval charts and highlight scenarios with wide rule uncertainty.
3. Share this guide with auditors and stakeholders so everyone relies on the same vocabulary when discussing factual or alternative explanations.

Keeping the interpretation workflow front and centre ensures calibrated explanations drive safe, explainable decisions rather than becoming another opaque model output.
