# Capabilities Manifest

This page catalogs all core capabilities supported by Calibrated Explanations, organized by audience and task type. Use it to discover features and navigate to detailed documentation.

```{admonition} Guarantees & Assumptions
:class: important

All capabilities share these foundational requirements:

- **Calibration set required**: A held-out calibration set (typically 20-25% of training data) is mandatory
- **Interval invariant**: All intervals satisfy `low <= predict <= high`
- **Uncertainty decomposition**: Intervals capture both aleatoric (data) and epistemic (model) uncertainty
- **Calibration validity**: Guarantees hold when calibration and test distributions match

See the ADR-021 formal semantics on GitHub:
<https://github.com/Moffran/calibrated_explanations/blob/main/docs/improvement/adrs/ADR-021-calibrated-interval-semantics.md>.
```

## Capability Matrix

| # | Capability | Task Type | Audience | Entry Point |
|---|------------|-----------|----------|-------------|
| 1 | [Calibrated classification (binary)](#calibrated-classification) | Classification | Practitioner | {doc}`../get-started/quickstart_classification` |
| 2 | [Calibrated classification (multiclass)](#calibrated-classification) | Classification | Practitioner | {doc}`../get-started/quickstart_classification` |
| 3 | [Conformal interval regression](#conformal-interval-regression) | Regression | Practitioner | {doc}`../get-started/quickstart_regression` |
| 4 | [Probabilistic regression](#probabilistic-regression) | Regression | Practitioner | {doc}`../foundations/concepts/probabilistic_regression` |
| 5 | [Factual explanations](#factual-explanations) | All | Practitioner | {doc}`../foundations/how-to/interpret_explanations` |
| 6 | [Alternative explanations](#alternative-explanations) | All | Practitioner | {doc}`../foundations/concepts/alternatives` |
| 7 | [Ensured explanations](#ensured-explanations) | All | Practitioner | {doc}`../practitioner/playbooks/ensured-explanations` |
| 8 | [Mondrian/conditional calibration](#mondrian-conditional-calibration) | All | Practitioner | {doc}`../practitioner/playbooks/mondrian-calibration` |
| 9 | [Fast explanations](#fast-explanations) | All | Advanced | {doc}`../practitioner/advanced/use_plugins` |
| 10 | [Normalization/difficulty estimation](#normalization-difficulty-estimation) | Regression | Advanced | {doc}`../practitioner/advanced/normalization-guide` |
| 11 | [Reject/defer decision support](#reject-defer-decision-support) | All | Practitioner | {doc}`../practitioner/playbooks/decision-policies` |
| 12 | [Prediction intervals with uncertainty](#prediction-intervals-with-uncertainty) | All | Practitioner | {doc}`../foundations/how-to/interpret_explanations` |
| 13 | [Triangular plot visualization](#triangular-plot-visualization) | All | Practitioner | {doc}`../foundations/concepts/alternatives` |

---

## Capability Details

### Calibrated Classification

**What it is**: Venn-Abers calibration wraps any scikit-learn classifier to produce calibrated probability estimates with uncertainty intervals. Works for both binary and multiclass problems.

**API**: `predict_proba(x, uq_interval=True)` returns `(probs, (low, high))`

**What it is NOT**: Raw model probabilities. Calibrated probabilities reflect the observed frequency of outcomes in the calibration set, not just model confidence scores.

**Learn more**: {doc}`classification`, {doc}`../get-started/quickstart_classification`

---

### Conformal Interval Regression

**What it is**: Conformal Predictive Systems (CPS) produce prediction intervals that contain the true value with a specified coverage probability (e.g., 90% intervals).

**API**: `predict(x, uq_interval=True, low_high_percentiles=(5, 95))`

**What it is NOT**: Bayesian credible intervals. CPS intervals are frequentist constructs with finite-sample coverage guarantees under exchangeability.

**Learn more**: {doc}`regression`, {doc}`../get-started/quickstart_regression`

---

### Probabilistic Regression

**What it is**: Query the calibrated probability that a numeric target exceeds (or falls within) a threshold. Combines CPS with Venn-Abers for calibrated threshold probabilities.

**API**: `predict_proba(x, threshold=t, uq_interval=True)` for P(y <= t)

**What it is NOT**: Interval regression. Probabilistic regression answers "what's the probability?" while interval regression answers "where will y fall?"

**Learn more**: {doc}`probabilistic_regression`, {doc}`../foundations/concepts/probabilistic_regression`

---

### Factual Explanations

**What it is**: Rule-based explanations showing why the model made its prediction. Each feature contribution includes uncertainty bounds reflecting calibration quality.

**API**: `explain_factual(x)` returns `CalibratedExplanation` with rules and weights

**What it is NOT**: Post-hoc approximations like LIME/SHAP. Factual explanations use the actual calibrated model behavior, not surrogate models.

**Learn more**: {doc}`../foundations/how-to/interpret_explanations`

---

### Alternative Explanations

**What it is**: Counterfactual, semi-factual, and super-factual rules showing what feature changes would flip or reinforce the prediction.

**API**: `explore_alternatives(x)` returns alternative rule sets with uncertainty

**What it is NOT**: Optimal counterfactuals. Alternatives explore feature perturbations but don't guarantee minimal-change counterfactuals.

**Learn more**: {doc}`../foundations/concepts/alternatives`

---

### Ensured Explanations

**What it is**: A strategy to reduce epistemic uncertainty in alternative explanations by filtering and ranking alternatives based on uncertainty interval width.

**API**: Five filters on alternatives, supported for classification, probabilistic regression, and plain regression:

- ``super_explanations()``
- ``semi_explanations()``
- ``counter_explanations()``
- ``ensured_explanations()``
- ``pareto_explanations()``

Plotting supports the ensured/triangular view via ``style="ensured"`` (alias for ``style="triangular"``) and ranking controls such as ``rnk_metric="ensured"`` and ``rnk_weight``.

**What it is NOT**: Guaranteed low-uncertainty. Ensured explanations help identify more reliable alternatives but cannot eliminate uncertainty.

**Research**: [Ensured: Explanations for Decreasing Epistemic Uncertainty (arXiv:2410.05479)](https://arxiv.org/abs/2410.05479)

**Learn more**: {doc}`../practitioner/playbooks/ensured-explanations`

---

<a id="mondrian-conditional-calibration"></a>

### Mondrian/Conditional Calibration

**What it is**: Calibrate separately for different subgroups (e.g., by protected attribute or domain segment) to produce group-specific uncertainty estimates.

**API**: Pass `bins=group_labels` to `calibrate()` or explanation methods

**What it is NOT**: Fairness-by-default. Mondrian calibration reveals group-specific uncertainty but doesn't automatically enforce fairness constraints.

**Research**: [Conditional Calibrated Explanations (xAI 2024)](https://link.springer.com/chapter/10.1007/978-3-031-63787-2_17)

**Learn more**: {doc}`../practitioner/playbooks/mondrian-calibration`

---

### Fast Explanations

**What it is**: An experimental plugin that accelerates rule generation through sampling, trading some detail for speed.

**API**: `explain_factual(x, fast=True)` or explicit plugin registration

**What it is NOT**: Production-ready by default. Fast explanations are opt-in experimental features that may sacrifice some calibration precision.

**Status**: Experimental (opt-in only)

**Learn more**: {doc}`../practitioner/advanced/use_plugins`

---

<a id="normalization-difficulty-estimation"></a>

### Normalization/Difficulty Estimation

**What it is**: Adjust prediction intervals based on instance-specific difficulty, producing wider intervals for harder-to-predict instances.

**API**: Internal configuration via `IntervalRegressor` parameters

**What it is NOT**: Automatic. Difficulty estimation requires explicit configuration and may need additional calibration data.

**Learn more**: {doc}`../practitioner/advanced/normalization-guide`

---

<a id="reject-defer-decision-support"></a>

### Reject/Defer Decision Support

**What it is**: Systematic handling of uncertain predictions through reject policies that flag, filter, or route uncertain instances.

**API**: `explain_factual(x, reject_policy=RejectPolicy.FLAG)`

**What it is NOT**: Automatic rejection. Policies provide the infrastructure; you define thresholds and routing logic.

**Learn more**: {doc}`../practitioner/playbooks/decision-policies`, {doc}`../practitioner/advanced/reject-policy`

---

### Prediction Intervals with Uncertainty

**What it is**: Every prediction method can return uncertainty intervals via `uq_interval=True`, quantifying both aleatoric and epistemic uncertainty.

**API**: `predict(x, uq_interval=True)` or `predict_proba(x, uq_interval=True)`

**What it is NOT**: Confidence intervals on model parameters. These are prediction intervals on future observations.

**Learn more**: {doc}`../foundations/how-to/interpret_explanations`

---

### Triangular Plot Visualization

**What it is**: A specialized visualization showing how alternative explanations trade off against the base prediction, with uncertainty overlays.

**API**: Available via PlotSpec or direct plotting from explanation objects

**What it is NOT**: A general-purpose plot. Triangular plots are designed specifically for comparing factual vs. alternative explanations.

**Learn more**: {doc}`../foundations/concepts/alternatives`, {doc}`../foundations/how-to/plot_with_plotspec`

---

## Audience Guide

### Practitioners (Day-to-Day Users)

Start with these capabilities:
1. Calibrated classification or regression (your task type)
2. Factual explanations (understand predictions)
3. Alternative explanations (explore what-ifs)
4. Prediction intervals (quantify uncertainty)

### Researchers (Replication and Extension)

Explore these advanced capabilities:
1. Ensured explanations (reduce epistemic uncertainty)
2. Mondrian calibration (conditional/fairness analysis)
3. Probabilistic regression (threshold queries)

### Advanced Users (Performance and Customization)

Consider these when needed:
1. Fast explanations (speed optimization)
2. Normalization/difficulty estimation (heteroscedastic data)
3. Reject/defer policies (production decision logic)

```{toctree}
:hidden:
:maxdepth: 1
```
