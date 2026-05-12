# CE-First Calibrated Explanations Guide (OSS)

## 1. Executive summary (short)

Calibrated Explanations (CE) provide **actionable explanations with calibrated uncertainty** for classification and regression. A **CE-First** workflow means you **always use the OSS `calibrated_explanations` library**, wrap your model with **`WrapCalibratedExplainer`**, **fit and calibrate** before explaining, and default to **calibrated outputs** for both humans and agents. This reduces hallucinated explanations, makes uncertainty explicit, and ensures standardized, auditable behavior.

---

## 2. CE-First policy (concise & actionable)

### CE-First checklist (agents must follow)

1. **Library presence**: If `calibrated_explanations` is missing, **fail fast** and instruct:
   ```bash
   pip install calibrated-explanations
   ```
2. **Wrapper**: Use **`WrapCalibratedExplainer`** (or a subclass). Never invent a new wrapper.
3. **Fit**: `explainer.fit(...)` → verify `explainer.fitted is True`.
4. **Calibrate**: `explainer.calibrate(...)` → verify `explainer.calibrated is True`.
5. **Explain**: Use **`explainer.explain_factual(...)`** or **`explorer.explore_alternatives(...)`**.
6. **Conjunctions**: Use **`explanations.add_conjunctions(...)`** or **`explanations[idx].add_conjunctions(...)`**.
7. **Narratives & plots**: Use `.to_narrative(format=...)` and `.plot(...)`.
8. **Probabilistic regression**: Use `threshold=` for probabilistic intervals, or `low_high_percentiles=` for conformal.

### Enforcement rules agents must apply programmatically

- **CE presence**: `import calibrated_explanations` or fail with install message.
- **Wrapper identity**: Ensure instance is `WrapCalibratedExplainer` or subclass.
- **State checks**: `.fitted` and `.calibrated` must be `True` before predictions/explanations.
- **Calibrated by default**: Do not return uncalibrated outputs unless explicitly requested.
- **Canonical calls**:
  - Factual: `explainer.explain_factual(X, ...)`
  - Alternatives: `explainer.explore_alternatives(X, ...)`

---

## 3. Canonical usage patterns (runnable examples)

> **Assumption (defensive)**: These examples assume the OSS quickstart API: `WrapCalibratedExplainer`, `fit`, `calibrate`, `explain_factual`, `explore_alternatives`, `predict`, `predict_proba`, `.fitted`, `.calibrated`, `to_narrative`, `plot`, `uq_interval`, `low_high_percentiles`. If signatures differ, the helper below probes the API and adapts or raises helpful errors.

### 3.1 Classification quickstart

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations import WrapCalibratedExplainer

X, y = load_breast_cancer(return_X_y=True)
x_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
x_cal, x_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

model = RandomForestClassifier(random_state=0)
explainer = WrapCalibratedExplainer(model)

# Fit + calibrate
explainer.fit(x_train, y_train)
assert explainer.fitted
explainer.calibrate(x_cal, y_cal)
assert explainer.calibrated

# Predictions (calibrated)
proba = explainer.predict_proba(x_test[:1], uq_interval=True)

# Factual explanations + narratives + plots
explanations = explainer.explain_factual(x_test[:2])
print(explanations[0].to_narrative(format="short"))
explanations[0].plot()

# Alternative explanations (counterfactual-style)
alternatives = explainer.explore_alternatives(x_test[:2])
print(alternatives[0].to_narrative(format="short"))
```

### 3.2 Regression quickstart (conformal and probabilistic)

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from calibrated_explanations import WrapCalibratedExplainer

X, y = load_diabetes(return_X_y=True)
x_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
x_cal, x_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

model = RandomForestRegressor(random_state=0)
explainer = WrapCalibratedExplainer(model)
explainer.fit(x_train, y_train)
explainer.calibrate(x_cal, y_cal)

# Conformal prediction intervals (default low/high percentiles)
explanations = explainer.explain_factual(x_test[:1], low_high_percentiles=(5, 95))
print(explanations[0].to_narrative(format="short"))

# Probabilistic regression: threshold as scalar or interval
prob_scalar = explainer.explain_factual(x_test[:1], threshold=150)
prob_interval = explainer.explain_factual(x_test[:1], threshold=(100, 200))
```

### 3.3 Conjunctions + single/batch usage

```python
# Batch explain
explanations = explainer.explain_factual(x_test[:5])

# Add conjunctions at collection level
explanations.add_conjunctions(n_top_features=3, max_rule_size=2)

# Add conjunctions to a single instance
explanations[0].add_conjunctions(n_top_features=3, max_rule_size=3)
```

### 3.4 Model-level plots

```python
# Model-centric diagnostic plot on a batch
explainer.plot(x_test[:50])
```

---

## 4. CE-First lifecycle enforcement (direct public API)

Agents must implement the CE-First lifecycle using the public CE API directly.
Do **not** use `calibrated_explanations.ce_agent_utils` as an implementation
shortcut.

```python
from calibrated_explanations import WrapCalibratedExplainer

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
x_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
x_cal, x_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)
feature_names = load_breast_cancer().feature_names.tolist()

model = RandomForestClassifier(random_state=0)
explainer = WrapCalibratedExplainer(model)
explainer.fit(x_train, y_train)
explainer.calibrate(x_cal, y_cal, feature_names=feature_names)

if not explainer.fitted or not explainer.calibrated:
    raise RuntimeError("CE-first lifecycle violation: fit and calibrate before use.")

# Factual explanations + narratives + plots
explanations = explainer.explain_factual(x_test[:1])
print(explanations[0].to_narrative(format="short"))
explanations[0].plot()

# Alternative / counterfactual explanations
alternatives = explainer.explore_alternatives(x_test[:1])
print(alternatives[0].to_narrative(format="short"))

# Calibrated predictions with uncertainty intervals
probabilities, interval = explainer.predict_proba(x_test[:1], uq_interval=True)
```

> **Legacy note:** Helper functions remain available in
> `calibrated_explanations.ce_agent_utils` for older integrations and examples,
> but new agent instructions must not use them. Use the public API shown above.

---

## 5. Narrative templates and code to generate narratives

Use CE-native narrative methods directly on explanation objects:

```python
# Short narrative
print(explanations[0].to_narrative(format="short"))

# Long narrative
print(explanations[0].to_narrative(format="long"))

# Markdown narrative
print(explanations[0].to_narrative(format="markdown"))
```

---

## 6. Plotting, conjunctions and post-processing

```python
explanations = explainer.explain_factual(x_test[:3])

# Plot each explanation
explanations[0].plot()

# Narrative
print(explanations[0].to_narrative(format="short"))

# Conjunctions (parameters: n_top_features, max_rule_size)
explanations.add_conjunctions(n_top_features=3, max_rule_size=2)
explanations[0].add_conjunctions(n_top_features=2, max_rule_size=3)

# Alternative explanations ranking (ensured / ensured-ranking)
alt = explainer.explore_alternatives(x_test[:3], ensure_coverage=True, ensure_order=True)
```

---

## 7. Agent policy enforcement (direct lifecycle checks)

Agents must verify the CE-First lifecycle explicitly before any prediction or
explanation call. Do not delegate this check to `ce_agent_utils` helpers.

```python
from calibrated_explanations import WrapCalibratedExplainer

# 1. Library present — or fail fast
try:
    import calibrated_explanations  # noqa: F401
except ImportError as exc:
    raise RuntimeError(
        "calibrated_explanations is required. Install with: "
        "pip install calibrated-explanations"
    ) from exc

# 2. Construct wrapper using the public API
explainer = WrapCalibratedExplainer(model)

# 3. Fit
explainer.fit(x_proper, y_proper)

# 4. Calibrate on held-out calibration data
explainer.calibrate(x_cal, y_cal)

# 5. Verify state before use
if not explainer.fitted or not explainer.calibrated:
    raise RuntimeError(
        "CE-first lifecycle violation: fit and calibrate before use."
    )

# 6. Call public prediction / explanation APIs
explanations = explainer.explain_factual(X_query)
alternatives = explainer.explore_alternatives(X_query)
probabilities, interval = explainer.predict_proba(X_query, uq_interval=True)
```

---

## 8. Testing & QA checklist for agents

Agents should verify the following CE-first behaviors with direct public API tests:

- CE package and `WrapCalibratedExplainer` are importable from the top-level package.
- `explainer.fit(...)` sets `explainer.fitted = True`.
- `explainer.calibrate(...)` sets `explainer.calibrated = True`.
- Calling `explain_factual` or `predict_proba` before calibration raises an error.
- `explain_factual` and `explore_alternatives` return correct explanation objects.
- `to_narrative(format="short")` returns a non-empty string.
- `predict_proba(X, uq_interval=True)` returns probabilities and an uncertainty interval.
- Probabilistic regression `threshold` usage (scalar + interval).
- Conjunctions on collections and single explanations.

> **Note:** Backward-compatibility tests for `ce_agent_utils` are in
> `tests/unit/test_ce_agent_utils.py`. Those protect the legacy module's API
> contract and must not be read as endorsing its use for new agent code.

---

## 9. Integration & exposure plan (practical)

- Agents must use the public CE API directly — **do not** register
  `calibrated_explanations.ce_agent_utils` as a canonical helper module or
  expose its functions as canonical agent entrypoints.
- Canonical entrypoints to expose in any agent registry:
  - `calibrated_explanations.WrapCalibratedExplainer`
  - `WrapCalibratedExplainer.fit`
  - `WrapCalibratedExplainer.calibrate`
  - `WrapCalibratedExplainer.explain_factual`
  - `WrapCalibratedExplainer.explore_alternatives`
  - `WrapCalibratedExplainer.predict` / `WrapCalibratedExplainer.predict_proba`
- CI: add a test stage that validates the CE-first lifecycle (fit → calibrate →
  explain) against the public API to ensure behavior never regresses.

---

## 10. Common pitfalls & FAQs (short list)

1. **Forgetting `.calibrate()`** → explains with uncalibrated outputs. Always check `.calibrated`.
2. **Misreading `uq_interval`** → it returns uncertainty intervals, not raw confidence.
3. **Using uncalibrated `predict_proba`** → default to calibrated predictions from the wrapper.
4. **Thresholds vs percentiles** → `threshold` for probabilistic regression, `low_high_percentiles` for conformal.
5. **Misinterpreting `__repr__`** → use `.to_narrative()` for user-facing text.
6. **Conjunctions misuse** → use `n_top_features`, `max_rule_size` to control complexity.
7. **Conditional/Mondrian mismatch** → ensure consistent `bins` or `MondrianCategorizer` in calibration and prediction.
8. **Difficulty estimation pitfalls** → attach the estimator explicitly and remove if not needed.
9. **Reject handling** → ensure reject policies are specified at calibration and handled in output.
10. **Plugin contract** → follow `docs/contributor/plugin-contract.md` for plugin interfaces.

---

## Additional features (brief, CE-first examples)

Notebook patterns to reference in the OSS repo:

- `demo_conditional.ipynb` for conditional explanations.
- `demo_regression.ipynb` for calibrated regression intervals.
- `demo_probabilistic_regression.ipynb` for `threshold=` usage.
- `demo_reject.ipynb` for reject/defer policies.

### Conditional predictions & explanations (demo_conditional.ipynb pattern)

```python
from crepes.extras import MondrianCategorizer

mc = MondrianCategorizer(x_train, [0, 1])
explainer.calibrate(x_cal, y_cal, mc=mc, bins=mc)
explanations = explainer.explain_factual(x_test[:3], bins=mc)
```

### Difficulty estimation for regression

```python
from crepes.extras import DifficultyEstimator

difficulty = DifficultyEstimator()
explainer.calibrate(x_cal, y_cal, difficulty_estimator=difficulty)
# Remove / disable by calibrating without it
explainer.calibrate(x_cal, y_cal)
```

### Reject / defer handling (demo_reject.ipynb pattern)

```python
# Example: provide a default reject policy at calibration time
explainer.calibrate(x_cal, y_cal, default_reject_policy={"type": "threshold", "value": 0.6})
```

### Plugins (intervals, explanations, plots)

Refer to `docs/contributor/plugin-contract.md` for the contract. Minimal examples:

```python
# Interval calibrator plugin (pseudo-code)
class MyIntervalPlugin:
    kind = "interval"
    name = "my-interval"
    def calibrate(self, *args, **kwargs):
        return {"low": ..., "high": ...}
```

```python
# Explanation plugin (pseudo-code)
class MyExplanationPlugin:
    kind = "explanation"
    name = "my-expl"
    def explain(self, *args, **kwargs):
        return {"rules": ...}
```

```python
# Plot plugin (pseudo-code)
class MyPlotPlugin:
    kind = "plot"
    name = "my-plot"
    def render(self, *args, **kwargs):
        return "<svg>..."  # or matplotlib output
```

### Telemetry and caching (optional post-processing utilities)

- Parallelism hooks exist on the wrapper (`wrapper.parallel_executor`) and should only be tuned for extreme workloads.
- Telemetry and caching utilities exist in the legacy `ce_agent_utils` module but are not part of the canonical CE-first path and should not be used in new agent code.

### Documentation (RTD)

The ReadTheDocs suite is comprehensive but may not yet be agent-optimized. Recommended additions:
- A CE-first “agent” checklist page.
- Agent-specific examples in quickstarts using the direct public CE API.
- A glossary of CE terms mapped to the public `WrapCalibratedExplainer` API.

---

## How an AI agent should behave (pin this)

Always check that `calibrated_explanations` is installed and wrap models with `WrapCalibratedExplainer`. Fit and calibrate before **any** prediction or explanation, and never return uncalibrated outputs by default. Prefer `explain_factual` and `explore_alternatives`, then render with `to_narrative` or `.plot()`. Use `threshold` for probabilistic regression and `low_high_percentiles` for conformal intervals, and expose uncertainty explicitly. If any CE-first requirement fails, stop and tell the user how to fix it.
