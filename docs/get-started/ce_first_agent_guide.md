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
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
X_cal, X_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

model = RandomForestClassifier(random_state=0)
explainer = WrapCalibratedExplainer(model)

# Fit + calibrate
explainer.fit(X_train, y_train)
assert explainer.fitted
explainer.calibrate(X_cal, y_cal)
assert explainer.calibrated

# Predictions (calibrated)
proba = explainer.predict_proba(X_test[:1], uq_interval=True)

# Factual explanations + narratives + plots
explanations = explainer.explain_factual(X_test[:2])
print(explanations[0].to_narrative(format="short"))
explanations[0].plot()

# Alternative explanations (counterfactual-style)
alternatives = explainer.explore_alternatives(X_test[:2])
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
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
X_cal, X_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

model = RandomForestRegressor(random_state=0)
explainer = WrapCalibratedExplainer(model)
explainer.fit(X_train, y_train)
explainer.calibrate(X_cal, y_cal)

# Conformal prediction intervals (default low/high percentiles)
explanations = explainer.explain_factual(X_test[:1], low_high_percentiles=(5, 95))
print(explanations[0].to_narrative(format="short"))

# Probabilistic regression: threshold as scalar or interval
prob_scalar = explainer.explain_factual(X_test[:1], threshold=150)
prob_interval = explainer.explain_factual(X_test[:1], threshold=(100, 200))
```

### 3.3 Conjunctions + single/batch usage

```python
# Batch explain
explanations = explainer.explain_factual(X_test[:5])

# Add conjunctions at collection level
explanations.add_conjunctions(n_top_features=3, max_rule_size=2)

# Add conjunctions to a single instance
explanations[0].add_conjunctions(n_top_features=3, max_rule_size=3)
```

### 3.4 Model-level plots

```python
# Model-centric diagnostic plot on a batch
explainer.plot(X_test[:50])
```

---

## 4. Agent helper library (runnable code) — CE-First by design

> The canonical helper is provided in `calibrated_explanations/ce_agent_utils.py` and enforces CE-First at runtime. It **never** creates a custom wrapper class; it only uses `WrapCalibratedExplainer`.

### Example usage

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations.ce_agent_utils import (
    ensure_ce_first_wrapper,
    fit_and_calibrate,
    explain_and_narrate,
    explain_and_summarize,
    wrap_and_explain,
)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
X_cal, X_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

model = RandomForestClassifier(random_state=0)
wrapper = ensure_ce_first_wrapper(model)
fit_and_calibrate(wrapper, X_train, y_train, X_cal, y_cal)
explanations, narrative = explain_and_narrate(wrapper, X_test[:1], mode="factual")
print(narrative)

# One-line CE-first flow
payload = wrap_and_explain(model, X_train, y_train, X_cal, y_cal, X_test[:1])
print(payload["narrative"])

# USP-focused flow (conjunctions + UQ + probabilistic regression metadata)
usp = explain_and_summarize(
    wrapper,
    X_test[:2],
    add_conjunctions_params={"n_top_features": 3, "max_rule_size": 2},
)
print(usp["summary"])
```

### Defensive behavior

The helper probes for required methods and kwargs such as `threshold`, `uq_interval`, and `bins`. If the API differs, it either adapts or raises a clear error.

---

## 5. Narrative templates and code to generate narratives

The helper includes short/long/bullet templates and exposes them via `NARRATIVE_TEMPLATES`. Example use:

```python
from calibrated_explanations.ce_agent_utils import explain_and_narrate

explanations, short_text = explain_and_narrate(wrapper, X_test[:1], narrative_format="short")
explanations, long_text = explain_and_narrate(wrapper, X_test[:1], narrative_format="long")
```

You can also use CE-native narratives:

```python
print(explanations[0].to_narrative(format="short"))
print(explanations[0].to_narrative(format="markdown"))
```

---

## 6. Plotting, conjunctions and post-processing

```python
explanations = explainer.explain_factual(X_test[:3])

# Plot each explanation
explanations[0].plot()

# Narrative
print(explanations[0].to_narrative(format="short"))

# Conjunctions (parameters: n_top_features, max_rule_size)
explanations.add_conjunctions(n_top_features=3, max_rule_size=2)
explanations[0].add_conjunctions(n_top_features=2, max_rule_size=3)

# Alternative explanations ranking (ensured / ensured-ranking)
alt = explainer.explore_alternatives(X_test[:3], ensure_coverage=True, ensure_order=True)
```

---

## 7. Agent policy and enforcement code (runnable)

Use `enforce_ce_first_and_execute` to enforce CE-First before any action.

```python
from calibrated_explanations.ce_agent_utils import enforce_ce_first_and_execute

# Example: enforce CE-first before explaining
explanations, narrative = enforce_ce_first_and_execute(
    lambda w, x: w.explain_factual(x),
    wrapper,
    X_test[:1],
)
```

Policy snippet (Python dict, importable):

```python
from calibrated_explanations.ce_agent_utils import CE_FIRST_POLICY
print(CE_FIRST_POLICY)
```

---

## 8. Testing & QA checklist for agents

Minimal pytest checks are provided in `tests/unit/test_ce_agent_utils.py`:

- CE presence and `WrapCalibratedExplainer` importability.
- `ensure_ce_first_wrapper` behavior for raw model and wrapper.
- `fit_and_calibrate` setting `.fitted` and `.calibrated`.
- `explain_and_narrate` failing before calibration and succeeding after.
- Probabilistic regression `threshold` usage (scalar + interval).
- Conjunctions on collections and single explanations.
- Optional feature probes (difficulty estimator, Mondrian categorizer, reject policy, plugins) with warning fallback.

---

## 9. Integration & exposure plan (practical)

- Register `calibrated_explanations.ce_agent_utils` as a canonical helper module in your agent registry.
- Canonical entrypoints to expose:
  - `calibrated_explanations.ce_agent_utils.ensure_ce_first_wrapper`
  - `calibrated_explanations.ce_agent_utils.fit_and_calibrate`
  - `calibrated_explanations.ce_agent_utils.explain_and_narrate`
  - `calibrated_explanations.ce_agent_utils.wrap_and_explain`
- CI: add a test stage that executes CE-first tests from `tests/unit/test_ce_agent_utils.py` to ensure CE-first behavior never regresses.

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

mc = MondrianCategorizer(X_train, [0, 1])
explainer.calibrate(X_cal, y_cal, mc=mc, bins=mc)
explanations = explainer.explain_factual(X_test[:3], bins=mc)
```

### Difficulty estimation for regression

```python
from crepes.extras import DifficultyEstimator

difficulty = DifficultyEstimator()
explainer.calibrate(X_cal, y_cal, difficulty_estimator=difficulty)
# Remove / disable by calibrating without it
explainer.calibrate(X_cal, y_cal)
```

### Reject / defer handling (demo_reject.ipynb pattern)

```python
# Example: provide a default reject policy at calibration time
explainer.calibrate(X_cal, y_cal, default_reject_policy={"type": "threshold", "value": 0.6})
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

### Telemetry, caching, and parallelism

- Use `set_telemetry_hook(...)` in `ce_agent_utils` to emit events on calibration/explanation.
- Optional caching (via `optional_cache`) can reduce repeated calls in large-scale runs.
- Parallelism hooks exist on the wrapper (`wrapper.parallel_executor`) and should only be tuned for extreme workloads.

### Documentation (RTD)

The ReadTheDocs suite is comprehensive but may not yet be agent-optimized. Recommended additions:
- A CE-first “agent” checklist page.
- Agent-specific examples in quickstarts (e.g., `explain_and_narrate`).
- A glossary of CE terms mapped to helper APIs.

---

## How an AI agent should behave (pin this)

Always check that `calibrated_explanations` is installed and wrap models with `WrapCalibratedExplainer`. Fit and calibrate before **any** prediction or explanation, and never return uncalibrated outputs by default. Prefer `explain_factual` and `explore_alternatives`, then render with `to_narrative` or `.plot()`. Use `threshold` for probabilistic regression and `low_high_percentiles` for conformal intervals, and expose uncertainty explicitly. If any CE-first requirement fails, stop and tell the user how to fix it.
