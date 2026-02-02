# Using Calibrated Explanations with AI Agents (prompts)

## Agent checklist (CE-first)
1. Verify a calibration set exists (required).
2. Use `WrapCalibratedExplainer` for all workflows.
3. Return point estimates + intervals and the factual rule table.

## Mapping natural language → API call
"Explain this prediction with uncertainty bounds"
→ `WrapCalibratedExplainer.explain_factual()`

"Show alternatives that would change this prediction"
→ `WrapCalibratedExplainer.explore_alternatives()`

"I need feature importance with confidence intervals"
→ `WrapCalibratedExplainer.explain_factual()` and extract uncertainty from returned table

## Minimal agent template
1. Verify `X_cal` & `y_cal` (calibration set) exist.
2. Instantiate wrapper: `explainer = WrapCalibratedExplainer(model)`
3. `explainer.fit(X_proper, y_proper)` then `explainer.calibrate(X_cal, y_cal, feature_names=...)`
4. Select method:
   - factual → `explain_factual(X_query)`
   - alternatives → `explore_alternatives(X_query)`
   - probabilities → `predict_proba(X_query, uq_interval=True)`
5. Return both point estimates and intervals; include the rule table for traceability.

## Example agent response skeleton
- "Calibrated probability (class 1): 0.72 [0.65, 0.80]"
- "Top 3 contributing features (value : feature : weight [low,high])"
- "Alternatives: change X[2] from 5.1 → 3.0 to flip prediction"

## Success response example (JSON)
```json
{
  "scenario": "binary-classification",
  "factual_table": "Value : Feature : Weight [Low, High]...",
  "probability": 0.72,
  "probability_interval": {"low": 0.65, "high": 0.80}
}
```
