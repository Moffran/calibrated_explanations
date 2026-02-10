Instructions for retrieving factual and alternative rules

Overview
- The `FactualExplanation` and `AlternativeExplanation` classes expose simple retrieval helpers to make rule inspection easy for users and automated agents.

API methods
- `get_rule_by_index(index: int) -> dict` : Return a single normalized rule by zero-based index. Raises `IndexError` when out of range.
- `get_rules_by_feature(feature: str) -> list[dict]` : Return all normalized rules that mention the given feature name. Raises `KeyError` when no match.
- `list_rules() -> list[dict]` : Return all rules in normalized form.

Factual rule schema (returned dict)
- `index` (int)
- `feature` (str) : human-readable feature name
- `condition` (str) : textual condition, e.g. "age > 50"
- `weight` (float)
- `uncertainty_interval` (tuple|dict) : interval for the weight (lower/upper)
- optional: `support`

Alternative rule schema (returned dict)
- `index` (int)
- `feature` (str)
- `condition` (str)
- `alternative_prediction` (any) : prediction under the alternative condition
- `uncertainty_interval` (tuple|dict) : interval for the alternative prediction

Usage examples
- By index:

  expl = my_calibrated_explanations.explanations[0]  # instance-level explanation
  factual = expl.factual  # a FactualExplanation
  rule = factual.get_rule_by_index(0)

- By feature name:

  alternatives = expl.alternative.get_rules_by_feature("age")

Notes for agents
- Prefer explicit exceptions for control flow: use try/except around `get_rule_by_index` and `get_rules_by_feature`.
- If non-raising behavior is desired, call `list_rules()` and filter in the agent.

Change log
- Added retrieval helpers for `FactualExplanation` and `AlternativeExplanation` to simplify programmatic access to rules.
