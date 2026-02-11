# Explanation retrieval

## Instructions for retrieving factual and alternative rules

### Overview
- The `FactualExplanation` and `AlternativeExplanation` classes expose simple retrieval helpers to make rule inspection easy for users and automated agents.

### API methods
- `get_rule_by_index(index: int) -> dict` — Return a single normalized rule by zero-based index. Raises `IndexError` when out of range.
- `get_rules_by_feature(feature: str) -> list[dict]` — Return all normalized rules that mention the given feature name. Raises `KeyError` when no match.
- `list_rules() -> list[dict]` — Return all rules in normalized form.

### Factual rule schema (returned dict)
- `index` (int)
- `feature` (str) — human-readable feature name
- `condition` (str) — textual condition, e.g. "age > 50"
- `weight` (float)
- `uncertainty_interval` (tuple|dict) — interval for the weight (lower/upper)
- optional: `support`

### Alternative rule schema (returned dict)
- `index` (int)
- `feature` (str)
- `condition` (str)
- `alternative_prediction` (any) — prediction under the alternative condition
- `uncertainty_interval` (tuple|dict) — interval for the alternative prediction

### Usage examples
- By index:

  ```python
  expl = my_calibrated_explanations.explanations[0]  # instance-level explanation
  factual = expl.factual  # a FactualExplanation
  rule = factual.get_rule_by_index(0)
  ```

- By feature name:

  ```python
  alternatives = expl.alternative.get_rules_by_feature("age")
  ```

## Filtering rules by size (`filter_rule_sizes`)

When you add conjunctive rules (for example via `add_conjunctions(...)`), rules can have different *sizes*:

- Size `1`: an atomic (single-feature) rule.
- Size `k > 1`: a conjunctive rule built from `k` atomic conditions.

You can filter an explanation down to only the sizes you care about using `filter_rule_sizes(...)`.

### How rule size is computed

Rule size is inferred from the rule's `feature` payload:

- If `feature` is a scalar (e.g. `0` or `"age"`), the size is `1`.
- If `feature` is a sequence/array (e.g. `[0, 1]`), the size is the number of elements.

### API

`filter_rule_sizes` exists on both a single instance-level explanation and on the collection:

- `CalibratedExplanation.filter_rule_sizes(rule_sizes=..., size_range=..., copy=True)`
- `CalibratedExplanations.filter_rule_sizes(rule_sizes=..., size_range=..., copy=True)`

Exactly one of the following must be provided:

- `rule_sizes`: an `int` or a sequence of `int` sizes to keep.
- `size_range=(min_size, max_size)`: an inclusive range of sizes to keep.

If `copy=True` (default), the original object is not mutated.

### Examples

Keep only atomic rules (exclude conjunctions):

```python
expl = calibrated_explanations[0]
atomic_only = expl.filter_rule_sizes(rule_sizes=1)
```

Keep only conjunctions of size 2 and 3:

```python
expl = calibrated_explanations[0].add_conjunctions(max_rule_size=3)
pair_and_triple = expl.filter_rule_sizes(rule_sizes=[2, 3])
```

Keep a size range (inclusive):

```python
expl = calibrated_explanations[0].add_conjunctions(max_rule_size=4)
up_to_triples = expl.filter_rule_sizes(size_range=(1, 3))
```

Filter the entire collection in one call:

```python
filtered = calibrated_explanations.filter_rule_sizes(rule_sizes=[1, 2])
```

### Notes for agents
- Prefer explicit exceptions for control flow: use try/except around `get_rule_by_index` and `get_rules_by_feature`.
- If non-raising behavior is desired, call `list_rules()` and filter in the agent.

### Change log
- Added retrieval helpers for `FactualExplanation` and `AlternativeExplanation` to simplify programmatic access to rules.
