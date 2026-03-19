# Terminology: thresholded vs probabilistic regression

This page maps two terms used for the same CE regression mode.

In user-facing docs and APIs, use `probabilistic regression`. In architecture and implementation discussions, `thresholded regression` may be used to describe the threshold mechanism.

Both terms refer to regression with a non-`None` `threshold` query that returns calibrated event probabilities with interval bounds.

For full guarantees, assumptions, explicit non-guarantees, and feature-level interval limits, use {doc}`calibrated_interval_semantics`.

## Use in docs

- User-facing guides, quickstarts, notebooks, and API docs: `probabilistic regression`
- ADRs and implementation details that discuss mechanism: `thresholded regression`
- If both terms appear on one page, state once that they map to the same mode

## Section 1: Definition analysis

### 1.1 What does "probabilistic regression" mean?

Definition source: `docs/foundations/concepts/probabilistic_regression.md`.

It refers to regression predictions queried as probability events by threshold, for example `P(y <= t)` or interval events. The output is a calibrated event probability plus interval bounds.

User-level API pattern:

```python
probabilities, probability_interval = explainer.predict_proba(
    x_test[:1],
    threshold=150,
    uq_interval=True,
)
```

### 1.2 What does "thresholded regression" mean?

Definition source: `docs/improvement/adrs/ADR-021-calibrated-interval-semantics.md`.

It is the same regression mode described from the implementation angle:

- Regression output is queried through threshold events
- CPS provides event scoring
- Venn-Abers calibrates event probabilities

Implementation path example from `IntervalRegressor.predict_probability()`:

```python
# Converts regression predictions to probabilities by thresholding
proba = self.split["cps"].predict(y_hat=..., y=y_threshold, ...)
# Then calibrates with Venn-Abers
va = VennAbers(None, (self.ce.y_cal[cal_va] <= y_threshold).astype(int), ...)
```

### 1.3 Evidence of equivalence

- ADR-021 section "Thresholded regression: CPS probabilities calibrated by Venn-Abers" explicitly describes the same path as the probabilistic regression flow.
- Runtime API signal is identical: this mode is selected by providing `threshold`.

### 1.4 Why two terms exist

| Aspect | "Thresholded regression" | "Probabilistic regression" |
| --- | --- | --- |
| Emphasis | Mechanism (threshold operation) | Output (calibrated probabilities) |
| Primary audience | Architecture and implementation contributors | Practitioners and API users |
| Typical context | ADRs, plugin internals, design notes | Quickstarts, concept guides, notebooks |

## Section 2: Terminology inventory

### 2.1 Representative "probabilistic regression" usage

| File | Context |
| --- | --- |
| `README.md` | Feature and quickstart routing |
| `docs/get-started/index.md` | Navigation and mode routing |
| `docs/get-started/quickstart_regression.md` | Task workflow |
| `docs/foundations/concepts/probabilistic_regression.md` | Dedicated concept page |
| `notebooks/core_demos/demo_probabilistic_regression.ipynb` | End-to-end example |

### 2.2 Representative "thresholded regression" usage

| File | Context |
| --- | --- |
| `docs/improvement/adrs/ADR-021-calibrated-interval-semantics.md` | Architecture semantics |
| `docs/improvement/adrs/ADR-013-interval-calibrator-plugin-strategy.md` | Plugin strategy terminology |
| `docs/improvement/legacy_user_api_contract.md` | Historical contract references |
| `docs/foundations/governance/optional_telemetry.md` | Technical telemetry context |

### 2.3 Code usage patterns

- Public call sites use `threshold=` as the mode switch.
- Internal APIs include both `threshold` and `y_threshold` names.
- Explanation containers expose probabilistic-regression state through `is_probabilistic_regression`.

## Section 3: Context-specific usage

### 3.1 User-facing documentation

Preferred term: `probabilistic regression`.

Reason: it communicates task intent and expected output without requiring implementation knowledge.

### 3.2 Technical architecture and implementation

Preferred term: `thresholded regression` when discussing mechanics.

Reason: it makes the threshold-event conversion explicit for contributors and plugin authors.

### 3.3 Evaluation and benchmarking

Benchmark materials often use `thresholded regression` to separate this mode from percentile regression.

## Section 4: Historical issues and current state

### 4.1 Historical issue: missing explicit mapping

Earlier materials mixed terms without a clear mapping statement.

Current state: ADR-021 now includes explicit terminology guidance, and this page serves as the Tier 3 terminology reference route.

### 4.2 Historical issue: mixed naming in tests and comments

Earlier test docstrings and comments mixed the terms without context labels.

Current state: core user-facing naming is `probabilistic regression`; technical notes may still use `thresholded regression` when describing mechanism.

### 4.3 Ongoing risk

Terminology drift can return when new docs are added quickly.

Control: keep this mapping in Tier 3 reference pages and keep Tier 1 and Tier 2 pages mode-specific with short semantics routing to {doc}`calibrated_interval_semantics`.

## Section 5: Recommended terminology policy

- Canonical user-facing term: `probabilistic regression`
- Allowed technical mechanism term: `thresholded regression`
- Do not present them as different modes
- When uncertain, prefer user-facing term and add one parenthetical mechanism note if needed

## Section 6: Related references

- {doc}`calibrated_interval_semantics`
- `docs/improvement/adrs/ADR-021-calibrated-interval-semantics.md`
- `docs/improvement/adrs/ADR-013-interval-calibrator-plugin-strategy.md`
- {doc}`probabilistic_regression`
- {doc}`terminology`

Entry-point tier: Tier 3.

