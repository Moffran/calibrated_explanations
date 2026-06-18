# CE Parameter Reference

Canonical definitions for the five user-facing parameters where naming confusion
is most likely. Read this page before filing a bug about unexpected behaviour
involving thresholds, confidence values, or significance levels.

Cross-references: [terminology.md](terminology.md),
[probabilistic_regression.md](probabilistic_regression.md),
[ADR-038: Call-time Configuration Taxonomy](https://github.com/Moffran/calibrated_explanations/blob/main/development/adrs/ADR-038-call-time-configuration-taxonomy.md).

---

## Disambiguation table — floats in (0, 1)

Three parameters are all floats in (0, 1) and are mathematically related, but they
are used in completely different contexts. Confusing them is the most common source
of subtle bugs.

| Current name | Canonical name (post-Task 17) | Role | Context | User-settable? |
|---|---|---|---|---|
| `confidence` | `reject_confidence` | Reject coverage target: minimum p-value to *accept* a prediction | `predict_reject` / reject orchestrator | YES |
| `significance` | `GuardedOptions.confidence` | Guard conformity threshold; `significance = 1 − confidence` | `explain_factual(..., guarded_options=GuardedOptions(confidence=...))` / `_guarded_explain` | YES (experimental) |
| `confidence_level` | `confidence_level` (unchanged) | Regression coverage level; derived from `low_high_percentiles` but also passable directly as an alternative to `threshold` | Regression methods | YES (alternative to `threshold`) |

**Key relationship:** `significance` and `confidence` (reject) are mathematical
inverses of the same conformal threshold concept. `significance = 0.1` is the same
conformity requirement as `confidence = 0.9`. The planned rename (Task 17) resolves
this by expressing both as coverage values.

---

## `threshold`

**Definition:** The regression output value used to define a binary event for
probabilistic regression. The explainer produces calibrated probabilities for
P(y ≤ threshold).

**Type and valid range:** `float`, `int`, or array-like (one value per instance for
instance-specific thresholds). `None` means "not applicable" — used only in
regression mode.

**Applies to:** `explain_factual`, `explore_alternatives`, `predict_proba` (in
regression mode). Ignored for classification tasks.

**Behavior:** When set, the CE regression path calls
`IntervalRegressor.predict_probability(x, y_threshold=threshold)`. The parameter is
renamed to `y_threshold` at the internal `IntervalRegressor` boundary to match the
`crepes` API convention; this split is an implementation detail and does not affect
the public interface.

**Disambiguation:**
- NOT `confidence_level`: `confidence_level` specifies the *width* of the
  calibrated interval (e.g., a 90 % interval), not a target output value. They are
  mutually exclusive: passing both raises `ConfigurationError` (enforced by
  `EXCLUSIVE_PARAM_GROUPS` in `api/params.py`).
- NOT `confidence`: `confidence` is the reject coverage target and belongs to the
  reject path, not the regression path.

---

## `low_high_percentiles`

**Definition:** A `(low, high)` percentile tuple that sets the width of the
calibrated uncertainty interval for regression. For example, `(5, 95)` produces a
90 % calibrated interval.

**Type and valid range:** Tuple of two floats, each in `[0, 100]`, with
`low < high`. Default: `(5, 95)`.

**Applies to:** All regression explanation methods (`explain_factual`,
`explore_alternatives`). Ignored for classification.

**Behavior:** Determines the lower and upper bounds of the calibrated interval.
The derived metric `confidence_level = (high - low) / 100` represents the nominal
coverage probability of the interval.

**Disambiguation:**
- NOT `threshold`: `threshold` is a target output value; `low_high_percentiles`
  governs interval width.

---

## `confidence_level`

**Definition:** The nominal coverage probability of the calibrated regression
interval, expressed as a fraction in (0, 1). Derived from `low_high_percentiles`
as `(high - low) / 100`. For example, `low_high_percentiles=(5, 95)` yields
`confidence_level=0.90`.

**Type and valid range:** `float` in (0, 1). Default: derived from
`low_high_percentiles=(5, 95)`, so `0.90`.

**Applies to:** Regression explanation methods. May be passed directly as a
caller-specified value instead of `low_high_percentiles`.

**Behavior:** When passed directly, CE reverses the formula to reconstruct
`low_high_percentiles`: `low = (1 - confidence_level) / 2 * 100`,
`high = 100 - low`. This makes it an *alternative input* to `low_high_percentiles`,
not a derived output.

**User-settable:** YES — `confidence_level` and `threshold` are mutually exclusive
alternatives for specifying the regression configuration. Passing both raises a
`ConfigurationError` (enforced by `EXCLUSIVE_PARAM_GROUPS` in `api/params.py`).

**Disambiguation:**
- NOT `confidence` (reject path): `confidence` is the reject coverage target and
  applies to the reject orchestrator, not regression intervals.
- NOT `significance`: `significance` is the guarded conformity p-value threshold
  and does not govern interval width.
- NOT `threshold`: mutually exclusive alternatives — use one or the other.

---

## `confidence`

> **Planned rename (Task 17):** `confidence` → `reject_confidence` to eliminate
> ambiguity with `confidence_level`.

**Definition:** The reject coverage target — the minimum calibrated probability
required to *accept* a prediction. A prediction with conformal p-value below
`1 - confidence` is rejected.

**Type and valid range:** `float` in (0, 1). Default: `0.95`.

**Applies to:** `predict_reject`, `apply_policy`, and the reject orchestrator.

**Behavior:** Higher values are *stricter* (fewer predictions accepted). A value of
`0.95` means predictions must have at least a 0.95 conformal coverage to be
accepted. Internally the reject path computes `epsilon = 1 - confidence` as the
conformal significance level.

**Disambiguation:**
- NOT `significance`: they are mathematical inverses
  (`significance = 1 − confidence`), but `significance` belongs to the guarded
  explanation path, while `confidence` belongs to the reject path. After Task 17,
  both surfaces will use the coverage convention so this inverse relationship will
  not need explaining.
- NOT `confidence_level`: `confidence_level` is a regression interval width
  parameter; `confidence` is a reject decision threshold.

---

## `significance`

> **Planned rename (Task 17):** `significance` → `GuardedOptions.confidence`
> using the coverage convention (`confidence = 1 − significance`).

**Definition:** The conformity significance level for the in-distribution guard.
The guard accepts a calibration bin if its KNN-based conformal p-value is ≥
`significance`; bins below this threshold are considered out-of-distribution and
pruned from the explanation.

**Type and valid range:** `float` in (0, 1]. Default: `0.1`.

**Applies to:** `explain_factual(guarded=True)`, `explore_alternatives(guarded=True)`,
and the internal `_guarded_explain` module. This parameter is part of the
`[EXPERIMENTAL]` guarded explanation API.

**Behavior:** A **larger** `significance` value is **stricter** (fewer bins
accepted). This is the reverse of how `confidence` works in the reject path: for
`significance`, raising the value makes the guard harder to pass; for `confidence`,
raising the value makes the reject threshold harder to pass. The planned rename to
`GuardedOptions.confidence` resolves this counter-intuitive direction by expressing
both surfaces as coverage values where higher = more inclusive.

**Disambiguation:**
- NOT `confidence` (reject path): mathematical inverse
  (`significance = 1 − confidence`), different path.
- NOT `confidence_level`: unrelated; `confidence_level` governs regression interval
  width, not conformity testing.
