# Guarded Explanation Evaluation Suite

This directory contains empirical evaluations of the **guarded explanation** feature
introduced in ADR-032. Each scenario asks one concrete question. Each metric answers
one sub-question. If a metric does not answer a question clearly, it is not reported.

---

## Why evaluate the guard?

The guard's ADR-032 promise is: *"fewer, better rules"* — not just fewer rules.
It achieves this by filtering perturbations whose representative values fall outside
the high-density region of the calibration distribution (measured via KNN conformal
p-values). Without ground-truth OOD labels, it is impossible to know from Scenario A
alone whether the guard is filtering the *right* instances or filtering arbitrarily.

The evaluation suite decomposes the promise into three testable claims:

| Claim | What it means | Tested by |
|---|---|---|
| **Detection** | p-values are a reliable OOD signal — OOD perturbations get low p-values, in-distribution ones get high p-values | A (indirect), B (direct) |
| **Calibration** | The false rejection rate on in-distribution data is ≤ the chosen significance α (conformal validity guarantee) | B |
| **Correctness** | The guard does not break CE invariants, payload schemas, or API contracts across task types and parameter extremes | C, D, E |

---

## Scenarios

### Scenario A — Domain plausibility (synthetic constraint)

**Script**: `guarded_vs_standard_scenario_a.py`

**Question**: When the guard removes a rule, does the removed rule's representative
value violate a known domain constraint?

**Why**: Scenario A is the one setting where we have a ground-truth measure of
plausibility (the constraint `x2 ≤ 2*x1 + 3`). We can directly ask: does the guard
preferentially remove *implausible* perturbations? This addresses the detection claim
indirectly — not by measuring p-value quality against OOD labels, but by measuring
whether filtered rules correspond to out-of-constraint instances.

**Key metric**:

- **`violation_rate`** — fraction of emitted rules whose representative value violates
  the domain constraint.
  - *How it answers the question*: if the guard is working as intended,
    `violation_rate(guarded) < violation_rate(standard)` — the guard preferentially
    removes constraint-violating perturbations. If guarded and standard have
    similar violation rates, the guard is not selecting better rules.
  - *Expected range*: guarded violation rate < standard violation rate, with the gap
    widening as significance increases (stricter guard).
  - *Red flag*: guarded ≥ standard violation rate suggests the guard is filtering
    in-constraint perturbations while letting out-of-constraint ones through.

---

### Scenario B — OOD detection quality

**Script**: `scenario_b_ood_detection_quality.py`

**Question**: Do the guard's p-values reliably separate out-of-distribution perturbations
from in-distribution perturbations?

**Why**: This is the most direct test of the detection claim. We generate instances
with a known ground-truth OOD label (shifted from the calibration distribution by a
controlled amount) and measure whether the guard's p-values correctly score them as
anomalous. Without this test, there is no way to know if the guard is detecting OOD
or filtering randomly.

**Setup**: Calibration from N(0, I_d). In-distribution test from N(0, I_d). OOD test
from N(shift_vector, I_d) at three shift magnitudes: mild (1σ), moderate (2σ),
extreme (5σ). Grid: n_dim ∈ {2, 5, 10, 20}, n_neighbors ∈ {1, 5, 10, 20},
normalize_guard ∈ {True, False}.

**Key metrics**:

- **`auroc`** — AUROC of (1 − mean_p_value_per_instance) as a binary OOD classifier
  against the ground-truth label.
  - *How it answers the question*: AUROC is threshold-free (does not depend on the
    chosen significance). It measures whether the guard assigns systematically lower
    p-values to OOD instances than to in-distribution ones. AUROC = 1.0 means perfect
    separation; AUROC = 0.5 means random — the guard cannot detect OOD.
  - *Expected range*: > 0.80 for moderate+ shift with n_dim ≤ 5 and normalize_guard=True.
    Degrades toward 0.5 as n_dim increases (curse of dimensionality).
  - *Red flag*: AUROC < 0.60 for extreme shift at n_dim ≤ 5 — the guard is near-random
    even when the distributional shift is severe.

- **`fpr_at_significance`** — fraction of in-distribution instance intervals where
  p_value < significance (i.e., the guard wrongly rejects an in-distribution perturbation).
  - *How it answers the question*: by conformal prediction theory, the expected false
    rejection rate is ≤ α. This metric verifies that the conformal calibration guarantee
    holds in practice.
  - *Expected range*: ≈ significance. Slightly below is acceptable (conservative).
  - *Red flag*: materially above significance (e.g., FPR > 0.15 when α = 0.10) — the
    calibration assumption is violated, possibly because the test set comes from a
    shifted distribution or the calibration set is too small.

**What this scenario exposes**:

- *Curse of dimensionality*: AUROC degrades as n_dim increases because KNN distances
  concentrate. The exact n_dim threshold where the guard becomes unreliable is quantified.
- *normalize_guard=False failure*: a single dominant-scale feature swamps the KNN
  distance metric, making detection unreliable regardless of shift magnitude.
- *n_neighbors=1 instability*: high AUROC variance across seeds signals unreliable
  guard behavior from single-neighbor distance estimates.
- *Unsmoothed p-value estimator*: `InDistributionGuard` computes
  `p_value = count(cal_scores >= test_score) / n_cal` with no +1 smoothing.
  A test instance whose KNN distance exceeds all n_cal calibration distances gets
  p_value = 0, which is below any positive significance threshold — the guard
  correctly rejects it even at very low significance (e.g., 0.001). This is
  documented in E2 of Scenario E.

---

### Scenario C — Regression invariants

**Script**: `scenario_c_regression.py`

**Question**: Does the guard preserve the interval invariant (`low ≤ predict ≤ high`)
in regression tasks, where a different internal code path is exercised?

**Why**: The `_guarded_explain.py` code has separate handling for regression vs.
classification — a different discretizer criterion and different interval semantics.
Critically, the regression path uses `warnings.warn` instead of `raise` for interval
violations. This means a regression-specific bug would go undetected in normal use.
Scenario A tests classification only; Scenario C closes that gap.

**Setup**: Synthetic sin(x)+noise with known OOD boundary at |x| > 3.5; sklearn
diabetes dataset. Models: RandomForestRegressor and Ridge.

**Key metric**:

- **`n_invariant_violations`** — count of audit interval records where
  `predict < low − ε` or `predict > high + ε` (ε = 1e-6).
  - *How it answers the question*: the interval invariant is a hard contract. Any
    violation is a bug in the regression code path. Because the code uses
    `warnings.warn` instead of `raise` for regression, violations pass silently in
    normal use — this scenario catches them by inspecting every emitted interval.
  - *Expected range*: exactly 0.
  - *Red flag*: any count > 0. See `invariant_violations.csv` for details.

**Secondary diagnostic (not a pass/fail criterion)**:

- `fraction_removed_ood` vs `fraction_removed_id` on the synthetic dataset —
  the guard should remove more intervals for |x| > 3.5 than for |x| ≤ 3. This
  confirms the guard responds to actual distributional shift in regression mode.

---

### Scenario D — Real dataset correctness

**Script**: `scenario_d_real_datasets.py`

**Question**: Does the guard's API remain correct (no exceptions, complete audit
payloads) across the full variety of real-world task types — multiclass classification,
high-dimensional data, small calibration sets?

**Why**: Synthetic scenarios are designed around the guard's assumptions. Real datasets
expose gaps we did not anticipate. In particular: multiclass classification has a
different `prediction["classes"]` shape that may break the audit payload; Bonferroni
correction on many-bin features may over-filter; high-dimensional data with tiny
calibration sets may make the guard degenerate. This scenario is a correctness sweep,
not a performance benchmark.

**Datasets**: breast_cancer (30 features, binary), iris (4 features, 3-class),
wine (13 features, 3-class), digits_01 (64 features, binary).

**Key metrics**:

- **`audit_field_completeness`** — boolean: every interval record in `get_guarded_audit()`
  contains all fields defined in ADR-032 Addendum.
  - *How it answers the question*: a missing field means the payload contract is broken.
    The most likely cause is a multiclass-specific bug where `prediction["classes"]` is
    a vector rather than a scalar, causing a downstream field to be omitted or malformed.
  - *Expected range*: True for every record.
  - *Red flag*: any False. See `audit_completeness_details.csv` for which fields are missing.

- **`fraction_instances_fully_filtered`** — fraction of test instances with
  `intervals_emitted = 0` (zero rules returned).
  - *How it answers the question*: the API must not crash on empty explanations (E1 in
    Scenario E tests this for classification). But if > 10% of instances get zero rules
    at α = 0.10, the guard is so aggressive that it is impractical on real data. This
    can happen with small calibration sets (iris: ~30 instances → p-value step ≈ 0.033)
    or with `use_bonferroni=True` on high-cardinality features.
  - *Expected range*: < 5% at significance=0.10 on real in-distribution data.
  - *Red flag*: > 10% at significance=0.10.

---

### Scenario E — Edge case behavior

**Script**: `scenario_e_edge_cases.py`

**Question**: Does the guard's API behave predictably — no exceptions, documented
behavior — at the extremes of its parameter space?

**Why**: Even if the guard works well on typical inputs, boundary conditions can reveal
implementation gaps: index errors in single-feature datasets, NaN propagation from
degenerate inputs, or silent no-ops when significance is below the minimum possible
p-value. Each case is designed around a specific code-path boundary.

| Case | Trigger | What is asserted |
|---|---|---|
| E1 | significance=0.9 on OOD instances | No crash; `intervals_emitted=0`; `plot()` and `get_rules()` run cleanly |
| E2 | significance=0.001, n_cal=200 | `n_removed_guard=0` — guard silently does nothing (design boundary, not a bug) |
| E3 | n_neighbors=1 | No crash; no NaN/inf in p-values |
| E4 | n_neighbors ≥ n_cal | No crash; k_actual saturates gracefully |
| E5 | merge_adjacent=True | Non-conforming bins not tagged is_merged=True (no bridging across OOD bins) |
| E6 | x.shape=(n, 1), single feature | No IndexError; exactly 1 feature in audit |
| E7 | All test instances identical | No NaN/inf in p-values |

**Key metric**: PASS/FAIL per case.

- *Expected range*: all cases PASS.
- *Red flag*: any unexpected FAIL. Note that E2 has expected behavior "guard silently
  does nothing" — this is the documented PASS state, revealing a design boundary that
  must appear in user-facing documentation.

---

## Running the suite

**Individual scenarios** (run from `evaluation/guarded/` so that `common_guarded`
is importable):

```bash
cd evaluation/guarded

# Quick smoke-test (a few minutes each)
python guarded_vs_standard_scenario_a.py --quick
python scenario_b_ood_detection_quality.py --quick
python scenario_c_regression.py --quick
python scenario_d_real_datasets.py --quick
python scenario_e_edge_cases.py --quick
```

**Full grid** (omit `--quick`; Scenario A and B take hours on CPU):

```bash
python scenario_b_ood_detection_quality.py   # 1–2 hours on CPU
python scenario_c_regression.py              # 15–30 min
python scenario_d_real_datasets.py           # 30–60 min
python scenario_e_edge_cases.py              # < 5 min
python guarded_vs_standard_scenario_a.py     # 2–6 hours on CPU
```

**Master runner** (all scenarios, quick mode):

```bash
python run_all_guarded.py --quick
python run_all_guarded.py --scenarios b,c,d,e --quick
python run_all_guarded.py --scenarios all
```

Artifacts are written to `evaluation/guarded/artifacts/`.

---

## Interpreting results

| Metric | Healthy | Red flag | Likely cause |
|---|---|---|---|
| `violation_rate`: guarded < standard | Yes | Guarded ≥ standard | Guard not filtering OOD perturbations |
| `auroc` | > 0.80 (moderate+ shift, n_dim ≤ 5) | < 0.60 (extreme shift) | Curse of dimensionality, or normalize_guard=False |
| `fpr_at_significance` | ≈ significance | >> significance | Calibration set too small, or data shift |
| `n_invariant_violations` | 0 | Any > 0 | Bug in regression code path |
| `audit_field_completeness` | True always | Any False | Multiclass payload shape bug |
| `fraction_instances_fully_filtered` | < 0.05 at α=0.10 | > 0.10 | Small n_cal, or Bonferroni overcorrection |
| Edge case PASS/FAIL | All PASS | Unexpected FAIL | API boundary not handled |

### The E2 design boundary

With n_cal calibration points, p-values are discrete with step 1/n_cal. When
`significance < 1/n_cal`, no p-value can ever fall below the threshold — the guard
silently does nothing. This is not a bug, but it must be documented and communicated to
users who set very small significance values.

Example: n_cal=200 → minimum p-value = 0.005. Setting significance=0.001 silently
disables the guard. The FPR at significance=0.001 will be exactly 0, which can
misleadingly appear as "perfect calibration."

### The Bonferroni interaction

With `use_bonferroni=True` and max_depth=3 (up to 8 bins per feature), each bin is
tested at `significance / 8`. At significance=0.10, the effective per-bin threshold
is 0.0125. On small datasets like iris (n_cal≈30, min p-value≈0.033), this means
no bin can ever be rejected under Bonferroni — another silent no-op. Scenario D
quantifies this effect via the `bonferroni_comparison.png` plot.
