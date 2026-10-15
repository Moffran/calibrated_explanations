> **Active scope:** Closure evidence for the batch-dependent thresholded-regression
> defect found through the CSE-004b-R diagnostic (CE 1.0.1). Historical record once merged.

# Thresholded regression batch dependence: root cause, fix, evidence

## 1. Summary

In CE 1.0.1, the calibrated probability `P(y <= t | x)` and its Venn-Abers interval for a
test object depended on the other rows in the same `predict_proba` / `predict` /
`explain_*` call: their number, order and content. Repeated identical calls were
deterministic, which hid the problem. Two objects that differ only in a feature the
learner ignores have identical raw predictions, but they could still receive different
calibrated outputs when they sat at different batch positions. That is the `x2` effect
CSE-004b-R isolated.

**Root cause.** The threshold score passed to Venn-Abers is crepes'
`ConformalPredictiveSystem.predict(y=t)`, which by default returns the *smoothed* CPD value

    Q(t, tau_i) = ( #{a : yhat_i + s_i a < t} + tau_i (#{a : yhat_i + s_i a = t} + 1) ) / (n + 1)

crepes generates `tau` as `np.random.seed(seed); gammas = np.random.rand(len(y_hat))`
on every call (`crepes/base.py`, CPS `predict`, `gammas[i]`). CE always passes a seed
(default 42), so `tau_i` is the i-th draw of a fixed sequence. It is a function of the
object's **row index**, not of the object. Because `Q` moves by up to `1/(n+1)` with
`tau`, and Venn-Abers maps the score through a step function, the calibrated output
jumps between isotonic steps as the row position changes.

A second, related defect: test scores came from `IntervalRegressor.cps`, a CPS fitted on
**all** calibration residuals, including the half whose labels Venn-Abers calibrates on.
The Venn-Abers calibration scores came from `split["cps"]`, fitted on the other half only.
Inductive Venn-Abers is valid only when one fixed score function is applied to both the
calibration and the test objects. Under smoothing, the two functions agree in expectation,
which masked the mismatch. Under a deterministic score they do not, as Section 5 shows.

**Fix** (`calibration/interval_regressor.py`). Both the calibration side
(`compute_proba_cal`) and the test side (`predict_proba`) now evaluate the **split CPS**
with `smoothing=False`. That makes the score `(#{yhat + s a <= t} + 1)/(n+1)`, a
deterministic function of `(yhat, sigma, bin)`, shared by both sides. This is the textbook
IVAP construction: the proper-training part fits the scorer, the calibration part fits the
isotonic regressions, and test objects are scored with the same scorer. No public API
changes.

## 2. Trace (first diverging value)

The values below are for one physical object, P05 threshold, Friedman-like synthetic data,
RF learner, 400 calibration objects. They come from `trace_stages.py`, instrumentation
only.

| Context | yhat | sigma | gamma (row draw) | CPS score | VA [p0, p1] | CE output |
|---|---|---|---|---|---|---|
| alone (row 0) | 3.164101 | 1.0 | 0.3745 | 0.000934 | [0, 0.00719] | 0.0071 |
| row 1 of 2 | 3.164101 | 1.0 | 0.9507 | 0.002371 | [0, 0.00971] | 0.0096 |
| row 5 of 16 | 3.164101 | 1.0 | 0.1560 | 0.000389 | [0, 0.00645] | 0.0064 |

`yhat` and `sigma` are identical in every context. The first diverging value is the CPS
score, and it reconstructs exactly as `(0 + (0 + 1) * gamma_row) / 401`. The gammas are the
seed-42 uniform draws at those row indices.

## 3. Hypotheses

| | Hypothesis | Evidence | Verdict |
|---|---|---|---|
| H1 | Venn-Abers couples test objects | `venn_abers.calc_probs` is an element-wise `searchsorted` on precomputed isotonic fits. Identical scores give identical outputs in any batch. | Rejected |
| H2 | Sorting or alignment mismatch | Raw predictions and sigmas align exactly per object, and singleton calls reproduce the trace. Monotonicity in `yhat` is restored after the fix. | Rejected |
| H3 | Shared state or caching | `compute_proba_cal` rebuilds the Venn-Abers calibrator per threshold from frozen arrays. Repeated calls give 0/1,344 differences. | Rejected (the state is reset, not leaked) |
| H4 | CPS batch-dependent before Venn-Abers | Row-indexed `gammas` in crepes smoothed p-values (Section 2). | **Confirmed root cause** |
| H5 | Why are tuple thresholds stable? | The tuple path calls crepes twice with the same seed and the same batch length, so object `i` gets the same `tau_i` in both calls. The `tau` terms cancel unless `yhat_i + s_i a` equals a bound exactly. Integer targets with a shallow tree reproduce tuple failures before the fix. | Explained |
| H6 | Numerical noise | Differences reach 0.18 (P95, California). They are deterministic, exactly reproducible, and explained by `tau`, not by floating point. | Rejected as noise |

Per-instance threshold arrays take a row-by-row loop, so every object effectively saw
`gammas[0]`. Scalar and array thresholds therefore also disagreed with each other.

## 4. Reproduction (public API only)

See `tests/integration/core/test_thresholded_regression_batch_invariance.py`. The learner
is `Pipeline(ColumnTransformer(keep x0, x1), estimator)`, so `x2` has no computational
path to the prediction. Before the fix, **45 tests failed**:

- scalar P05/P50/P95 batch, permutation, padding and halving contexts;
- `x2` twins;
- scalar vs per-instance threshold path;
- `predict` interval parity;
- monotonicity in `yhat`;
- factual explanation weights;
- tuple thresholds under exact ties.

All of them pass after the fix. A standalone script on synthetic data reproduced the CSE
pattern with the same signature as CSE: P05 16/24 and P95 12/24 same-batch twin
differences, within-spec 0/24, repeats 0. After the fix, all counts are 0.

## 5. Calibration quality (before vs after)

Thresholds are the q ∈ {0.05, 0.25, 0.5, 0.75, 0.95} quantiles of the training targets.
Each dataset has 5 seeds and a 50/25/25 split. Values are means over cells; *t* is the
paired t-statistic of after − 1.0.1. "A" is smoothing removal only; "B" is the shipped fix
(smoothing removal plus the shared split-CPS scorer).

**Unconditional, RF and Ridge (diabetes, Friedman-1, California; 150 cells)**

| Metric | 1.0.1 | A | B (shipped) | B − 1.0.1 (t) |
|---|---|---|---|---|
| Brier | 0.09561 | 0.09689 | 0.09508 | −0.0005 (−0.5) |
| Log loss | 0.30885 | 0.32067 | 0.31562 | +0.0068 (0.8) |
| ECE (10 bins) | 0.06412 | 0.06674 | 0.05963 | −0.0045 (−2.9) |
| Mean VA width | 0.06258 | 0.06741 | 0.06552 | +0.0030 (2.1) |
| Event rate inside mean [low, high] | 65% | 68% | 73% | |
| Max batch deviation | 0.183 | 0 | 0 | |

**Normalized (crepes DifficultyEstimator) and Mondrian (3 bins on yhat), RF (Friedman-1, California; 50 cells each)**

| Config | Metric | 1.0.1 | B | B − 1.0.1 (t) |
|---|---|---|---|---|
| Normalized | Brier | 0.05692 | 0.05673 | −0.0002 (−2.0) |
| Normalized | ECE | 0.03005 | 0.02893 | −0.0011 (−1.6) |
| Normalized | width | 0.03516 | 0.03587 | +0.0007 (1.6) |
| Mondrian | Brier | 0.05876 | 0.05995 | +0.0012 (0.6) |
| Mondrian | ECE | 0.03742 | 0.04123 | +0.0038 (0.9) |
| Mondrian | width | 0.05144 | 0.06039 | +0.0090 (1.1) |
| Both | max batch deviation | 0.030 | 0 | |

What the evaluation shows:

- Removing smoothing alone (A) restores invariance, but it degrades ECE and width
  significantly in the unconditional setting (ECE t = 2.7, width t = 4.5). That
  degradation is the scorer mismatch described in Section 1. B removes it.
- B shows no significant calibration loss anywhere. It is slightly better unconditionally
  and with normalization.
- Under Mondrian, B's Venn-Abers intervals are about 0.009 wider (not significant). The
  likely mechanism is that per-bin calibration sets are small, and the deterministic score
  produces ties that the random `tau` used to break, so intervals reflect the score's
  actual resolution.
- Individual outputs can move noticeably when the calibration set is small. On diabetes
  with 180 calibration objects (about 90 Venn-Abers points), single-object probabilities
  changed by 0.05 on average and by up to 0.22. Aggregate Brier and ECE did not worsen
  there.

## 6. Compatibility

- **Changed:** numeric outputs of every thresholded regression path (scalar and tuple
  thresholds; `predict`, `predict_proba`, factual, alternative and fast explanations,
  reject with regression thresholds). Thresholded factual feature weights in 1.0.1
  carried batch noise of up to 0.034 on diabetes; it is now 0.
- **Unchanged:** classification (Venn-Abers on learner probabilities, no crepes p-values),
  percentile regression intervals (`predict_uncertainty`, no `tau` term), and serialized
  state schema (`split_parts` and the residual arrays are unchanged; the split CPS is refit
  on load as before). A save/load round trip reproduces thresholded outputs exactly and
  stays batch-invariant; `test_restored_state_reproduces_instance_wise_outputs` pins this.
- **Online calibration** (`CalibratedExplainer.reinitialize(learner, xs, ys)` →
  `insert_calibration`) has two **pre-existing** defects that this change neither
  introduces nor repairs:
  - New rows are indexed `len(residual_cal) - 1 + i` instead of `len(residual_cal) + i`.
    After adding 5 rows to 100, index 99 appears twice in the split and index 104 is
    missing.
  - New residuals are inserted unnormalised into the CPS alphas, which is wrong when a
    difficulty estimator is active.

  As a result the split CPS no longer equals a refit on part 0. Before the fix, the same
  defects affected the full CPS used for test scores. Outputs after online calibration
  remain deterministic and instance-wise, but they are not a valid inductive Venn-Abers
  predictor until `insert_calibration` is repaired. This overlaps with the v1.0.2 task T7
  (#208), which concerns the interval-calibration update path.
- **Golden values:** no existing test pinned thresholded regression numbers. Before the
  fix, the full existing suite had one failure,
  `test_runtime_version_matches_installed_package_metadata`. It passes in isolation and is
  an order-dependent environment check, unrelated to this change.

## 7. Contract

`CE-REQ-PRED-PROB-INSTANCE-001` (claim `CE-CAP-PRED-PROB-001`, TIF `CE-TIF-PRED-PROB-002`):
for a frozen calibration state, the thresholded probability and both interval endpoints
are a function of the object's model input (and Mondrian bin) only. They are exactly equal
across batch sizes, positions and permutations. The contract holds for deterministic
learners; a learner that is itself batch-dependent is outside CE's control.

Semantics, so CSE does not need its own calibration logic: CE's probabilistic regression is
an **instance-wise inductive Venn-Abers predictor**, not a transductive one. The smoothed
CPS was randomized, not transductive; position-indexed seeding made its randomness look
like batch coupling.

## 8. Out of scope, recorded separately

- **Mondrian bins missing from one calibration half** (pre-existing; relevant to
  ADR-039). The split is a seeded permutation that ignores bins, so a small bin can fall
  entirely into one half:
  - *Bin only in the CPS half (part 0):* no Venn-Abers calibrator exists for the bin, and
    the point estimate and both bounds come out as silent `0`.
  - *Bin only in the Venn-Abers half (part 1):* the split CPS has no alphas for the bin,
    and crepes silently returns score `0` on both the calibration side and, after this
    fix, the test side. The output is an uninformative but internally consistent
    Venn-Abers interval (for example `[0.5, 1.0]` from one calibration object). Before
    the fix, a real test score was compared against the silent `0` calibration scores.

  Neither case warns. ADR-039 forbids implicitly substituting an output for a bin that
  has no fitted calibrator. The remedy is a bin-stratified split or an explicit
  `ValidationError`/`UserWarning`; it needs its own issue and decision.
- `insert_calibration` index and normalisation defects (Section 6).
- The open one-sided percentile intervals that use `min(y_cal)` / `max(y_cal)` as
  placeholder bounds (`predict_uncertainty`) are a different defect and are untouched here.
- `CE-TIF-PRED-PROB-001` and `CE-CAP-PRED-PROB-001` describe the event as `P(Y > t)`,
  but the API returns `P(y <= t)` (documentation inconsistency).
- crepes 0.9.1 `MondrianCategorizer.apply` references an undefined `learner` (should be
  `self.learner`) when fitted with `learner=`. This is an upstream bug; the evaluation
  uses `f=learner.predict` instead.

## 9. ADR alignment

| ADR | Assessment |
|---|---|
| ADR-001 | No new imports; the change is confined to `calibration/interval_regressor.py`. |
| ADR-002 | No new error paths. |
| ADR-011 / ADR-020 | No public signature changes; the legacy API contract is unaffected. The numeric change is a correctness fix (the ADR-039 precedent: superseded behaviour is a defect, not supported API), so no deprecation cycle applies. Recorded in CHANGELOG. |
| ADR-013 | `IntervalRegressor` remains the only thresholded backbone; no plugin implements a separate path. |
| ADR-021 | §3 amended (2026-10-15): split-CPS scorer shared by both sides, `smoothing=False`, instance-wise contract. Both CPS and Venn-Abers are retained. |
| ADR-028 | No new logging or fallbacks. |
| ADR-029 | Regression reject scores change numerically through `predict_probability`; there is no contract change. |
| ADR-030 | Public-API tests; the ratification lane passes (0 private-member violations, no marker or anti-pattern regressions). |
| ADR-031 | Schema unchanged; the round trip is exact (tested). |
| ADR-039 | Bins are forwarded unchanged. The pre-existing half-coverage gap is recorded in Section 8. |
| ADR-040 | New requirement `CE-REQ-PRED-PROB-INSTANCE-001`, new TIF `CE-TIF-PRED-PROB-002`, raw evidence, and updated claim assumptions; the existing TIF's evidence history is untouched. The chain check reports 0 errors. |

## 10. Not yet done

The CSE validation (re-running CSE-004b-R on the frozen FULL archive, then CSE-004b FULL)
was not run from this change: the CSE environment was not available. It must be run
against this branch, with new run IDs and the original runs preserved, before the defect
is declared resolved for CSE.
