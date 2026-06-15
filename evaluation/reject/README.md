# Reject Evaluation Suite

This directory evaluates the repository's implemented reject integration against
the real `WrapCalibratedExplainer` runtime APIs described by ADR-029.

Scenarios 1-7 map to a paper contribution (C1-C4) or research question (RQ1-RQ6) from the
accompanying ESWA paper. Scenario 8 is a repository-focused empirical ablation that measures the
current indirect effect of `difficulty_estimator` on reject behavior without changing reject
scoring. Scenario 9 extends this with a direct strategy comparison for experimental
difficulty-normalized reject scoring. Scenario 10 evaluates the second experimental
ambiguity-normalized novelty-penalized strategy. Scenario 11 evaluates matched
reject-rate operating points before any public API promotion decision.

All scenario result tables include singleton precision/recall diagnostics when a
prediction set and compatible empirical label are available. Precision is the
correct-singleton fraction among singleton outputs; recall is the fraction of all
labelled rows resolved as correct singletons.

## Scenarios

### Core scenarios (run by default)

- **Scenario 1 — Binary marginal coverage** (`scenario_1_binary_coverage.py`, RQ1):
  Multi-dataset test of marginal coverage preservation across 26 binary datasets at
  epsilon in {0.01, 0.05, 0.10} with Clopper-Pearson CIs.  Directly tests C1 (Proposition 1).
  Distinguishes finite-sample coverage shortfalls from structural violations (CI upper bound
  below 1-epsilon).  Status: `formal_target`.

- **Scenario 2 — Multiclass correctness proxy** (`scenario_2_multiclass_correctness.py`, RQ2):
  Empirical evaluation of CE multiclass reject as a binary correctness proxy using
  `hinge` and `ensured` NCFs across 20 multiclass datasets. Tests C2 pragmatically,
  but not as a K-class conformal label-set guarantee. A `{1}` singleton accepts the
  top-1 prediction under the proxy; a `{0}` singleton is a proxy-negative aggregate
  over all non-top1 classes, not a selected alternative class. Primary empirical
  accuracy is computed in proxy space by comparing singleton `{0}/{1}` outcomes with
  `1[top-1 prediction is correct]`; accepted top-1 accuracy is only a diagnostic on
  `{1}` rows. Status: `empirical`.

- **Scenario 3 - Thresholded regression binary-event reject validity** (`scenario_3_regression_threshold_baseline.py`, RQ3):
  Multi-dataset empirical analysis of thresholded regression reject as binary conformal
  classification over user-defined regression events. Scalar thresholds use `y <= threshold`;
  interval thresholds use `low < y <= high`. The scenario reports event-label coverage,
  empty/singleton/ambiguity counts, novelty/ambiguity/reject rates, and singleton
  precision/recall/error. It is not an interval-width selector and does not evaluate
  conformal prediction interval regression. Status: `empirical`.

- **Scenario 4 — NCF and blend weight grid** (`scenario_4_ncf_weight_grid.py`, RQ4):
  Grid of `hinge`, `margin`, `ensured` × w in {0.3, 0.5, 0.7, 1.0} across binary and
  multiclass datasets.  Tests C2 NCF selection guidance.  Demonstrates that margin at w=0.3
  collapses to near-zero accept rates and that w >= 0.7 converges NCFs to hinge-like behavior.
  Column `accept_rate` is the fraction accepted — NOT ICP label-set coverage.
  Status: `empirical`.

- **Scenario 5 — Explanation quality on accepted instances** (`scenario_5_explanation_quality.py`, RQ5):
  Accuracy delta and ECE delta between all-instance and accepted-instance subsets, segmented by
  reject-rate regime (low ≤15%, moderate 15–40%, high >40%).  Tests C4.
  Status: `empirical`.

- **Scenario 6 — Finite-sample stress tests** (`scenario_6_finite_sample_stress.py`, RQ6):
  Coverage violations at small n_cal in {10, 20, 50, 100, 200} and extreme confidence
  (epsilon in {0.005, 0.01}) across binary datasets.  Violation is computed from actual
  coverage on both probes.  Status: `empirical`.

- **Scenario 8 — Difficulty estimator reject ablation** (`scenario_8_difficulty_reject_ablation.py`):
  Empirical comparison of four classification arms:
  1. `difficulty_estimator=None`, reject NCF `default`
  2. `difficulty_estimator=...`, reject NCF `default`
  3. `difficulty_estimator=None`, reject NCF `ensured`
  4. `difficulty_estimator=...`, reject NCF `ensured`
  It measures whether the existing path `difficulty_estimator -> VennAbers probability scaling ->
  reject NCF -> ConformalClassifier` already changes accept/reject behavior before any
  difficulty-normalized reject NCF is added. Status: `empirical`.

- **Scenario 9 — Difficulty-normalized reject NCF strategy ablation** (`scenario_9_difficulty_normalized_ncf.py`):
  Empirical six-arm comparison between the current indirect difficulty path and the new
  experimental direct score-normalization strategy:
  1. A: no VA difficulty, `builtin.default`, `ncf=default`
  2. B: VA difficulty, `builtin.default`, `ncf=default`
  3. C: no VA difficulty, `experimental.difficulty_normalized`, `ncf=default`
  4. D: VA difficulty, `experimental.difficulty_normalized`, `ncf=default`
  5. E: no VA difficulty, `experimental.difficulty_normalized`, `ncf=ensured`
  6. F: VA difficulty, `experimental.difficulty_normalized`, `ncf=ensured`
  Primary scientific contrast is A vs C. D and F are diagnostic for potential
  difficulty double-counting when both VA difficulty and direct score normalization are enabled.
  Status: `empirical`.

- **Scenario 10 - Ambiguity-normalized novelty-penalized reject strategy** (`scenario_10_ambiguity_novelty_reject.py`):
  Empirical comparison of three classification arms:
  1. A: `builtin.default`, `ncf=default`
  2. C: `experimental.difficulty_normalized`, `ncf=default`
  3. G: `experimental.ambiguity_normalized_novelty_penalized`, `ncf=default`
  Primary scientific contrast is C vs G. The scenario tests whether a separate
  novelty penalty can shift some difficult cases from ambiguous multi-label sets
  toward novelty empty sets without losing the accepted-accuracy benefit observed
  in Scenario 9. Status: `empirical`.

- **Scenario 11 - Matched operating-point reject selection** (`scenario_11_operating_point_selection.py`):
  Matched reject-rate operating-point comparison at target reject rates
  `{0.10, 0.20, 0.30, 0.40}`. The primary comparison is A vs C:
  `builtin.default` against `experimental.difficulty_normalized`, both with
  `ncf=default` and no VA difficulty. The secondary comparison is C vs G:
  direct difficulty normalization against the novelty-aware variant. This is an
  operating-guidance experiment and does not promote public API. Status:
  `empirical`.

### Supplementary scenarios (pass `--supplementary` flag)

- **Scenario 7 — NCF coverage validity sweep** (`scenario_7_ncf_coverage_validity.py`):
  Empirical companion to Proposition 1. Measures coverage at epsilon in {0.05, 0.10}
  across the full (NCF, w) grid on binary datasets and 5 seeds. The scenario reports
  row-level and collapsed-by-condition violations separately because `default` ignores `w`.
  Separates coverage validity diagnostics from accuracy analysis in Scenario 4.

- **Scenario 12 — Coverage validity: arm A vs arm C** (`scenario_12_coverage_validity_difficulty_normalized.py`):
  RT-3 red-team obligation. Mirrors Scenario 1 but runs arm A (`builtin.default`) and arm C
  (`experimental.difficulty_normalized`) side by side across binary datasets at epsilon in
  {0.05, 0.10}. Difficulty normalization changes the nonconformity score definition, so coverage
  validity must be verified empirically for arm C separately. Structural violations (CI upper bound
  below 1-epsilon) are flagged per arm. Status: `empirical`.

- **Scenario 13 — n_cal sweep: arm A vs arm C structural violations** (`scenario_13_ncal_coverage_sweep.py`):
  RT-3 follow-up. Sweeps calibration set size n_cal ∈ {50, 100, 200, 400} to test the
  variance-inflation hypothesis for arm C structural violations observed in Scenario 12. If
  difficulty normalization inflates calibration score variance (a finite-sample effect), arm C
  structural violation rates should decrease monotonically as n_cal grows and converge toward arm A
  rates at large n_cal. If rates do not decrease, a genuine exchangeability violation is indicated
  and arm C requires a redesign before any public promotion. Status: `empirical`.

- **Scenario 14 — Routing policy contract validation** (`scenario_14_routing_policy_contract.py`):
  Validates seven routing contract invariants for FLAG / ONLY_ACCEPTED / ONLY_REJECTED policies
  across all binary datasets. The red-team analysis (Bug 1) showed that an incorrect
  `prediction_set` access path produced vacuously-true "0 violations" in Scenario 7 — a routing
  contract bug that silently contaminated a formal validity measurement. Invariants tested: FLAG
  rejected mask shape; FLAG prediction_set accessible via `result.metadata["prediction_set"]`; FLAG
  original_count; ONLY_ACCEPTED source_indices cardinality; ONLY_REJECTED source_indices
  cardinality; disjoint union covers all n_test instances; no degraded_mode markers on healthy
  data. Status: `contract`.

### What this suite does NOT measure

- ICP monotonicity as a standalone scenario — implementation invariant for unit tests.
- Confidence sweep on a single binary dataset — absorbed by Scenario 1 full mode.
- The current indirect effect of `difficulty_estimator` on classification reject — measured by
  Scenario 8.
- **Difficulty-normalised regression reject (C3)** — blocked by RT-2 sigma-normalisation fix.
  Scenario 3 now validates the ordinary thresholded-regression binary event contract. C3 still
  requires a calibration path that normalises sigma without changing the event probability
  calibration used by other features. Unblock condition: RT-2 merged and validated.
- **Fallback-mode coverage validity** — the orchestrator has three fallback paths
  (`predict_p_to_predict_set_fallback`, `bulk_to_per_instance_fallback`) recorded in
  `degraded_mode_markers`. Coverage under these fallback paths is not validated by any current
  scenario. Scenario 14 (I7 invariant) verifies that healthy data does not trigger fallbacks; a
  separate obligation exists to verify coverage still holds when fallbacks ARE triggered. Unblock
  condition: create a scenario that injects controlled failures (e.g. a `ConformalClassifier`
  subclass with a broken `predict_p`) and measures coverage on the fallback path.

## Artifact layout

Each scenario writes a bundle under `evaluation/reject/artifacts/`:

- `<scenario>.csv` — row-level metrics.
- `<scenario>.json` — machine-readable metadata and top findings.
- `<scenario>.md` — human-readable scenario summary.
- `summary.md` — cross-scenario outcome summary regenerated by the runner.

- `reject_evaluation_suite_report.md` - manually curated synthesis report
  with method, results, discussion, and conclusion sections across all scenarios.

## Run the suite

Quick smoke run (core scenarios):

```pwsh
python -m evaluation.reject.run_all_reject --quick
```

Full run (core scenarios):

```pwsh
python -m evaluation.reject.run_all_reject --full
```

Include supplementary scenarios:

```pwsh
python -m evaluation.reject.run_all_reject --quick --supplementary
```

Regenerate only the top-level summary:

```pwsh
python -m evaluation.reject.summarize_results
```

Run an individual scenario directly:

```pwsh
python -m evaluation.reject.scenario_1_binary_coverage --quick
python -m evaluation.reject.scenario_2_multiclass_correctness --quick
python -m evaluation.reject.scenario_8_difficulty_reject_ablation --quick
python -m evaluation.reject.scenario_9_difficulty_normalized_ncf --quick
python -m evaluation.reject.scenario_10_ambiguity_novelty_reject --quick
python -m evaluation.reject.scenario_11_operating_point_selection --quick
python -m evaluation.reject.scenario_13_ncal_coverage_sweep --quick
python -m evaluation.reject.scenario_14_routing_policy_contract --quick
```

## Interpretation notes

- Higher **accepted accuracy** with tolerable coverage loss indicates value from
  the reject gate.
- Lower **ECE** on the accepted subset indicates better-calibrated decisions
  after rejection.
- For **Scenario 4**, the `accept_rate` column is fraction of test instances
  accepted — not ICP label-set coverage.  Do not confuse the two.
- For **Scenario 3**, coverage and singleton diagnostics are computed against derived binary
  event labels, not against regression interval coverage or interval width.
- For **Scenario 8**, any observed difference comes from the existing interval-calibration path;
  reject scoring formulas themselves remain unchanged.
- For **Scenario 9**, compare A vs C first (cleanest contrast). Treat D/F as diagnostic because
  they may double-count difficulty (VA scaling + direct score normalization).
- For **Scenario 10**, compare C vs G first. G is still experimental and uses an
  evaluation-only novelty estimator; it should not be interpreted as a public API recommendation.
- For **Scenario 11**, interpret deltas at matched target reject rates. This is the
  decision-gate evidence before any `ncf="difficulty_normalized"` public API promotion.

## Latest Scenario 8-11 snapshot

For the latest run artifacts, read the generated markdown reports first:

- `evaluation/reject/artifacts/scenario_8_difficulty_reject_ablation.md`
- `evaluation/reject/artifacts/scenario_9_difficulty_normalized_ncf.md`
- `evaluation/reject/artifacts/scenario_10_ambiguity_novelty_reject.md`
- `evaluation/reject/artifacts/scenario_11_operating_point_selection.md`

Current artifact-level recommendation trend:

- Scenario 8: indirect VA difficulty path is stricter and captures more errors,
  with an accepted-accuracy cost in this setup.
- Scenario 9: arm C (`experimental.difficulty_normalized`, no VA difficulty)
  is the preferred experimental baseline from the primary A-vs-C contrast.
- Scenario 10: novelty-aware arm G remains exploratory; C is still the simpler
  recommended baseline.
- Scenario 11: matched operating-point evidence is mixed; difficulty-normalized
  scoring should not be promoted to a public NCF yet, and the novelty-aware
  strategy remains internal/experimental.

## Design constraints followed

- Uses `WrapCalibratedExplainer` and CE-first helper utilities instead of
  ad-hoc wrappers.
- Keeps evaluation code under `evaluation/` to respect ADR-010's runtime/eval
  split.
- Uses the implemented `RejectPolicy`, `RejectPolicySpec`, and reject envelope
  contracts instead of synthetic placeholders.
- Public NCF set exercised by this suite: `default`, `ensured`. Entropy remains excluded.
