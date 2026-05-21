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

## Scenarios

### Core scenarios (run by default)

- **Scenario 1 — Binary marginal coverage** (`scenario_1_binary_coverage.py`, RQ1):
  Multi-dataset test of marginal coverage preservation across 26 binary datasets at
  epsilon in {0.01, 0.05, 0.10} with Clopper-Pearson CIs.  Directly tests C1 (Proposition 1).
  Distinguishes finite-sample coverage shortfalls from structural violations (CI upper bound
  below 1-epsilon).  Status: `formal_target`.

- **Scenario 2 — Multiclass correctness classifier** (`scenario_2_multiclass_correctness.py`, RQ2):
  Empirical evaluation of CE multiclass reject as a conformal correctness classifier using
  `hinge`, `margin`, and `ensured` NCFs across 20 multiclass datasets.  Tests C2.  Demonstrates
  hinge collapse on small-n-class datasets (flagged via `expected_collapse`) and margin
  selectivity.  Status: `empirical`.

- **Scenario 3 — Threshold regression heuristic baseline** (`scenario_3_regression_threshold_baseline.py`, RQ3):
  Multi-dataset empirical analysis of threshold-based regression reject.  Establishes the null
  result: threshold reject does not select by uncertainty — accepted-subset interval width equals
  full-set interval width (the threshold rejects by predicted value quantile, not interval
  width).  Status: `empirical`.

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

### Supplementary scenarios (pass `--supplementary` flag, requires RT-2 fix)

- **Scenario 7 — NCF coverage validity sweep** (`scenario_7_ncf_coverage_validity.py`):
  Empirical companion to Proposition 1.  Verifies coverage ≥ 1-epsilon at epsilon in {0.05, 0.10}
  across the full (NCF, w) grid on binary datasets.  Separates coverage validity check from
  accuracy analysis in Scenario 4.

### What this suite does NOT measure

- API routing behavior (FLAG vs ONLY_ACCEPTED vs ONLY_REJECTED) — CI integration concern only.
- ICP monotonicity as a standalone scenario — implementation invariant for unit tests.
- Confidence sweep on a single binary dataset — absorbed by Scenario 1 full mode.
- The current indirect effect of `difficulty_estimator` on classification reject — measured by
  Scenario 8.
- Difficulty-normalised regression reject (C3) — deferred to a standalone scenario pending the
  RT-2 sigma-normalisation-only calibration fix.

## Artifact layout

Each scenario writes a bundle under `evaluation/reject/artifacts/`:

- `<scenario>.csv` — row-level metrics.
- `<scenario>.json` — machine-readable metadata and top findings.
- `<scenario>.md` — human-readable scenario summary.
- `summary.md` — cross-scenario outcome summary regenerated by the runner.

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
```

## Interpretation notes

- Higher **accepted accuracy** with tolerable coverage loss indicates value from
  the reject gate.
- Lower **ECE** on the accepted subset indicates better-calibrated decisions
  after rejection.
- For **Scenario 4**, the `accept_rate` column is fraction of test instances
  accepted — not ICP label-set coverage.  Do not confuse the two.
- For **Scenario 3**, the `interval_width_delta` near zero is the expected null
  result: threshold reject does not select by uncertainty.
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
