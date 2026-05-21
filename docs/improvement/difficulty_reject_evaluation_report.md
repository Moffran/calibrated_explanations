# Difficulty-Aware Reject Evaluation Report

Date: 2026-05-21

## Scope

This report interprets the difficulty-aware reject-option evaluation artifacts
from Scenarios 8, 9, 10, and the full Scenario 11 run. It gives guidance for:

- Step 8: promote a successful experimental strategy to public API.
- Step 9: write practitioner and research documentation.
- Ongoing reject-framework design around difficulty-aware scoring.

The relevant artifacts are:

- `evaluation/reject/artifacts/scenario_8_difficulty_reject_ablation.*`
- `evaluation/reject/artifacts/scenario_9_difficulty_normalized_ncf.*`
- `evaluation/reject/artifacts/scenario_10_ambiguity_novelty_reject.*`
- `evaluation/reject/artifacts/scenario_11_operating_point_selection.*`

## Executive Decision

Do not run Step 8 yet.

The full Scenario 11 run is now the strongest decision-gate evidence. It covers
`46` datasets, `5` seeds, `21,390` confidence-sweep rows, `2,760` selected
operating-point rows, and `1,840` paired delta rows. Its recommendation remains:

- promotion recommendation: `do_not_promote`
- novelty strategy recommendation: `continue_experimental`

For the primary A-vs-C comparison, direct difficulty-normalized scoring gives
only a tiny accepted-accuracy gain at the lowest nominal target reject rate and
negative accepted-accuracy deltas at the higher targets:

- target `0.10`: `+0.0012`
- target `0.20`: `-0.0029`
- target `0.30`: `-0.0070`
- target `0.40`: `-0.0089`

The effect is not statistically or operationally stable enough for public API
promotion. The fraction of finite dataset/seed groups with positive accepted
accuracy declines from `44.6%` at target `0.10` to `27.7%` at target `0.40`.

The conclusion is now sharper than after the quick run:

1. The existing Venn-Abers difficulty path should not be treated as the public
   answer.
2. `experimental.difficulty_normalized` remains the lead experimental design,
   but not a supported public API.
3. The novelty-aware variant shows some novelty-selectivity signal in Scenario
   11, but accepted accuracy and error-capture behavior do not justify
   promotion.
4. The reject framework needs operating-point diagnostics before any public
   difficulty-aware reject mode is exposed.
5. Step 9 may proceed only as interim documentation, with Step 8 explicitly
   deferred.

## Evidence Summary

### Scenario 8: Existing VA Difficulty Path

Scenario 8 measured the existing indirect path:

`difficulty_estimator -> VennAbers probability scaling -> reject NCF -> ConformalClassifier`

It did not change reject scoring. The result is a warning against relying on the
current indirect difficulty path as the public answer.

For `ncf="default"`:

- accept rate changed by `-40.6 pp`
- rejected-error capture changed by `+24.8 pp`
- accepted accuracy changed by `-44.4 pp`
- empirical coverage changed by `+8.1 pp`

For `ncf="ensured"`:

- accept rate changed by `-26.6 pp`
- rejected-error capture changed by `+18.5 pp`
- accepted accuracy changed by `-44.8 pp`
- empirical coverage changed by `+7.4 pp`

Interpretation: VA difficulty makes reject behavior much stricter, but not in a
useful accepted-subset way. It captures more errors because it rejects far more
instances, not because it produces a better operating point.

### Scenario 9: Direct Difficulty-Normalized NCF Grid Sweep

Scenario 9 compared the current path against direct reject-score normalization.
The clean comparison is A vs C:

- A: no VA difficulty, `builtin.default`, `ncf="default"`
- C: no VA difficulty, `experimental.difficulty_normalized`, `ncf="default"`

Aggregate A vs C:

- reject rate changed by `+23.1 pp`
- difficulty gap, rejected minus accepted, changed by `+0.3912`
- difficulty reject AUC changed by `+0.2474`
- ambiguity rate changed by `+27.9 pp`
- novelty rate changed by `-4.8 pp`
- matched-bin accepted accuracy changed by `+6.9 pp`

Robustness check across dataset/seed groups:

- C increased reject rate in `56.1%` of dataset/seed groups.
- C increased accepted accuracy in `27.4%` of groups with finite accepted accuracy.
- C increased empirical coverage in `56.5%` of groups with defined coverage.
- C increased ambiguity rate in `100%` of groups.
- C decreased novelty rate in `100%` of groups.

Interpretation: direct normalization reliably changed reject geometry in the
grid sweep. It made the strategy look promising enough to justify a matched
operating-point experiment, but the grid-sweep result alone was never sufficient
for public API promotion.

### Scenario 10: Novelty-Aware Variant Grid Sweep

Scenario 10 compared:

- A: `builtin.default`, `ncf="default"`
- C: `experimental.difficulty_normalized`, `ncf="default"`
- G: `experimental.ambiguity_normalized_novelty_penalized`, `ncf="default"`

The primary comparison is C vs G.

Aggregate G vs C:

- novelty rate changed by `+0.0019`
- empty-set rate changed by `+0.0019`
- ambiguity rate changed by `+0.0047`
- multi-label rate changed by `+0.0047`
- accepted accuracy changed by `+0.0037`
- novelty reject AUC changed by `-0.0371`

Robustness check across dataset/seed groups:

- G increased novelty rate in `49.1%` of groups.
- G increased ambiguity rate in `43.0%` of groups.
- G increased accepted accuracy in `29.6%` of groups with finite accepted accuracy.
- G increased novelty reject AUC in `14.8%` of groups.
- G increased reject rate in `27.0%` of groups.

Interpretation: the novelty-aware variant did not solve the ambiguity-vs-novelty
separation problem in the grid sweep. It remained useful as an internal research
branch, but not as a promotion candidate.

### Scenario 11: Full Matched Operating-Point Selection

Scenario 11 compared strategies at nominal target reject rates `0.10`, `0.20`,
`0.30`, and `0.40`. For each dataset/seed/arm, it swept confidence values and
selected the row whose observed reject rate was closest to the target.

Primary comparison:

- A: `builtin.default`, `ncf="default"`
- C: `experimental.difficulty_normalized`, `ncf="default"`

Secondary comparison:

- C: `experimental.difficulty_normalized`, `ncf="default"`
- G: `experimental.ambiguity_normalized_novelty_penalized`, `ncf="default"`

#### A vs C

Accepted-accuracy deltas, C minus A:

- target `0.10`: mean `+0.0012`, fraction positive `0.446`
- target `0.20`: mean `-0.0029`, fraction positive `0.346`
- target `0.30`: mean `-0.0070`, fraction positive `0.308`
- target `0.40`: mean `-0.0089`, fraction positive `0.277`

Other A-vs-C findings:

- empirical coverage improved at targets `0.10`, `0.20`, and `0.40`, but not at
  `0.30`;
- rejected-error capture was positive only at target `0.10`;
- difficulty-reject AUC improved at targets `0.10` and `0.40`, but worsened at
  `0.20` and `0.30`;
- difficulty gap improved at `0.10`, `0.20`, and `0.40`, but worsened at `0.30`;
- ambiguity did not consistently increase;
- novelty did not consistently decrease.

This weakens the Scenario 9 story. Direct normalization still changes the reject
surface, but it does not consistently produce better accepted subsets at matched
operating points.

#### C vs G

Novelty-aware scoring has a clearer routing effect in the full operating-point
run than in Scenario 10, but not enough for promotion.

G minus C:

- novelty-rate mean delta across targets: `+0.0084`
- empty-set-rate mean delta across targets: `+0.0084`
- novelty-reject-AUC mean delta across targets: `+0.0845`
- ambiguity-rate mean delta across targets: `-0.0114`
- accepted-accuracy mean delta across targets: `-0.0005`

By target:

- target `0.10`: accepted accuracy `-0.0033`, novelty AUC `+0.0568`
- target `0.20`: accepted accuracy `+0.0000`, novelty AUC `+0.0719`
- target `0.30`: accepted accuracy `+0.0008`, novelty AUC `+0.1228`
- target `0.40`: accepted accuracy `+0.0005`, novelty AUC `+0.0866`

Interpretation: G can increase novelty selectivity relative to C, and often
does so by increasing empty sets while reducing multi-label ambiguity. That is a
useful research signal. It is still not a public API signal because accepted
accuracy is essentially flat, rejected-error capture is negative at every
target, and the strategy is more complex.

#### Operating-Point Attainability

Scenario 11 also exposes a framework issue: nominal target reject rates are not
always attainable with the current confidence sweep. Mean target-distance values
are large, especially for the lower targets:

- A at target `0.10`: mean target error `0.4072`
- C at target `0.10`: mean target error `0.4116`
- A at target `0.40`: mean target error `0.2816`
- C at target `0.40`: mean target error `0.2695`

The selected operating points often have much higher observed reject rates than
the nominal target. The experiment is still valuable because both arms are
matched by the same selection rule, but public operating guidance would need
attainability diagnostics and warnings.

## Overall Pattern

The evaluations now show a consistent hierarchy of evidence.

First, difficulty can strongly affect reject behavior. It is not a cosmetic
metadata field. When difficulty enters probabilities or scores, it changes
prediction-set geometry and therefore changes accept, ambiguity reject, and
novelty reject rates.

Second, where difficulty enters matters. The existing VA-difficulty path affects
probabilities before reject scoring and behaves like an aggressive reject gate.
Direct score normalization is more targeted because it changes conformal reject
nonconformity scores before p-values and prediction sets.

Third, grid-sweep improvements do not necessarily survive operating-point
matching. Scenario 9 found a promising geometry shift and a positive matched-bin
accepted-accuracy signal. Scenario 11 asks the more practical question: what
happens when a user tries to operate around a desired reject rate? The answer is
mixed and mostly not favorable for public promotion.

Fourth, novelty separation is plausible but immature. Scenario 11 shows that the
novelty-aware variant can increase novelty AUC and empty-set routing relative to
C. It does not improve accepted accuracy in a decision-grade way and it reduces
rejected-error capture.

## What Actually Improved

The strongest improvement is architectural:

Direct difficulty-normalized scoring is a better experimental design than
indirect VA difficulty scaling.

It expresses the intended score transformation directly:

`alpha_new(x, y) = alpha_base(x, y) / difficulty(x)`

That is the right part of the pipeline to modify if the difficulty estimator is
fixed before calibration alphas are computed. It is easier to audit than a
post-hoc reject threshold and easier to compare against `builtin.default`.

Empirically, however, the full Scenario 11 result does not support promotion.
The only positive accepted-accuracy operating point is target `0.10`, and the
mean gain there is only `+0.0012`. The higher targets are negative and the
fraction-positive rates are below `0.5` at all targets.

For G, what improved is novelty routing, not accepted-subset quality. That is a
useful research direction, but not a user-facing feature yet.

## What Is Noise Or Not Yet Convincing

The following should not be used as promotion arguments:

- Scenario 8's higher error capture, because it is coupled to severe
  over-rejection and lower accepted accuracy.
- Scenario 9's grid-average geometry shift alone, because Scenario 11 weakens
  the practical operating-point case.
- Scenario 10's tiny accepted-accuracy increase for G, because novelty
  selectivity worsened in that grid sweep.
- Scenario 11's small A-vs-C gain at target `0.10`, because it is tiny and
  positive in fewer than half of finite groups.
- Scenario 11's C-vs-G novelty AUC gain, because it comes without a meaningful
  accepted-accuracy gain and with worse rejected-error capture.

The most important unresolved issue is operating-point controllability. If the
framework cannot get close to desired reject rates for many datasets, then a
public difficulty-aware strategy needs more than a new NCF name. It needs
diagnostics for attainable reject-rate regimes.

## Implications For The Reject Framework

### 1. Keep Difficulty As Experimental Score Logic

Difficulty should remain in the experimental strategy registry. The framework
should not add public `ncf="difficulty_normalized"` validation yet.

The internal strategy layer is still the right place because it allows:

- score-level changes without expanding `RejectPolicySpec`;
- metadata and audit fields that identify experimental behavior;
- side-by-side evaluation against `builtin.default`;
- rollback without public API migration.

### 2. Preserve Built-In Default Behavior

`builtin.default` remains the supported baseline. Difficulty experiments should
not change default reject behavior, default NCF validation, or policy
serialization.

Difficulty-aware scoring changes prediction sets directly. It is not a benign
post-processing step. Public users should not receive this behavior unless they
opt into a future supported API with clear documentation.

### 3. Treat Difficulty As Part Of The Conformal Score Pipeline

The framework should continue applying difficulty before conformal p-values and
prediction sets are computed. Post-hoc thresholding would not recalibrate the
score distribution and would make the semantics harder to defend.

The key invariant remains:

`calibration alphas` and `test alphas` must be transformed by the same fixed
difficulty scoring rule.

The Step 6/7 implementation direction is therefore correct, but the public API
gate remains closed.

### 4. Keep Provenance Metadata And Warnings

Difficulty estimators used for conformal scoring must be fixed before
calibration alphas are computed. The provenance checks protect the validity
story.

The framework should keep:

- permissive default behavior for third-party estimators;
- strict opt-in validation;
- metadata indicating provenance availability and warning status;
- warnings for calibration-label or residual leakage without cross-fitting.

### 5. Separate Ambiguity And Novelty Carefully

Scenario 11 gives the novelty-aware variant a reason to keep existing as an
internal experiment: it improves novelty AUC relative to C at matched operating
points. But it should not be public because it does not improve accepted
accuracy or error capture.

If pursued further, the framework likely needs:

- better novelty-estimator design;
- target-rate matched evaluation on attainable operating points;
- explicit documentation of when empty sets are desirable;
- no promise that novelty penalty improves accepted accuracy.

### 6. Add Operating-Point Diagnostics Before API Design

Scenario 11 shows that target-distance diagnostics are central. A future public
API should not merely expose a new NCF string; users will need to know whether a
desired reject rate is attainable.

Future framework support may need:

- helper diagnostics for attainable reject-rate ranges;
- metadata for selected confidence and target-distance;
- examples that tune reject operating points rather than using one fixed
  confidence blindly;
- warnings when observed reject rates are far from requested targets.

## Guidance For Step 8

Do not promote.

Step 8 should be deferred. Full Scenario 11 does not show strong enough
accepted-accuracy, difficulty-selectivity, or operating-point controllability to
justify public API exposure.

If Step 8 is eventually revisited, promote only the direct score-normalized
strategy. Do not promote the novelty-aware variant.

Candidate public API when ready:

```python
RejectPolicySpec.flag(ncf="difficulty_normalized", w=...)
```

This remains a candidate shape only. It should not be implemented now.

Promotion gates before Step 8:

1. A matched operating-point run on attainable targets must show positive
   accepted-accuracy deltas in a clear operating regime.
2. Difficulty-reject AUC or difficulty gap must improve consistently at those
   attainable matched targets.
3. Target-distance diagnostics must be part of the evidence, not an afterthought.
4. Estimator-quality sensitivity must show that the result is not driven by a
   brittle deterministic estimator.
5. Documentation must distinguish:
   - VA difficulty probability scaling;
   - direct reject-score normalization;
   - post-hoc thresholding, which is not the same thing.
6. Provenance safeguards must remain in metadata and tests.
7. Public docs must avoid formal conditional-validity claims.

Step 8 should avoid:

- public exposure of `experimental.ambiguity_normalized_novelty_penalized`;
- broad claims that difficulty improves accepted accuracy;
- implicit double-counting through VA difficulty plus reject-score difficulty;
- expanding `NormalizationStrategy`, which is already used for Venn-Abers
  interval/probability normalization.

## Guidance For Step 9

Step 9 should proceed only as interim documentation and research summary.

It should say:

1. Default reject remains the supported public behavior.
2. The existing VA-difficulty path can over-reject and is not the recommended
   difficulty-aware reject design.
3. Direct score normalization is the lead experimental design.
4. Full matched operating-point evidence does not justify public API promotion.
5. Novelty-aware scoring is internal only, despite its novelty-AUC signal.
6. Difficulty must be part of the conformal scoring pipeline, not a post-hoc
   threshold.
7. Difficulty-estimator provenance matters for validity.
8. Operating-point attainability is now a first-class design concern.

Step 9 should not include examples using public
`RejectPolicySpec.flag(ncf="difficulty_normalized")` unless Step 8 has landed.
If examples are included, they must use the experimental strategy name and mark
the workflow as experimental.

## Recommended Next Work

The next useful work is not Step 8. It is a narrower operating-point and
estimator-sensitivity pass.

Recommended next experiment:

1. Filter Scenario 11 to dataset/seed/target groups where both arms can get
   close to the target reject rate.
2. Report the same A-vs-C and C-vs-G deltas on those attainable subsets.
3. Add estimator-quality sensitivity:
   - deterministic baseline estimator;
   - noisy estimator;
   - random difficulty control;
   - optional diagnostic oracle-like proxy only if clearly excluded from
     validity claims.
4. Report target-distance distributions by dataset, arm, and target.
5. Reassess Step 8 only if direct normalization improves accepted accuracy and
   difficulty selectivity at meaningful matched targets.

Until then, the right status is:

- `experimental.difficulty_normalized`: continue development.
- `experimental.ambiguity_normalized_novelty_penalized`: keep internal, do not
  promote.
- public reject API: unchanged.
