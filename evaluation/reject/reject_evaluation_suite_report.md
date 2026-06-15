# Reject Evaluation Suite Report

This report is the human-oriented synthesis layer for the 12-scenario reject
evaluation suite under `evaluation/reject/`. The CSV files in
`evaluation/reject/artifacts/` are the row-level evidence; the JSON files are
the machine-readable metadata. This document exists to explain the *logic and
intent* behind the scenarios, to draw out the connections between them, and to
give any reader — whether encountering this suite for the first time or
reviewing a design change — a coherent picture of what the evaluation measures,
why it is organized the way it is, and what the current evidence supports.

---

## Part I — Background: What is reject in calibrated explanations?

### The prediction-set structure

`CalibratedExplainer` operates on inductive conformal prediction. For each test
instance it produces a *prediction set*: a subset of the label space that is
statistically guaranteed, at confidence `1 - epsilon`, to contain the true
label for a well-calibrated model. In the binary and multiclass classification
settings three structural outcomes are possible:

- **Singleton set** `{y}` — exactly one class is conforming. The system has
  enough confidence to resolve a unique decision.
- **Multi-label set** (ambiguity) — more than one class is conforming. The
  system cannot discriminate between them at the requested confidence level.
- **Empty set** (novelty) — no class is conforming. The instance lies outside
  the distribution the calibration set represents, or the model's nonconformity
  scores are inconsistent.

Reject is the act of withholding a prediction-set outcome from downstream use.
An instance is *accepted* when its prediction set satisfies the active reject
policy; otherwise it is *rejected* (deferred). The reject policy is configured
through `RejectPolicySpec` and evaluated via the `WrapCalibratedExplainer` API.

The word "reject" here does not mean "classify as negative" — it means "decline
to surface this output to the user or downstream system". Rejected instances
can be routed to a human reviewer, a fallback model, or logged for further
investigation. What matters for evaluation is therefore not just how many
instances are rejected but *which* instances, *why* they were rejected, and
whether the accepted subset is of higher quality than the full set.

### Why multi-objective evaluation is necessary

A system can trivially achieve very high accepted-set accuracy by rejecting
almost everything. A system can trivially accept everything and show baseline
accuracy. Neither is useful. The evaluation suite therefore tracks a family of
quantities that together diagnose whether a reject policy is doing something
meaningful:

| Quantity | What it asks |
|---|---|
| `coverage` | Does the prediction set contain the true label at the claimed rate? |
| `reject_rate` | What fraction of instances are withheld? |
| `accepted_accuracy` | How accurate are accepted-set decisions? |
| `accepted_accuracy_delta` | Does accepted accuracy exceed full-set accuracy, and by how much? |
| `singleton_precision` | Among singleton outputs, what fraction are correct? |
| `singleton_recall` | Among all labelable instances, what fraction are resolved as correct singletons? |
| `ambiguity_rate` | What fraction of instances become multi-label sets (uncertain)? |
| `novelty_rate` | What fraction become empty sets (out-of-distribution)? |
| `rejected_error_capture_rate` | What fraction of errors in the full set are captured in the rejected subset? |
| `difficulty_reject_auc` | How well does the reject ranking order instances by estimated difficulty? |
| ECE delta | Does the accepted subset have better or worse expected calibration error? |

No single metric suffices. A reject method that improves `accepted_accuracy`
purely by removing many instances will show high `accuracy_delta` but low
`singleton_recall`. A method that captures more errors will show a high
`rejected_error_capture_rate` but may also reduce `accept_rate` to the point of
operational uselessness. Coverage is a formal guarantee; accepted accuracy is
not.

---

## Part II — Metric lexicon

Several terms appear throughout all scenarios. This section defines them once.

**NCF (nonconformity function)**: The function that converts a model's raw
output into a scalar nonconformity score. The score measures how much the
calibration instance's output deviates from the expected pattern for its true
label. The suite tests two public NCFs: `default` (hinge-based) and `ensured`
(a stricter variant). The choice of NCF determines how the prediction-set
boundaries are drawn and therefore which instances are singletons, ambiguous, or
empty.

**Blend weight `w`**: A scalar `[0,1]` that mixes the model's raw probability
output with the NCF score. At `w=1.0` the NCF operates on the raw output alone;
at lower weights the blend smooths the scores. Scenario 4 explores how `w`
interacts with NCF choice.

**Coverage vs. accept rate**: These are frequently confused. *Coverage* is the
fraction of test instances whose prediction set contains the true label — a
conformal validity property. *Accept rate* is the fraction of instances the
reject policy accepts. A policy can have high accept rate and low coverage
(many singletons, many wrong) or low accept rate and high coverage (few
singletons, all correct). They are measured on different axes.

**Structural violation**: A coverage shortfall where the Clopper-Pearson
confidence interval upper bound falls below `1 - epsilon`. This means the
violation cannot plausibly be attributed to finite-sample noise — it is a
systematic failure of the conformal guarantee at this dataset and epsilon. A
finite-sample violation (CI covers `1 - epsilon`) is a known feature of small
calibration sets and is expected occasionally; a structural violation is not.

**Singleton precision and recall**: Precision is the fraction of singleton
outputs that are correct. Recall is the fraction of *all labelled test
instances* that are resolved as correct singletons. They form a pair: precision
asks "when the system makes a singleton decision, how often is it right?"; recall
asks "how much of the population is the system actually serving?". A method with
high precision but low recall is conservative but covers little of the
population.

**Arm notation (Scenarios 9–11)**: A–G label experimental configurations.
Arm A is the stable baseline (`builtin.default` strategy, no VA difficulty, NCF
`default`). Arm C is the experimental difficulty-normalized baseline
(`experimental.difficulty_normalized`, no VA difficulty, NCF `default`). Arms
B, D involve Venn-Abers difficulty rescaling layered on top of the score;
arms E, F combine difficulty normalization with `ensured` NCF; arm G adds a
novelty penalty on top of C.

---

## Part III — Evaluation design: five thematic clusters

The 12 scenarios are not independent experiments. They form a deliberate
evaluation structure with five thematic clusters. Understanding which scenarios
belong together, and why, is the key to reading the evidence correctly.

```
CLUSTER I   — Validity foundation          Scenarios 1, 7, 12
CLUSTER II  — Baseline characterization   Scenarios 2, 3
CLUSTER III — Configuration & quality     Scenarios 4, 5
CLUSTER IV  — Finite-sample robustness    Scenario 6
CLUSTER V   — Difficulty-aware research   Scenarios 8, 9, 10, 11
```

Each cluster is described in its own section below, with cross-scenario
analysis. The per-scenario numerical results then follow in the Results section.

---

## Part IV — Cluster I: The validity foundation (Scenarios 1, 7, 12)

### What these scenarios share

All three scenarios ask the same fundamental question: *does the conformal
prediction-set mechanism produce outputs whose true-label inclusion rate meets
the nominal confidence level?* This is the core formal property that
distinguishes calibrated conformal prediction from ordinary machine learning
confidence estimates. Without it, the rest of the evaluation is incomplete
because accept/reject decisions are built on top of these sets.

The three scenarios approach coverage validity from three angles:

- **Scenario 1** asks whether coverage holds for the stable baseline binary
  reject path across 26 datasets and three epsilon values. It is the primary
  benchmark.
- **Scenario 7** asks whether coverage holds across the *full NCF and weight
  grid* — i.e. whether any supported `(NCF, w)` combination breaks the baseline
  guarantee that Scenario 1 established.
- **Scenario 12** asks specifically whether the experimental arm C
  (`experimental.difficulty_normalized`) preserves coverage relative to arm A
  (`builtin.default`). It was motivated by the RT-3 red-team obligation: any
  strategy that modifies the nonconformity score must have its coverage validity
  verified empirically before promotion.

### Scenario 1: Binary marginal coverage sweep

This is the primary validity check for the suite. Twenty-six binary
classification datasets are evaluated at epsilon `{0.01, 0.05, 0.10}`,
producing 78 (dataset, epsilon) rows. Coverage violations are separated into
*finite-sample violations* (the Clopper-Pearson CI still covers `1 - epsilon`,
so noise is a plausible explanation) and *structural violations* (CI upper bound
strictly below `1 - epsilon`, indicating a systematic problem).

The observed violation rate is 31/78 (39.7%). This is higher than it might
appear alarming: violations are counted per `(dataset, epsilon)` row, not per
dataset, and finite-sample variability is expected especially at small `n_cal`
and `epsilon = 0.01`. The mean coverage across all rows is 0.9424. Structural
violations number only 4/78, all concentrated in datasets like `creditA` and
`colic` where calibration set sizes are modest relative to the stringency of the
epsilon.

The key interpretive point is that Scenario 1 is a characterization, not a
claim that binary reject always meets the formal target. It tells us where the
regime of clean behavior sits. The scenario also reports accepted accuracy as a
diagnostic — it is not the conformal claim.

### Scenario 7: NCF coverage validity sweep (supplementary)

This supplementary scenario extends Scenario 1's coverage check across the full
`(NCF, w)` grid: both `default` and `ensured` NCFs crossed with `w in
{0.3, 0.5, 0.7, 1.0}`, at epsilon `{0.05, 0.10}` across 26 binary datasets and
5 seeds. The question is whether any combination in the public-facing
configuration space produces empirical coverage shortfalls.

The result is **not clean**. The full artifact contains **841 row-level coverage
violations out of 2080 rows** and **100 row-level structural violations**. These
row counts should not be read as 100 independent failures because `default`
ignores `w`, so the same default result is repeated across four weight rows.
Collapsed by `(dataset, seed, ncf, epsilon)`, the structural count is **38/520**.

This replaces the older, incorrect interpretation that Scenario 7 had zero
violations. The earlier clean result was contaminated by an evaluation routing
bug: Scenario 7 looked for `result.prediction_set`, while FLAG stores prediction
sets in `result.metadata["prediction_set"]`. After fixing the access path, all
2080 rows have defined coverage and the violations are visible.

The implementation red-team also found a boundary bug in the core prediction-set
path: manual `predict_p()` thresholding used `p > epsilon`, whereas Crepes'
`predict_set()` includes labels with `p >= epsilon`. This has now been corrected.
It did not remove the Scenario 7 shortfalls, which means the remaining signal
should be treated as a real empirical warning rather than a simple thresholding
artifact.

The clearest empirical tendency is singleton collapse on harder datasets. When
the prediction sets are mostly singletons, coverage has little room to exceed
ordinary model accuracy. The updated artifact therefore reports baseline
accuracy, coverage lift over baseline accuracy, singleton rate, ambiguity rate,
novelty rate, and mean prediction-set size. The main structural clusters are
`je4243`, `heartS`, `creditA`, `liver`, `kc3`, `colic`, and `pc1req`; these are
not random-looking isolated cells. Mean coverage is close to the nominal target
in aggregate (`default`, epsilon 0.05: 0.9497; `default`, epsilon 0.10: 0.8980;
`ensured`, epsilon 0.05: 0.9554; `ensured`, epsilon 0.10: 0.9088), but the
per-dataset/per-split conditional behavior is uneven.

There is an important conformal-methods caveat: standard split conformal gives
marginal validity over the calibration/test draw, not guaranteed coverage for
every fixed dataset, seed, and test batch. A Clopper-Pearson upper bound below
`1 - epsilon` is a strong diagnostic for that fixed batch, but it is not by
itself a theorem-level contradiction of marginal conformal validity. Scenario 7
should therefore be read as a transparency and stress diagnostic for the CE
reject implementation, not as a formal pass/fail proof of conformal
classification.

### Scenario 12: Coverage validity — arm A vs arm C

Scenario 12 is the strongest counterargument to promoting arm C to stable API.
It mirrors the Scenario 1 protocol but runs arms A and C side by side: 26
binary datasets, 5 random seeds, epsilon `{0.05, 0.10}`, producing 260 rows per
arm.

The results show a meaningful divergence:

| | Arm A (`builtin.default`) | Arm C (`difficulty_normalized`) |
|---|---|---|
| Violations | 115/260 (44.2%) | 135/260 (51.9%) |
| Structural violations | **13/260** | **23/260** |
| Mean coverage | **0.9238** | **0.9146** |

Arm C has nearly twice as many structural violations as arm A (23 vs 13), and
its mean coverage is 0.92 pp lower. The structural violation gap is the decisive
finding: these are cases where even the Clopper-Pearson confidence interval
cannot attribute the shortfall to chance. Difficulty normalization modifies the
nonconformity score, and that modification changes the distribution of scores in
ways that reduce the conformal validity of the resulting prediction sets.

This does not mean arm C is wrong or useless — it means that before it can be
promoted from experimental to stable, the coverage deficit must be understood
and addressed. The current evidence says it should remain experimental.

### Cluster I synthesis

The three validity scenarios tell a coherent story. The baseline binary path
(Scenarios 1 and 7) meets the conformal guarantee in the expected range —
imperfectly across all datasets and epsilons in finite samples, but with very
few structural failures and zero failures across the full NCF/weight grid.
Experimental direct difficulty normalization (Scenario 12) weakens this
property measurably. The cluster establishes a clear gate: coverage validity is
a prerequisite, not an afterthought, for any promotion decision.

---

## Part V — Cluster II: Baseline characterization (Scenarios 2, 3)

### What these scenarios share

These two scenarios establish what the stable CE API currently provides for
non-binary tasks, and — critically — what it *does not* claim. Both deliver
an honest null result or a constrained interpretation that prevents misuse of
the API.

### Scenario 2: Multiclass correctness proxy

The CE multiclass reject does not produce a K-class conformal label set in the
same sense as the binary case. The evaluation opts into the
`experimental.multiclass_top1_correctness` strategy and treats the output as a
*binary correctness proxy*: a `{1}` singleton means the top-1 prediction
conforms under the proxy; a `{0}` singleton means the non-top-1 aggregate
event conforms. This is semantically different from conformal label-set
prediction over K classes — no alternative class is selected by a `{0}`
singleton.

The practical value of this proxy is real. Across 20 multiclass datasets at
epsilon `{0.05, 0.10}`:

- Mean proxy singleton accuracy: **0.8841** — when the system commits to a
  singleton, it is right 88% of the time.
- Mean singleton recall: **0.5031** — about half of all test instances are
  resolved as correct singletons.
- Mean non-accepted rate: **0.4711** — nearly half the population is deferred.
- Collapse events: 8 — cases where the proxy produces no useful singletons at
  all.

The non-accepted rate of 47% is high by operational standards, but it reflects
the honest difficulty of the task. The proxy is most useful for cautious
top-1 acceptance policies where a human will handle deferred instances. It
should not be used to claim which alternative class should be chosen when `{0}`
fires.

Hinge NCF is used for both `default` and `ensured` in this scenario. Margin NCF
was excluded because it produces identical scores for both columns, making
singletons structurally impossible — a design finding preserved in the scenario
record.

### Scenario 3: Thresholded regression binary-event reject validity

Regression reject in CE is evaluated as ordinary binary conformal classification
over a user-defined regression event. The scenario tests whether this binary
event formulation preserves transparent event-label coverage and singleton diagnostics.
Scalar thresholds define event `1` as `y <= threshold`; interval thresholds define
event `1` as `low < y <= high`.

Across the regression datasets, scalar threshold quantiles, one shared interval
threshold `(q25, q75)`, and confidence levels `{0.90, 0.95}`, Scenario 3 reports:

- event prevalence;
- empirical event coverage and confidence-interval violations;
- empty, singleton, and ambiguity prediction-set counts;
- novelty, ambiguity, singleton, and reject rates;
- singleton precision, recall, and empirical singleton error.

This is not an interval-width selection scenario and it is not conformal
prediction interval regression. The thresholded-regression reject question is:
"is the binary event conformally resolved as `{0}`, `{1}`, `{0,1}`, or `{}`?"
Accepted rows are singleton event prediction sets; rejected rows are empty or
ambiguous event prediction sets.

The scenario is relevant when the operational decision is genuinely a threshold
event, such as "is the target at or below this limit?" or "is the target inside
this clinically relevant interval?" It should not be used as a general-purpose
uncertainty screen for regression intervals.

### Cluster II synthesis

Both scenarios deliver constrained interpretations of existing functionality.
Scenario 2 says: use multiclass reject for cautious top-1 acceptance, report
singleton precision and recall, and do not claim K-class conformal prediction.
Scenario 3 says: use thresholded regression reject for binary event questions,
not for interval-width selection. These are not weaknesses of the API — they are
honest characterisations that prevent the API from being oversold.

---

## Part VI — Cluster III: Configuration space and quality (Scenarios 4, 5)

### What these scenarios share

These scenarios explore how configuration choices (NCF, weight, confidence
level) and operational conditions (reject-rate regime) affect the empirical
quality of the accepted subset. Neither makes a coverage claim — both operate
in the empirical layer above the conformal guarantee.

### Scenario 4: NCF and blend-weight grid

Forty-six classification datasets (binary and multiclass) are evaluated across
the full grid: NCF `{default, ensured}` × `w {0.3, 0.5, 0.7, 1.0}`. The
primary metric is the accepted-accuracy delta: how much does accepted accuracy
exceed the non-reject baseline?

The headline finding is that the choice of `(NCF, w)` can swing accepted
accuracy dramatically. The best observed delta is **+0.381** — nearly 38 pp
above baseline on a single `(dataset, NCF, w)` combination. However, several
structural patterns deserve attention:

- `w >= 0.7` converges NCF behavior. The `default` and `ensured` curves become
  nearly identical at high weights, removing NCF choice as a meaningful degree
  of freedom.
- `w = 0.3` amplifies NCF differences. With `ensured` at `w = 0.3`, accept
  rates on some datasets collapse nearly to zero because the stricter NCF
  rejects almost everything.
- `ensured` at low weights is operationally unreliable — it trades higher
  singleton precision for much lower singleton recall.

The scenario is most useful for configuration exploration before deployment: it
tells operators which `(NCF, w)` combinations are safe and which are degenerate.
It should not be read as a coverage result.

### Scenario 5: Explanation quality on accepted instances

This scenario asks whether rejecting uncertain instances measurably improves the
empirical quality of the subset that is shown to users. It uses 46 datasets at
confidence 0.95 and segments results into three reject-rate regimes:

| Regime | N datasets | Mean reject rate | Mean accuracy delta | Mean ECE delta |
|---|---|---|---|---|
| Low (≤ 15%) | 12 | 0.0584 | **+0.021** | −0.005 |
| Moderate (15–40%) | 9 | 0.2757 | **+0.061** | −0.027 |
| High (> 40%) | 25 | 0.6550 | **+0.144** | −0.100 |

The accuracy benefit scales with reject rate, but so does the loss of coverage:
high-regime datasets accept on average only 35% of instances. The accuracy delta
in the high regime (+0.144) looks impressive, but it is a *selection effect* —
the system is only making decisions for the easiest third of the population.

The ECE delta deserves careful interpretation. Negative ECE delta means the
accepted subset is *more* miscalibrated than the full set. This seems
paradoxical — shouldn't rejecting uncertain cases improve calibration? — but it
is a known selection artefact. If the easiest cases for which the model has
high confidence are already well-calibrated, removing uncertain cases changes
the calibration baseline rather than improving it uniformly. The sign of the
ECE delta is therefore regime-dependent and should not be used as a simple
quality indicator without knowing the baseline.

The practical conclusion from Scenario 5 is that accept/reject regimes should
be chosen based on the operational context: low-to-moderate reject rates give
meaningful accuracy improvements without abandoning large parts of the
population; high reject rates produce very accurate accepted subsets but at the
cost of population coverage.

### Cluster III synthesis

Scenarios 4 and 5 together characterise the quality landscape of the accepted
subset as a function of configuration. They reveal that the empirical quality
gain from rejection is real but varies strongly with NCF choice, weight, and
reject-rate regime. Both scenarios also confirm that accepted accuracy can
improve dramatically without any change to the underlying model — purely through
stricter selection. This is useful to know but must not be confused with a
coverage improvement. The distinction between conformal validity (Cluster I) and
empirical quality (Cluster III) is fundamental to correct evaluation.

---

## Part VII — Cluster IV: Finite-sample robustness (Scenario 6)

### Motivation and design

Conformal prediction guarantees are asymptotic: they hold exactly only in the
limit of infinite calibration data. In practice, calibration sets may be small
(10–200 instances) and users may request extreme confidence levels (epsilon =
0.005 or 0.01). Scenario 6 stress-tests these regimes to characterise where the
system becomes unreliable.

### Results

The scenario probes five calibration sizes `{10, 20, 50, 100, 200}` and extreme
confidence levels `{epsilon = 0.005, 0.01}` on selected binary datasets.

- **28 of 51 probe points** show coverage violations.
- **Max reject rate: 1.0** — at `n_cal = 10`, `epsilon = 0.05`, the system
  rejects every test instance because the calibration set is too small to define
  a meaningful threshold.
- **27 violations** come from the small-calibration probe; only **1** from the
  extreme-confidence probe.

The `n_cal = 10`, `epsilon = 0.05` row for `breast_cancer` is the canonical
example of total collapse: coverage is 1.0 (the only "safe" outcome when no
instance is accepted), reject rate is 1.0. This is not a bug — it is the
conformal mechanism doing exactly what it should when calibration data is
insufficient: it refuses to make singleton predictions rather than making
overconfident ones.

The key deployment guidance is: **before setting an epsilon target, verify that
the available calibration set size supports it**. Scenario 6 provides the
empirical characterization of where that boundary lies for the tested datasets.

---

## Part VIII — Cluster V: The difficulty-aware research arc (Scenarios 8–11)

### The central question

The CE API includes a `difficulty_estimator` parameter that adjusts
nonconformity scores via Venn-Abers probability rescaling. A natural extension
is to use difficulty information more directly — normalizing reject scores by
difficulty so that the hardest instances are prioritised for rejection. Scenarios
8–11 form a progressive four-stage investigation into this research direction.

The stages are ordered deliberately:

1. **Scenario 8** establishes the baseline: what does the *existing* VA
   difficulty path already do to reject behavior, before any new strategy is
   introduced?
2. **Scenario 9** introduces the experimental direct strategy (arm C) and
   compares it against the baseline in a six-arm ablation designed to expose
   also the potential for double-counting.
3. **Scenario 10** extends arm C with a novelty penalty (arm G) to test whether
   the reject taxonomy can be refined.
4. **Scenario 11** applies the strongest methodological control — matched
   operating-point selection — to evaluate all three strategies at comparable
   reject rates, providing the final promotion gate.

Each stage was motivated by findings from the previous one. This is not
post-hoc rationalization; the scenario sequence reflects a genuine research
progression.

### Scenario 8: The VA-difficulty baseline

Before introducing any new reject scoring strategy, it is essential to know what
the *existing* `difficulty_estimator` parameter already does. Scenario 8
answers this by comparing four arms across 46 datasets, 5 seeds, and 9
confidence levels: `use_difficulty` in `{False, True}` crossed with NCF
in `{default, ensured}`.

The existing difficulty path works through Venn-Abers probability rescaling.
The calibrated probabilities are adjusted based on a difficulty estimate before
the NCF is applied. This means the reject scoring formula itself is unchanged —
difficulty enters only indirectly through the rescaled probabilities.

The aggregate effect is striking:

| | `default` | `ensured` |
|---|---|---|
| Accept rate change | **−42.0 pp** | **−24.7 pp** |
| Accepted accuracy change | −10.1 pp | −10.5 pp |
| Rejected error capture rate change | **+18.8 pp** | **+7.6 pp** |
| Empirical coverage change | +4.4 pp | +2.8 pp |

Enabling difficulty through the VA path makes the system dramatically more
conservative. With `default` NCF it goes from accepting 63% of instances to
accepting 21% — a 42 pp reduction in accept rate. It captures meaningfully more
errors in the rejected set (18.8 pp more for `default`), but at a severe cost:
accepted accuracy *falls* by 10 pp because the system is now accepting only a
narrow slice of easy instances that the model was already handling well.

The difficulty gap metric confirms the mechanism is working: with difficulty
enabled, rejected instances are measurably harder than accepted ones (0.039
difficulty units for `default`, 0.076 for `ensured`). But the practical verdict
is that the VA path acts as a *blunt reject gate* — it selects harder instances
but cannot be tuned to target a specific reject rate without crossing the entire
difficulty distribution.

Scenario 8 is crucial because without it, Scenario 9 could not isolate the
effect of the new strategy. The baseline must be understood first.

### Scenario 9: Six-arm ablation — direct difficulty normalization

Scenario 9 introduces the experimental direct strategy: rather than rescaling
probabilities via VA, arm C computes a difficulty-adjusted nonconformity score
directly (`experimental.difficulty_normalized`, no VA difficulty, NCF
`default`). The six arms are:

| Arm | Strategy | VA difficulty | NCF |
|---|---|---|---|
| A | `builtin.default` | no | default |
| B | `builtin.default` | yes | default |
| C | `experimental.difficulty_normalized` | no | default |
| D | `experimental.difficulty_normalized` | yes | default |
| E | `experimental.difficulty_normalized` | no | ensured |
| F | `experimental.difficulty_normalized` | yes | ensured |

The **primary scientific contrast is A vs C** — cleanest because neither arm
uses VA difficulty, so the only variable is whether direct normalization is
applied. Arms B and D are diagnostic for what happens when both VA scaling and
direct normalization are active simultaneously (potential double-counting). Arms
E and F test whether `ensured` NCF changes the picture.

**A vs C headline results** (46 datasets, 5 seeds, 9 confidences):

- A vs C reject rate delta: **+0.0313** — arm C rejects slightly more.
- A vs C difficulty gap delta: **+0.2950** — arm C's rejected instances are
  substantially harder relative to accepted ones.
- A vs C difficulty-reject AUC delta: **+0.1985** — arm C's rejection ranking
  correlates better with instance difficulty.
- A vs C matched-bin accepted accuracy delta: **−0.0053** — at comparable reject
  rates, arm C's accepted accuracy is marginally *lower* than arm A's.

The difficulty-AUC result shows that direct normalization *works as a
mechanism*: it successfully shifts rejection toward harder instances. But the
matched-bin accepted accuracy result shows that this shift does not translate
into better accuracy for the accepted set at comparable reject rates. The harder
instances that arm C adds to the rejected set are not necessarily the ones that
arm A accepted incorrectly.

**Double-count diagnostics (D vs B, F vs E)**: Combining VA scaling and direct
normalization produces unstable behavior. The D-B reject rate delta is −0.147
and the F-E delta is +0.109 — the two effects interact non-additively. The
difficulty-gap deltas (D-B: +0.391; F-E: +0.285) are larger than the A-C
primary contrast alone, suggesting the combination amplifies the difficulty
signal but not in a controlled way. Arms D and F should not be used for
production: they risk double-counting the difficulty adjustment.

**The arm C recommendation** stems from the primary A-vs-C contrast: it is the
cleanest experimental baseline for difficulty-normalized reject scoring. It
avoids VA double-counting, it produces difficulty-AUC improvements, and it does
not regress coverage as badly as the double-count arms. Arm B (VA difficulty
alone) was already characterized in Scenario 8 and is known to be a blunt gate.

**Metric consistency note (RT-5)**: The full-grid A-vs-C AUC delta (+0.1985) is
higher than the Scenario 11 matched operating-point delta (+0.0155). This is a
selection effect, not a contradiction. The full-grid average includes high-
confidence rows where the difficulty-AUC advantage is strongest (+0.249 at
conf ≥ 0.91); matched operating-point selection targets moderate reject rates
where the advantage is much smaller. The Scenario 11 evidence is the correct
gate for promotion decisions.

### Scenario 10: Novelty-aware rejection — arm G

Ambiguous instances (multi-label sets) and novel instances (empty sets) are
qualitatively different failure modes. Ambiguity means the model cannot
discriminate between multiple conforming classes; novelty means the instance
falls outside the calibrated distribution. Arm G (`experimental.ambiguity_
normalized_novelty_penalized`) adds a separate novelty penalty (weight 0.1)
on top of arm C to test whether this distinction can be made actionable in the
rejection taxonomy.

**C vs G headline results** (46 datasets, 5 seeds, 9 confidences):

- G vs C novelty rate delta: **+0.0036** — arm G shifts slightly more
  instances into empty-set rejection.
- G vs C ambiguity rate delta: **+0.0065** — ambiguity rate also increases
  slightly (unexpected; the novelty penalty was intended to *reduce* it).
- G vs C accepted accuracy delta: **+0.0041** — arm G has marginally higher
  accepted accuracy than arm C.
- G vs C novelty-reject AUC delta: **−0.0302** — arm G's ranking by novelty
  score is actually *worse* than arm C's.

The deltas are small and the novelty-reject AUC result is surprising: adding a
novelty penalty does not improve the system's ability to rank instances by
novelty. The ambiguity rate increase is also unexpected — the penalty was
designed to push empty-set rejection up and multi-label rejection down, but both
increase slightly. This suggests that at novelty weight 0.1, the penalty is too
weak to produce a reliable directional shift.

The recommended arm remains C. Arm G is useful as a research tool for
investigating the taxonomy of rejection reasons, but its effect sizes are too
small and too inconsistent for a stable API role.

### Scenario 11: Matched operating-point selection — the promotion gate

Scenarios 9 and 10 compare strategies averaged over a wide confidence grid.
This introduces a confound: at different confidence values, arms A and C will
naturally have different reject rates, and comparing accuracy at different reject
rates is not a fair test. A strategy that rejects 30% of instances should be
compared to a baseline that also rejects 30%, not to one that rejects 15%.

Scenario 11 applies matched operating-point selection: for each strategy, the
confidence value that achieves the target reject rate closest to `{0.10, 0.20,
0.30, 0.40}` is selected independently. Deltas are then computed only at matched
pairs.

**A vs C at matched reject rates**:

| Target reject rate | C minus A accepted accuracy |
|---|---|
| 0.10 | **+0.0037** |
| 0.20 | **−0.0001** |
| 0.30 | **−0.0046** |
| 0.40 | **−0.0062** |

The pattern is clear: arm C shows a very small advantage at low reject rates
(+0.4 pp at 10% reject) and a small disadvantage at moderate-to-high reject
rates. The mean difficulty-reject AUC delta across all targets is +0.0155 —
substantially smaller than the full-grid +0.1985 from Scenario 9, confirming the
metric consistency note.

The fraction-positive statistics (how often C beats A on each metric, per
dataset-seed pair) hover near 0.35–0.45 for accepted accuracy across all target
rates. Arm C does not consistently outperform arm A; it is better on roughly
one-third to 40% of dataset-seed pairs and worse on the rest.

**C vs G at matched reject rates**:

| Metric | G minus C |
|---|---|
| Mean novelty rate delta | +0.0102 |
| Mean accepted accuracy delta | **−0.0023** |
| Novelty-reject AUC delta | **+0.0611** |
| Ambiguity rate delta | −0.0194 |

At matched operating points, arm G shows a cleaner novelty-routing signal than
in Scenario 10 (novelty-reject AUC +0.061, ambiguity down −0.019). However,
accepted accuracy is −0.002 relative to C. The novelty-aware strategy is better
at *diagnosing* why instances are being rejected (routing to novelty vs.
ambiguity) but does not improve the quality of accepted outputs.

The Scenario 11 promotion verdict is `do_not_promote`. The matched
operating-point evidence for difficulty normalization is too weak, and the
novelty-aware strategy remains internal-only.

### Cluster V synthesis

The four-stage arc reveals a coherent research story. The VA-difficulty path
(Scenario 8) is a blunt reject gate that works by making the system globally
more conservative — useful for understanding the baseline but not for targeted
difficulty selection. Direct difficulty normalization (Scenario 9) succeeds as
a mechanical improvement to reject selectivity (better difficulty-AUC) but
does not convert this into accepted accuracy gains at matched operating points.
The novelty extension (Scenario 10) clarifies the rejection taxonomy but lacks
sufficient effect size for promotion. The matched operating-point gate (Scenario
11) is the correct evaluation frame and its evidence does not support public API
promotion of either experimental strategy in their current form.

The research value of the arc is clear: it establishes what difficulty-aware
rejection can and cannot do with the current architecture, identifies the
double-counting risk, and defines the correct evaluation protocol for future
strategy development.

---

## Part IX — Numerical results summary

| Scenario | Datasets | Key result | Interpretation |
|---|---:|---|---|
| 1 | 26 binary | Mean coverage 0.9424; 4/78 structural violations | Baseline binary path broadly meets the formal target; finite-sample shortfalls are expected. |
| 2 | 20 multiclass | Proxy singleton accuracy 0.8841; singleton recall 0.5031; non-accepted rate 0.4711 | Multiclass proxy yields high precision on committed decisions but defers nearly half the population. |
| 3 | 22 regression | Mean interval-width delta ≈ 0.0000; reject rate 0.2083 | Threshold reject is a value selector, not an uncertainty selector. |
| 4 | 46 classification | Best accepted accuracy delta +0.381; w ≥ 0.7 converges NCFs | NCF and weight choice materially change accepted-set quality; this is not coverage. |
| 5 | 46 classification | Mean accuracy delta +0.093; ECE delta sign regime-dependent | Accepted-subset quality gains are real selection effects; ECE is unreliable as a simple quality metric. |
| 6 | 4 binary | 28/51 violations; max reject rate 1.0 | Small calibration sets and extreme epsilon produce fragile behavior; verify feasibility before deployment. |
| 7 | 26 binary | 841/2080 row-level violations; 100/2080 row-level structural violations; 38/520 collapsed structural violations | Full NCF/weight grid exposes empirical shortfalls; use as a regression guard and investigation target, not as clean validity evidence. |
| 8 | 46 classification | Default accept rate delta −42.0 pp; error capture +18.8 pp; accepted accuracy −10.1 pp | VA-difficulty path is a strict gate, not a targeted selector. |
| 9 | 46 classification | A vs C difficulty-AUC delta +0.1985; matched-bin accepted accuracy −0.0053 | Arm C selects harder cases but does not improve accepted accuracy at matched reject rates. |
| 10 | 46 classification | G vs C novelty rate delta +0.0036; accepted accuracy +0.0041 | Novelty penalty produces small, inconsistent routing shifts; arm C remains the experimental baseline. |
| 11 | 46 classification | Best A vs C delta +0.0037 at 10% reject; mean delta negative at 20–40% | Matched operating-point evidence does not support public API promotion. |
| 12 | 26 binary | Arm A structural violations 13/260; arm C structural violations 23/260; arm C mean coverage 0.9146 vs arm A 0.9238 | Direct difficulty normalization weakens coverage validity; this is the primary promotion gate blocker. |

---

## Part X — Cross-scenario synthesis

### The validity-quality distinction

The most important conceptual separation in the suite is between *validity*
(conformal coverage) and *quality* (accepted-subset accuracy, ECE, singleton
precision/recall). Scenarios 1, 7, and 12 live in the validity tier; Scenarios
4, 5, 8, 9, 10, and 11 live in the quality tier; Scenarios 2, 3, and 6
characterise the operational limits of existing functionality.

Higher accepted accuracy is not evidence of better validity. A reject policy can
achieve very high accepted accuracy by rejecting many instances — this is a
selection effect, not a conformal guarantee. Singleton recall is the corrective:
it asks what fraction of the full population is actually being served correctly.
A system with 0.95 singleton precision and 0.20 singleton recall is serving only
a fifth of the population correctly; a system with 0.85 singleton precision and
0.70 singleton recall is less precise but far more useful.

### Difficulty-awareness is real but not yet sufficient

Across Scenarios 8–11, the evidence for difficulty-aware rejection is
scientifically credible: the VA path captures more errors, arm C improves
difficulty-AUC, arm G shifts novelty routing. But none of these mechanisms
translates robustly into *accepted-accuracy improvements at matched reject
rates* across the 46-dataset suite. The effect exists in certain regimes (high
confidence, low reject rate) but is too inconsistent for stable API status.

Coverage validity (Scenario 12) adds the strongest constraint: arm C's modified
nonconformity score degrades the formal guarantee in a non-trivial way. Before
promotion, a coverage-preserving formulation of difficulty normalization is
needed.

### The operating-point matching principle

Scenarios 9 and 11 together teach an important evaluation methodology lesson.
A strategy that changes the confidence-to-reject-rate mapping will appear to
perform differently at any given confidence value than a baseline that maps that
same confidence to a different reject rate. The only fair comparison is at
matched reject rates, which Scenario 11 provides. Any future evaluation of
experimental strategies should follow the Scenario 11 protocol, not the
full-grid averaging of Scenario 9 in isolation.

### Where singleton precision and recall most matter

Singleton precision and recall are most informative in Scenarios 2, 4, 5, and
9–11. In Scenario 2, they reveal that the multiclass proxy has high precision
(0.884) but leaves half the population unresolved (recall 0.503). In Scenarios
9 and 11, they confirm that the A-vs-C differences in accepted accuracy are
mirrored in singleton precision (essentially unchanged) and singleton recall (a
small reduction in arm C at moderate reject rates). The precision-recall pair
is always more informative than either metric alone.

---

## Part XI — Conclusions

### Per-scenario conclusions

**Scenario 1.** Binary coverage meets the formal target in the expected
finite-sample regime. Structural violations are rare (4/78). Retain as the
primary baseline validity scenario.

**Scenario 2.** The multiclass top-1 correctness proxy is operationally useful
for cautious acceptance policies. Report singleton precision and recall
alongside accepted top-1 accuracy. Do not interpret `{0}` singletons as
alternative class selections, and do not claim K-class conformal coverage.

**Scenario 3.** Thresholded regression reject is binary conformal event
classification. Use it when the operational event is `y <= threshold` or
`low < y <= high`; do not interpret it as regression interval-width selection.

**Scenario 4.** NCF and weight choice strongly affect accepted-set quality.
Use this scenario for configuration exploration. `w ≥ 0.7` is the
operationally safe region; `ensured` at `w = 0.3` risks operational collapse.

**Scenario 5.** Accepted-subset accuracy gains are real but are selection
effects. The gain scales with reject rate, but so does loss of population
coverage. Choose reject regimes based on operational requirements, not
maximizing accuracy delta.

**Scenario 6.** Small calibration sets (n_cal ≤ 20) and extreme epsilon produce
fragile behavior, including total rejection. Use as a feasibility check before
setting deployment parameters.

**Scenario 7.** The full binary NCF/weight grid does not pass cleanly after the
prediction-set access bug is fixed. Retain it as a regression guard, but treat
its current violations as an investigation target rather than as evidence that
the NCF grid is deployment-clean.

**Scenario 8.** The existing VA-difficulty path is a conservative gate:
it accepts 42 pp fewer instances with `default` and captures 19 pp more errors,
but accepted accuracy falls by 10 pp. It is useful for understanding the
pre-existing baseline before introducing direct strategies.

**Scenario 9.** Arm C (`experimental.difficulty_normalized`) improves reject
selectivity by difficulty (AUC +0.199) but does not improve accepted accuracy at
matched reject rates (−0.005). Combining VA difficulty and direct normalization
risks double-counting. Arm C is the recommended experimental baseline for
further development.

**Scenario 10.** Arm G (novelty-aware) produces small, inconsistent routing
shifts. Novelty-reject AUC does not improve reliably; ambiguity routing is also
inconsistent at novelty weight 0.1. Arm C remains the simpler experimental
baseline.

**Scenario 11.** Matched operating-point evidence is the most deployment-like
evaluation frame and shows no consistent accepted-accuracy advantage for either
arm C or arm G at 20–40% reject rates. The promotion verdict is `do_not_promote`.

**Scenario 12.** Arm C has 23/260 structural coverage violations versus arm A's
13/260, and mean coverage 0.9146 versus 0.9238. This is the principal blocker
for arm C promotion: it weakens the conformal validity foundation.

### Overarching conclusion

The evaluation evidence supports a conservative and well-differentiated release
posture:

**Promote as stable.** The baseline binary conformal reject path remains the
most mature option, but Scenario 7 no longer supports calling the full NCF/weight
grid coverage-clean. Stability should be scoped to well-tested operating points,
with Scenario 1 and Scenario 7 artifacts reviewed before validity-sensitive use.

**Treat as a scoped empirical tool.** The multiclass top-1 correctness proxy
(Scenario 2) and thresholded regression reject (Scenario 3) are useful but must
be used within their stated semantics. Report singleton precision/recall for
multiclass and binary-event precision/recall for thresholded regression.

**Keep experimental.** Arm C (`experimental.difficulty_normalized`) and arm G
(`experimental.ambiguity_normalized_novelty_penalized`) should remain
experimental. The scientific mechanisms are real, the coverage deficit is
measurable, and the matched operating-point evidence does not meet the bar for
stable API.

**Use as deployment feasibility checks.** Scenario 6 should be run before any
deployment that involves small calibration sets or extreme epsilon settings.
Scenario 4 should be run before finalising NCF and weight choices.

The meta-lesson from this suite is that reject evaluation requires simultaneous
attention to at least three axes: validity (does the guarantee hold?), selectivity
(does rejection target the right instances?), and population coverage (what
fraction of users are still served?). A method that excels on one axis while
degrading another is not ready for stable promotion.
