# Scenario A2: Representative Constraint Filtering in Guarded Factual Explanations

## Scientific question

Do guard p-values improve factual explanations by removing representative perturbations that violate a known structural data-generating constraint (x1 <= 2*x0 + 3)?

## Semantics statement

This evaluation is representative-level only. We score only representative perturbation values from emitted/candidate records. We do not probe interval boundaries or interiors, and we do not claim whole-interval validity.

## Setup

- Seeds: 12
- Models: logreg, rf
- Dimensionality profiles: [2, 10]
- Train/Calibration/Test sizes: 5500/3200/2600
- Sampled test instances per seed-model-dim: 500
- Guard settings: normalize_guard=True, n_neighbors=5, merge_adjacent=False, significance=[0.01, 0.05, 0.1, 0.2]
- Methods: standard, multibin_noguard, guarded, random_pruned_multibin

## Primary result: representative violation rate

Summary over seed-model-level aggregates (mean and 95% CI):

```csv
metric,method,mean,ci95_low,ci95_high,n_pairs

representative_violation_rate,guarded,0.0014105237077811385,0.0011224815611667533,0.0016985658543955237,192

representative_violation_rate,multibin_noguard,0.0046717365113027156,0.004347334332567244,0.004996138690038187,192

representative_violation_rate,random_pruned_multibin,0.004674701704713323,0.00434092792395485,0.0050084754854717965,192

representative_violation_rate,standard,0.02463235630004884,0.02395006828787939,0.025314644312218294,192
```

Paired Wilcoxon tests on seed-model-n_dim-significance aggregates:

```csv
comparison,metric,wilcoxon_p_value,median_difference_guarded_minus_baseline,paired_seed_model_n

guarded vs multibin_noguard,representative_violation_rate,6.977657777256645e-32,-0.002951176257867439,192

guarded vs random_pruned_multibin,representative_violation_rate,6.7723895627344705e-31,-0.0024837139389401776,192
```

Interpretation target: guarded should be lower than both multibin_noguard and random_pruned_multibin if improvement is not merely discretizer change or rule-count reduction.

## Mechanism result: candidate violation AUROC

Candidate-level AUROC uses 1 - p_value as anomaly score and labels each constrained-feature candidate by representative-level constraint violation.

```csv
metric,method,mean,ci95_low,ci95_high,n_pairs

candidate_violation_auroc,guarded,0.9141350269230695,0.9078196939436328,0.9204503599025061,192

candidate_violation_auroc,multibin_noguard,0.9141350269230695,0.9078196939436328,0.9204503599025061,192

candidate_violation_auroc,random_pruned_multibin,,,,0

candidate_violation_auroc,standard,,,,0
```

This tests alignment between guard score ranking and representative-level structural violations.

## Cost: rule count

rule_count is the mean number of emitted factual rules per instance.

```csv
metric,method,mean,ci95_low,ci95_high,n_pairs

rule_count,guarded,4.906343898640655,4.461064351810351,5.351623445470959,192

rule_count,multibin_noguard,5.698842243053711,5.172906519793845,6.224777966313578,192

rule_count,random_pruned_multibin,4.906343898640655,4.461064351810351,5.351623445470959,192

rule_count,standard,5.420602912317271,4.934451398921812,5.90675442571273,192
```

## Interpretation with Scenario B

Scenario B shows that guard p-values carry distributional signal under controlled shift. Scenario A2 complements this by showing that the same p-values, when used inside factual explanation generation, preferentially remove representative perturbations that violate a known structural constraint.

Acceptable claim: Under representative-level semantics, guarded factual explanations reduce constraint-violating representative perturbations relative to an unguarded multibin baseline, and the guard p-values rank violating representatives as less conforming.

## Limitations

This study evaluates representative perturbation points only and does not certify full emitted interval conditions. It focuses on one known synthetic structural constraint and two model families. Results should not be generalized to arbitrary domain constraints without task-specific validation.
