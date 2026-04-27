# Scenario A: Guarded vs Standard Calibrated Explanations

## Setup
- Seeds: 40
- Models: logreg, rf
- Test instances sampled per seed: 200
- Guard grid: significance=[0.05, 0.1], n_neighbors=[5], merge_adjacent=[False]

## Purpose
This report keeps only the paper-facing metrics for Scenario A.
The main comparison is the factual-mode violation rate under the emitted guarded rule format and the known domain constraint.
A secondary tradeoff metric is the factual-mode rule count.

## Factual violation rate
```csv
metric,model,mode,significance,guarded_mean,standard_mean,multibin_noguard_mean,median_difference_guarded_minus_standard,wilcoxon_p_value,_mean_reduction

violation_rate,rf,factual,0.1,0.07122916666666666,0.013437499999999996,0.0739375,0.056874999999999995,3.568126323893353e-08,-0.057791666666666665

violation_rate,logreg,factual,0.1,0.09673958333333334,0.017354166666666664,0.09538541666666667,0.08020833333333333,3.568126323893353e-08,-0.07938541666666668

violation_rate,rf,factual,0.05,0.07372916666666667,0.013437499999999996,0.0739375,0.060208333333333336,3.5668648332760216e-08,-0.06029166666666667

violation_rate,logreg,factual,0.05,0.09840625,0.017354166666666664,0.09538541666666667,0.08145833333333333,1.8189894035458565e-12,-0.08105208333333333
```

## Factual rule count
```csv
metric,model,mode,significance,guarded_mean,standard_mean,multibin_noguard_mean,median_difference_guarded_minus_standard,wilcoxon_p_value

rule_count,logreg,factual,0.05,3.1645000000000003,3.30325,3.4865000000000004,-0.15500000000000003,2.2236721140023346e-05

rule_count,rf,factual,0.05,3.645125,3.7521249999999995,3.90125,-0.09749999999999992,3.020392878891708e-07

rule_count,logreg,factual,0.1,2.9185000000000003,3.30325,3.4865000000000004,-0.35749999999999993,3.561822768321922e-08

rule_count,rf,factual,0.1,3.4204999999999997,3.7521249999999995,3.90125,-0.31999999999999984,3.554271352232285e-08
```

## Notes
The CSV outputs retain additional diagnostics for engineering use.
They are not intended as main paper evidence.
A practical starting point from the factual violation-rate table is significance=0.1.