# Scenario A Guarded vs Standard CE Experiment

This script runs an empirical comparison between guarded and standard Calibrated Explanations for Scenario A (correlated 2D synthetic data with domain constraint `x2 <= 2*x1 + 3`).

## Run

```powershell
python evaluation/guarded_vs_standard_scenario_a.py --quick
python evaluation/guarded_vs_standard_scenario_a.py
```

## Outputs

Under `artifacts/guarded_vs_standard/scenario_a/`:

- `run_config.json`
- `per_instance_records.csv`
- `metrics_records.csv`
- `summary_metrics.csv`
- `retention_curve.png`
- `plausibility_vs_significance.png`
- `prediction_agreement_boxplot.png`
- `stability_bar.png`
- `runtime_bar.png`
- `report.md`

## Runtime estimate

- `--quick`: a few minutes.
- full run (`30` seeds, `200` instances/seed, `30` bootstraps): can be several hours depending on CPU.

## Notes

- Guarded and predictor calibration data consistency is asserted at runtime.
- Interval invariant `low <= point <= high` is asserted for rule-level prediction payloads.
- For factual baseline parity with guarded discretization, the script uses a multi-bin (`entropy`) standard factual path.
