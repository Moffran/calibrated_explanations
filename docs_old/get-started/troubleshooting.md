# Environment troubleshooting

Use this checklist if the quickstarts fail or telemetry payloads look empty.

| Symptom | Diagnosis | Fix |
| ------- | --------- | --- |
| `NotFittedError` when explaining | `fit` or `calibrate` was skipped. | Call `fit` on the wrapper, then `calibrate` with calibration data. |
| `ValidationError` about NaNs | Input arrays contain NaN/inf values. | Clean the data or add an imputer in the preprocessing pipeline. |
| Missing `predict_proba` on classifier | Estimator lacks probability estimates. | Enable probability outputs (e.g., `probability=True` for SVC) or switch to an estimator with `predict_proba`. |
| Empty telemetry dictionary | Old version or attribute stripped. | Upgrade to `calibrated-explanations>=0.9.0` and access `explainer.runtime_telemetry`. |
| Plotting ImportError | `viz` extra not installed. | `pip install "calibrated_explanations[viz]"` in the same environment. |
| CLI cannot find plugins | PATH not configured or package not installed in active env. | Run `python -m calibrated_explanations.plugins.cli list all` to bypass PATH resolution. |

## Collect diagnostics

```bash
python -m calibrated_explanations.plugins.cli list explanations --trusted-only
python -c "from calibrated_explanations import __version__; print(__version__)"
```

Attach the command outputs when filing an issue so maintainers can reproduce
setup differences quickly.
