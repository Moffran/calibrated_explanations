# Ensured Evaluation: How To Run

This folder contains the reproducible experiment runners for the ensured-explanations study.

## Prerequisites

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
pip install -r evaluation/requirements.txt
```

Notes:
- The runners read datasets from `data/` (`data/*.csv`, `data/Multiclass/multi/*.csv`, `data/reg/*`).
- Outputs are written to `evaluation/ensure/results/` by default.

## Quick Sanity Run

Run a small subset first:

```powershell
python -m evaluation.ensure.experiment_ensure_binary --limit-datasets 2
python -m evaluation.ensure.experiment_ensure_multiclass --limit-datasets 2
python -m evaluation.ensure.experiment_ensure_regression --limit-datasets 2
```

## Full Experiments

Default configuration:
- `test_size=100`
- `calibration_sizes=100 300 500`
- conjunction settings: `n_top_features=5`, `max_rule_size=2`

Run all three tasks:

```powershell
python -m evaluation.ensure.experiment_ensure_binary
python -m evaluation.ensure.experiment_ensure_multiclass
python -m evaluation.ensure.experiment_ensure_regression
```

Default result files:
- `evaluation/ensure/results/results_ensure_binary.pkl`
- `evaluation/ensure/results/results_ensure_multiclass.pkl`
- `evaluation/ensure/results/results_ensure_regression.pkl`

## Custom Runs

You can override key parameters:

```powershell
python -m evaluation.ensure.experiment_ensure_binary `
  --test-size 100 `
  --calibration-sizes 100 300 500 `
  --n-top-features 5 `
  --max-rule-size 2 `
  --out evaluation/ensure/results/custom_binary.pkl
```

The same flags are available for `experiment_ensure_multiclass` and `experiment_ensure_regression`.

## Export LaTeX Tables

After generating the `.pkl` result files:

```powershell
python -m evaluation.ensure.export_ensure_tables_latex
```

Custom input/output paths:

```powershell
python -m evaluation.ensure.export_ensure_tables_latex `
  --binary evaluation/ensure/results/results_ensure_binary.pkl `
  --multiclass evaluation/ensure/results/results_ensure_multiclass.pkl `
  --regression evaluation/ensure/results/results_ensure_regression.pkl `
  --out-dir evaluation/ensure/latex_tables `
  --show-std
```

Generated tables are written under:
- `evaluation/ensure/latex_tables/binary/`
- `evaluation/ensure/latex_tables/multiclass/`
- `evaluation/ensure/latex_tables/regression/`
- plus combined tables in `evaluation/ensure/latex_tables/`.

## Notebook / Paper Workflow

- `analysis_ensure.ipynb` can be used for inspection/plots from the saved `.pkl` files.
- `paper.tex` consumes generated LaTeX tables from `latex_tables/`.
