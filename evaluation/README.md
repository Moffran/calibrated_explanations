# Evaluation and Benchmarks

This folder contains experimental scripts and notebooks used for research and evaluation. These assets are intentionally optional to keep the core package lean while still shipping everything required to reproduce the published studies.

## Environment

 Create an isolated environment and install the optional dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install "calibrated_explanations[eval]"

# Alternatively, install from the provided requirements or conda environment:
pip install -r evaluation/requirements.txt
```

Or with conda:

```powershell
conda env create -f evaluation/environment.yml
conda activate ce-evaluation
```

## Notes

- Some notebooks rely on plotting; consider also installing:

```powershell
pip install "calibrated_explanations[viz]"
```

- Large datasets referenced in `data/` are not versioned for size reasons.
- Result archives (`*.pkl`, `.zip`) stored alongside the scripts provide the
  published baselines for quick comparison.

## Datasets

- Binary classification benchmarks load CSVs in the repository root `data/`
  directory (for example: `colic.csv`, `creditA.csv`, `diabetes.csv`).
- Multiclass experiments consume the OpenML exports bundled under
  `data/Multiclass/`.
- Ensured-explanation and fast plugin studies use the curated datasets inside
  `evaluation/ensure/` and `evaluation/fastCE/`.

## Entrypoints

- `Classification_Experiment_sota.py` reproduces the 25-dataset binary study and
  writes `results_sota.pkl` for verification.
- `multiclass/` and `regression/` contain notebooks that mirror the multiclass
  and interval regression publications; open them in Jupyter or run with
  `papermill` for batch execution.
- `ensure/` and `fastCE/` hold the ensured-explanations and acceleration
  artefacts, including notebooks and helper scripts.
- `scripts/compare_explain_performance.py` benchmarks the optimised explainer
  variants discussed in the performance appendix.
