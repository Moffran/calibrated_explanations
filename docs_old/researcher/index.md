# Researcher hub

{{ hero_calibrated_explanations }}

Connect calibrated explanations back to the published literature, benchmark
artefacts, and replication workflows. This hub surfaces the theory threads that
underpin probabilistic calibration and interval-aware interpretation while
giving you an actionable reproduction plan.

## Replicate published experiments

1. **Provision the evaluation environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .[dev,eval]
   ```
   The `[eval]` extra pulls in `xgboost`, `venn-abers`, and plotting libraries
   used across the flagship studies.
2. **Match the published datasets** – CSVs and archives live under
   [`data/`](https://github.com/Moffran/calibrated_explanations/tree/main/data).
   The binary classification suite reads directly from files such as
   `colic.csv`, `creditA.csv`, and `diabetes.csv`; the regression and ensured
   experiments use the corresponding subdirectories.
3. **Execute the scripted pipelines** – notebooks and scripts live in
   [`evaluation/`](https://github.com/Moffran/calibrated_explanations/tree/main/evaluation):
   - `Classification_Experiment_sota.py` runs the 25-dataset binary study using
     `train_test_split(..., random_state=42)` and persists pickled results next
     to the script (`results_sota.pkl`).
   - The `multiclass/` and `regression/` folders contain notebooks mirroring the
     multiclass and interval regression papers.
   - `ensure/` and `fastCE/` host the ensured-explanations and plugin
     acceleration artefacts, each with accompanying notebooks and saved
     payloads.
4. **Compare outputs** – result archives (`*.pkl`, `.zip`) ship with the repo so
   you can diff your runs against the published tables. Preserve the bundled
   random seeds (`0` or `42` depending on the asset) to align distributions.
5. **Document deviations** – note any changes to dataset versions, preprocessing,
   or calibrator settings in your replication log and cross-reference active
   ADRs in {doc}`../governance/release_checklist` before publishing.

## Build intuition before extending

- {doc}`../get-started/quickstart_classification` – Baseline experiments for
  binary and multiclass studies with sample outputs.
- {doc}`../get-started/quickstart_regression` – Probabilistic and interval
  calibration walkthroughs mirroring the regression paper.
- {doc}`../how-to/export_explanations` – Persist explanation payloads for
  replication packages and downstream analysis.
- {doc}`../how-to/interpret_explanations` – Shared vocabulary for factual vs.
  alternative narratives when writing papers.

## Publish responsibly

- {doc}`../research/index` – Research roadmap, benchmark notes, and funding
  acknowledgements.
- {doc}`../research/theory_and_literature` – Full bibliography across
  classification, regression, and uncertainty calibration.
- {doc}`../citing` – Ready-to-use citations for papers that reference calibrated
  explanations.
- {doc}`../overview/index` – Positioning statements to include in abstracts and
  grant reports.
- {doc}`../contributing` – Upstream contributions guide for sharing new
  calibration techniques.

```{toctree}
:maxdepth: 1
:hidden:

../research/index
../research/theory_and_literature
../citing
../get-started/quickstart_classification
../get-started/quickstart_regression
../how-to/export_explanations
../how-to/interpret_explanations
../overview/index
../governance/release_checklist
../contributing
```

{{ optional_extras_template }}
