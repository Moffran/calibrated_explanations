# Replication workflow

Use this workflow to reproduce the binary classification, multiclass,
regression, ensured, and fast calibrated explanations studies published by the
team.

## 1. Provision the evaluation environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev,eval]
```

The ``[eval]`` extra installs ``xgboost``, ``venn-abers``, and plotting
libraries referenced throughout the studies.

## 2. Match the published datasets

Use the manifests under
[`evaluation/`](https://github.com/Moffran/calibrated_explanations/tree/main/evaluation)
for dataset sources, preprocessing notes, and random seeds.

## 3. Execute the scripted pipelines

Run the notebooks and scripts in the evaluation directory that align with your
study:

- ``Classification_Experiment_sota.py`` covers the 25-dataset binary baseline
  and persists ``results_sota.pkl`` for diffs.
- ``multiclass/`` and ``regression/`` notebooks implement the multiclass and
  interval regression papers.
- ``ensure/`` and ``fastCE/`` contain ensured-explanations and fast plugin
  artefacts, each with accompanying result archives.

## 4. Compare outputs

Each evaluation asset ships with ``*.pkl`` or ``.zip`` archives so you can diff
against the published tables. Preserve the bundled random seeds (``0`` or
``42`` depending on the asset) to align distributions.

## 5. Document deviations

Record any dataset or calibrator changes in your replication log and cross-link
active ADRs via {doc}`../../foundations/governance/release_checklist` before you
publish.

```{toctree}
:maxdepth: 1
:hidden:

../../foundations/how-to/export_explanations
../../foundations/how-to/interpret_explanations
```
