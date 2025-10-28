# Research hub

Calibrated Explanations builds on a multi-year research program spanning
classification, probabilistic and interval regression, and uncertainty-aware
interpretation tooling. Start here to find the published literature,
benchmarks, and funding context, then drill into the dedicated pages for full
references.

## Replication workflow

1. Clone the repository and create a virtual environment that installs the
   evaluation extras:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .[dev,eval]
   ```

2. Load the dataset manifests and split definitions documented in the
   [evaluation README](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/README.md).
   Each benchmark lists the OpenML ID, preprocessing notes, and the random seeds
   used in the published studies.
3. Execute the notebooks and scripts under the
   [`evaluation/`](https://github.com/Moffran/calibrated_explanations/tree/main/evaluation)
   directory that match your study (binary classification, multiclass, regression,
   probabilistic, ensured explanations). Result archives (`*.pkl`, `.zip`)
   accompany each run so you can diff against the published tables.
4. Capture metrics with the same scoring functions and calibration settings
   referenced in the notebooks; they mirror the methodology described in the
   peer-reviewed papers linked below.
5. Document any deviations in your replication log so they can be reviewed
   alongside active ADRs and release milestones.

## Resources

- {doc}`theory_and_literature` – Published papers, in-flight studies, benchmarks,
  and funding acknowledgements with links to replication assets.
- {doc}`../citing` – Copy BibTeX entries for binary & multiclass classification
  plus probabilistic and interval regression papers when preparing your own
  publications.

```{toctree}
:maxdepth: 1

theory_and_literature
```
