# Calibrated Explanations ([Documentation](https://calibrated-explanations.readthedocs.io/en/latest/))

[![Calibrated Explanations PyPI version][pypi-version]][calibrated-explanations-on-pypi]
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/calibrated-explanations.svg)](https://anaconda.org/conda-forge/calibrated-explanations)
[![GitHub (Pre-)Release Date](https://img.shields.io/github/release-date-pre/Moffran/calibrated_explanations)](https://github.com/Moffran/calibrated_explanations/blob/main/CHANGELOG.md)
[![Docstring coverage](https://img.shields.io/badge/docstring%20coverage-94%25-brightgreen)](https://github.com/Moffran/calibrated_explanations/blob/main/reports/docstring_coverage_20251025.txt)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/Moffran/calibrated_explanations/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/calibrated-explanations)](https://pepy.tech/project/calibrated-explanations)

Calibrated Explanations turns any scikit-learn-compatible estimator into a
calibrated explainer that returns:

- **Factual rules** – the calibrated reasons your model backed its prediction.
- **Alternative rules** – what needs to change to flip or reinforce that
  decision, complete with uncertainty bounds.
- **Prediction intervals** – uncertainty-aware probabilities or regression
  ranges that quantify both aleatoric and epistemic risk.

Every quickstart, notebook, and benchmark follows the same recipe: fit your
estimator, calibrate on held-out data, then interpret the returned rule table
before acting.

---

## Your first calibrated explanation (≈5 minutes)

1. **Install the essentials**
   ```bash
   python -m pip install calibrated-explanations
   ```

   **Optional extras:**
   | Extra | Purpose | Key Packages |
   | :--- | :--- | :--- |
   | `[viz]` | Plotting and visualizations | `matplotlib` |
   | `[notebooks]` | Jupyter notebook support | `ipython`, `jupyter`, `nbconvert` |
   | `[eval]` | Reproducing benchmarks | `lime`, `shap`, `xgboost`, `scipy` |
   | `[external-plugins]` | High-performance plugins | `numpy>=1.24`, `pandas>=2.0`, `scikit-learn>=1.3` |

   Install with: `pip install "calibrated-explanations[viz,notebooks]"`

2. **Run the quickstart** – this mirrors the smoke-tested docs example.
   ```python
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from calibrated_explanations import WrapCalibratedExplainer

   dataset = load_breast_cancer()
   x_train, x_test, y_train, y_test = train_test_split(
       dataset.data,
       dataset.target,
       test_size=0.2,
       stratify=dataset.target,
       random_state=0,
   )
   x_proper, x_cal, y_proper, y_cal = train_test_split(
       x_train,
       y_train,
       test_size=0.25,
       stratify=y_train,
       random_state=0,
   )

   explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
   explainer.fit(x_proper, y_proper)
   explainer.calibrate(x_cal, y_cal, feature_names=dataset.feature_names)

   factual = explainer.explain_factual(x_test[:1])
   alternatives = explainer.explore_alternatives(x_test[:1])
   probabilities, probability_interval = explainer.predict_proba(x_test[:1], uq_interval=True)
   low, high = probability_interval
   print(f"Calibrated probability: {probabilities[0, 1]:.3f}")
   print(factual[0])
   ```
3. **Check the output** – the first factual explanation prints a calibrated rule
   table. A real run looks like:
   ```text
   Prediction [ Low ,  High]
   0.077 [0.000, 0.083]
   Value : Feature                                  Weight [ Low  ,  High ]
   0.07  : mean concave points > 0.05               -0.418 [-0.576, -0.256]
   0.15  : worst concave points > 0.12              -0.308 [-0.548,  0.077]
   0.34  : worst concavity > 0.22                   -0.090 [-0.123,  0.077]
   ```
   - The header row shows the calibrated prediction and its low/high credible
     interval.
   - Each subsequent line is a factual rule: the observed value, the matching
     feature, and its signed contribution with uncertainty bounds.
4. **Interpret what you see** – follow the
   [Interpret Calibrated Explanations guide](https://calibrated-explanations.readthedocs.io/en/latest/foundations/how-to/interpret_explanations.html)
   to learn how calibrated intervals, rule weights, and the triangular plot work
   together. The [triangular alternatives tutorial](https://calibrated-explanations.readthedocs.io/en/latest/foundations/concepts/alternatives.html)
   then shows how to narrate trade-offs across alternative rules.

---

## Mental model: fit → calibrate → explain → interpret

1. **Fit** your preferred estimator.
2. **Calibrate** with held-out data to align predicted and observed outcomes.
3. **Explain** with `explain_factual` for calibrated rules and
   `explore_alternatives` for semi-, super-, and counterfactuals.
4. **Interpret** using the how-to guides so decisions account for both aleatoric
   and epistemic uncertainty.

This workflow is identical across binary, multiclass classification, as well as probabilistic, and
interval regression tasks, the difference lies in how you configure the underlying estimator and read the returned intervals.

---

## Choose your path

### New practitioners (first run)
- Stay on this README quickstart, then open the
  [classification quickstart](https://calibrated-explanations.readthedocs.io/en/latest/get-started/quickstart_classification.html)
  for a notebook-friendly walk-through with the breast cancer dataset.
- Compare factual vs. alternative explanations using the
  [triangular plot tutorial](https://calibrated-explanations.readthedocs.io/en/latest/foundations/concepts/alternatives.html).

### Practitioners (day-to-day usage)
- Follow the
  [practitioner hub](https://calibrated-explanations.readthedocs.io/en/latest/practitioner/index.html)
  for production checklists, integration how-tos, and interpretation playbooks.
- Explore the
  [probabilistic regression quickstart](https://calibrated-explanations.readthedocs.io/en/latest/get-started/quickstart_regression.html)
  when you need calibrated thresholds.
- Opt into plugins only when needed via
  `pip install "calibrated-explanations[external-plugins]"`—they remain
  optional extensions.

### Researchers
- Reproduce published studies through the
  [researcher hub](https://calibrated-explanations.readthedocs.io/en/latest/researcher/index.html),
  which links directly to benchmark manifests, dataset splits, and evaluation
  notebooks.
- Fetch replication artefacts from the
  [evaluation README](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/README.md)
  and align with the release plan checkpoints.
- Cite the work using the ready-made entries in
  [docs/citing.md](https://calibrated-explanations.readthedocs.io/en/latest/citing.html).

### Contributors
- Start with the
  [contributor hub](https://calibrated-explanations.readthedocs.io/en/latest/contributor/index.html)
  for development environment setup, plugin guardrails, and quality gates.
- Review the
  [contributor hub](https://calibrated-explanations.readthedocs.io/en/latest/contributor/index.html)
  before submitting pull requests.

### Maintainers
- Track release readiness through the root-level
  [`ROADMAP.md`](https://github.com/Moffran/calibrated_explanations/blob/main/ROADMAP.md),
  [docs/foundations/governance/release_checklist.md](https://calibrated-explanations.readthedocs.io/en/latest/governance/release_checklist.html),
  and the implementation plan in
  [`docs/improvement/RELEASE_PLAN_v1.md`](https://github.com/Moffran/calibrated_explanations/blob/main/docs/improvement/RELEASE_PLAN_v1.md).
- Confirm Standards and ADR alignment via
  [docs/improvement/standards/](https://github.com/Moffran/calibrated_explanations/tree/main/docs/improvement/standards) and
  [docs/improvement/adrs/](https://github.com/Moffran/calibrated_explanations/tree/main/docs/improvement/adrs)
  and keep docs navigation synced with the
  [IA crosswalk](https://calibrated-explanations.readthedocs.io/en/latest/foundations/governance/nav_crosswalk.html).

---

## Documentation map

- **API reference** – start with the
  [API index](https://calibrated-explanations.readthedocs.io/en/latest/api/index.html),
  then browse CLI, plugin, serialization, and visualization references.
- **Architecture overview** – the
  [architecture notes](https://calibrated-explanations.readthedocs.io/en/latest/foundations/concepts/architecture.html)
  connect runtime components, telemetry, and plugin boundaries.
- **Contributor guidance** – see the
  [contributor hub](https://calibrated-explanations.readthedocs.io/en/latest/contributor/index.html)
  for setup, quality gates, and process notes.
- **Release notes & changelog** – check
  [release notes](https://calibrated-explanations.readthedocs.io/en/latest/foundations/governance/release_notes.html)
  and the project
  [CHANGELOG](https://github.com/Moffran/calibrated_explanations/blob/main/CHANGELOG.md).
- **Plugin CLI** – inspect registered plugins and trust state with
  `ce.plugins list all` (see the
  [CLI reference](https://calibrated-explanations.readthedocs.io/en/latest/api/cli.html)).
- **Project governance** – review
  [GOVERNANCE.md](https://github.com/Moffran/calibrated_explanations/blob/main/GOVERNANCE.md),
  [SECURITY.md](https://github.com/Moffran/calibrated_explanations/blob/main/SECURITY.md),
  and the
  [Code of Conduct](https://github.com/Moffran/calibrated_explanations/blob/main/CODE_OF_CONDUCT.md).
- **Support** – see
  [SUPPORT.md](https://github.com/Moffran/calibrated_explanations/blob/main/SUPPORT.md)
  for the fastest way to get help.

---

## Licensing & Contributions

Contributions to this project are licensed under the same terms as the project
itself (BSD 3-Clause). By contributing, you agree to the
[Developer Certificate of Origin (DCO)](https://developercertificate.org/)
and that your contributions will be available under the project's license.
See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for details on how to
sign off your commits.

---

## Feature highlights

- **Calibrated prediction confidence** for binary and multiclass classification.
- **Uncertainty-aware feature importance** with aleatoric and epistemic bounds.
- **Probabilistic and interval regression** that mirrors the classification API.
- **Alternative explanations with triangular plots** for visualising trade-offs.
- **Conjunctional and conditional rules** for interaction and fairness analysis.
- **Optional plugin lane** for fast explanations and telemetry—disabled by
  default, opt-in when you need it.

---

## Installation options

```bash
python -m pip install calibrated-explanations           # PyPI
conda install -c conda-forge calibrated-explanations    # conda-forge
python -m pip install "calibrated-explanations[dev]"    # local development tooling
python -m pip install "calibrated-explanations[viz]"    # plotting extras
```

Python ≥3.8 is supported. Optional extras remain additive so the core package
stays lightweight.

---

## Research and reproducibility

1. **Set up the evaluation environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .[dev,eval]
   ```
   The optional `[eval]` extras pull in `xgboost`, `venn-abers`, and plotting
   dependencies used across the published studies.
2. **Load the benchmark assets** – datasets live in the
   [`data/`](https://github.com/Moffran/calibrated_explanations/tree/main/data)
   directory (CSV files and zipped archives) and are referenced directly by the
   evaluation scripts.
3. **Re-run the flagship experiments** – each paper has a matching notebook or
   script under [`evaluation/`](https://github.com/Moffran/calibrated_explanations/tree/main/evaluation):
   - `Classification_Experiment_sota.py` and the accompanying notebooks cover
     the 25-dataset binary classification suite.
   - `multiclass/` and `regression/` host the multiclass and interval
     regression pipelines, respectively.
   - `ensure/` and `fastCE/` contain the ensured-explanations and accelerated
     plugin studies.
   Result archives (`*.pkl`, `.zip`) sit beside each run for quick comparison.
4. **Keep results traceable** – preserve the random seeds baked into the scripts
   (typically `42` or `0`) and record any deviations alongside the active ADRs
   noted in [`docs/improvement/adrs/`](https://github.com/Moffran/calibrated_explanations/tree/main/docs/improvement/adrs).
5. **Cite the sources** – the
   [theory & literature overview](https://calibrated-explanations.readthedocs.io/en/latest/researcher/advanced/theory_and_literature.html)
   lists DOIs, arXiv IDs, and funding acknowledgements to include in your work.

---

## Contributing and maintenance workflow

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .[dev] -c constraints.txt
   python -m pip install -r docs/requirements-doc.txt -c constraints.txt
   ```
2. **Run the quality gates locally**
   ```bash
   pytest
   ruff check .
   mypy src tests
   ```
3. **Build the documentation (optional but encouraged)**
   ```bash
   make -C docs html
   ```
4. **Open a pull request** referencing the active milestone and relevant ADRs.
   The [PR guide](https://calibrated-explanations.readthedocs.io/en/latest/foundations/governance/pr_guide.html)
   lists the checklist used during reviews.
5. **Review community health docs** – contributions are expected to follow the
   [Code of Conduct](https://github.com/Moffran/calibrated_explanations/blob/main/CODE_OF_CONDUCT.md),
   the contribution licensing guidance in
   [CONTRIBUTING](https://github.com/Moffran/calibrated_explanations/blob/main/.github/CONTRIBUTING.md),
   and the support/security policies in
   [SUPPORT.md](https://github.com/Moffran/calibrated_explanations/blob/main/SUPPORT.md)
   and [SECURITY.md](https://github.com/Moffran/calibrated_explanations/blob/main/SECURITY.md).

---

## License and citation

- Licensed under the [BSD 3-Clause License](https://github.com/Moffran/calibrated_explanations/blob/main/LICENSE).
- Cite Calibrated Explanations using the entries in
  [`CITATION.cff`](https://github.com/Moffran/calibrated_explanations/blob/main/CITATION.cff)
  or [docs/citing.md](https://calibrated-explanations.readthedocs.io/en/latest/citing.html).

---

## Acknowledgements & support

Funded by the [Swedish Knowledge Foundation](https://www.kks.se/) through the
Knowledge Intensive Product Realization SPARK environment at Jönköping
University. For questions or support, open an issue on
[GitHub](https://github.com/Moffran/calibrated_explanations/issues) or review
the guidance in
[SUPPORT.md](https://github.com/Moffran/calibrated_explanations/blob/main/SUPPORT.md).

[pypi-version]: https://img.shields.io/pypi/v/calibrated-explanations.svg
[calibrated-explanations-on-pypi]: https://pypi.org/project/calibrated-explanations/
