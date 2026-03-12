# calibrated-explanations

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/calibrated-explanations)](https://pypi.org/project/calibrated-explanations/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/calibrated-explanations.svg)](https://anaconda.org/conda-forge/calibrated-explanations)
[![Docs](https://readthedocs.org/projects/calibrated-explanations/badge/?version=latest)](https://calibrated-explanations.readthedocs.io)
[![CI](https://github.com/Moffran/calibrated_explanations/actions/workflows/ci-main.yml/badge.svg)](https://github.com/Moffran/calibrated_explanations/actions/workflows/ci-main.yml)

**Trustworthy AI explanations with uncertainty intervals and counterfactuals, for any scikit-learn model.**

---

## What Problem Does It Solve?

Most XAI tools — LIME, SHAP — explain whatever the model outputs. If the model's predicted probabilities are miscalibrated (and they often are, especially for tree-based models trained without a calibration step), the explanations inherit that miscalibration. The feature weights reflect an overconfident model, not a reliable signal you can act on.

`calibrated-explanations` fixes this at the root. It calibrates the model first — using Venn-Abers predictors for classification and Conformal Predictive Systems for regression — then explains the calibrated output. Every explanation includes a **calibrated uncertainty interval** that shows how confident the model actually is, not just a point estimate that hides model uncertainty.

---

## What Does the Output Look Like?

Calling `explanation[0].to_narrative(output_format="text", expertise_level="advanced")` returns a structured text narrative. The output below is **(illustrative)** — a loan-approval context using the exact format produced at runtime:

```text
Factual Explanation:
--------------------------------------------------------------------------------
Prediction: APPROVE
Calibrated Probability: 0.840
Prediction Interval: [0.710, 0.930]

Factors impacting the calibrated probability for class APPROVE positively:
annual_income (45200) >= 45000 — weight ≈ +0.312 [+0.198, +0.421]
credit_history_years (5.2) >= 5 — weight ≈ +0.187 [+0.091, +0.284]
outstanding_debt (2800) < 3000 — weight ≈ +0.143 [+0.055, +0.231]
employment_status (permanent) = permanent — weight ≈ +0.098 [+0.012, +0.185]

Factors impacting the calibrated probability for class APPROVE negatively:
missed_payments (3) > 2 — weight ≈ -0.201 [-0.334, -0.068]
```

- The **Prediction Interval** `[0.710, 0.930]` shows the calibrated uncertainty range — narrow means high confidence, wide (e.g., `[0.12, 0.89]`) means the model is uncertain and the decision should be treated with caution.
- Each **factor line** shows the observed value, the matching rule condition, the signed weight (positive = pushes toward the predicted class), and the **endpoint envelope describing the prediction boundary shift** — all computed from calibrated probabilities, not raw model scores.

Calling `alt[0].to_narrative(output_format="text", expertise_level="advanced")` on the result of `explore_alternatives` shows what needs to change to flip or reinforce the decision, with each alternative backed by a calibrated interval:

```text
Alternative Explanations:
--------------------------------------------------------------------------------
Prediction: APPROVE
Calibrated probability: 0.840
Prediction Interval: [0.710, 0.930]

Alternatives to increase the calibrated probability for class APPROVE:
- If missed_payments <= 1 then 0.921 [0.856, 0.970]
- If outstanding_debt < 2000 then 0.893 [0.814, 0.949]

Alternatives to decrease the calibrated probability for class APPROVE:
- If annual_income < 30000 then 0.518 [0.344, 0.686]
- If credit_history_years < 2 then 0.601 [0.447, 0.752]
```

- Each **alternative line** shows the rule that would need to hold, the resulting calibrated probability if that rule were satisfied, and the **uncertainty interval on that alternative** — narrow means the model is confident even in the counterfactual region.

---

## Quick Start

```bash
pip install calibrated-explanations
```

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from calibrated_explanations import WrapCalibratedExplainer

d = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(d.data, d.target, test_size=0.2, stratify=d.target, random_state=42)
X_pr, X_cal, y_pr, y_cal = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)
explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=42))
explainer.fit(X_pr, y_pr);
explainer.calibrate(X_cal, y_cal, feature_names=d.feature_names)
exp = explainer.explain_factual(X_te[:1])
print(exp[0].to_narrative(output_format="text", expertise_level="advanced"))
exp[0].plot()
```

---

## What Can It Explain?

| Task | Description | Key Method |
|---|---|---|
| Binary classification | Binary yes/no decision with calibrated probabilities | `explain_factual`, `explore_alternatives` |
| Multiclass | Multiclass classification (3+ classes), per-class explanations | `explain_factual`, `explore_alternatives` |
| Regression with intervals | Predict a value with a conformal uncertainty interval defined by given percentiles | `explain_factual(low_high_percentiles=(5, 95))`, `explore_alternatives(low_high_percentiles=(10, 90))` |
| Probabilistic regression | Explain a probability query on a regression target (e.g., $P(y \le t)$ or $P(t_l < y \le t_h)$ ) | `explain_factual(threshold=t)`, `explore_alternatives(threshold=(t_l, t_h))` |

All four modes use the same API — the wrapper infers classification vs regression from the underlying estimator (or you pass `mode` to `calibrate`), and you add `threshold` only for probabilistic regression.

---

## What Makes It Different?

| Feature | LIME | SHAP | calibrated-explanations |
|---|---|---|---|
| Calibrated probabilities and predictions | No | No | **Yes** |
| Uncertainty intervals per explanation | No | No | **Yes** |
| Built-in counterfactual / alternative rules | No | No | **Yes** |
| Deterministic (stable) output | No | High (TreeExplainer) | **Yes** |
| Uncertainty-qualified counterfactuals (Ensured framework) | No | No | **Yes** |
| Conditional calibration by group | No | No | **Yes** |

> **Honest limitation:** CE does not currently provide global feature importance rankings — tasks requiring aggregated SHAP-style importance plots should use SHAP for that component.

---

## Uncertainty-Qualified Counterfactuals (Ensured)

Standard counterfactual methods tell you "change feature X to value Y and the decision flips." They do not tell you whether the model is actually confident about that alternative scenario. The counterfactual may point to a region of input space where the model has seen very little training data, meaning the flip is a formal artefact, not a reliable prediction.

CE's **Ensured** framework (Löfström et al., arXiv:2410.05479) addresses this directly. The `ensured_explanations()` filter keeps only counterfactuals where the model's calibrated interval lies **fully in the opposite class** — providing formal coverage evidence that the alternative decision is not merely a point estimate crossing the class boundary. The result: every surfaced counterfactual is backed by conformal guarantees.

```python
explainer.explore_alternatives(X_query)[0].ensured_explanations()  # X_query: array-like, shape (n_samples, n_features)
```

> **Read more:** [Ensured explanations playbook](https://calibrated-explanations.readthedocs.io/en/latest/practitioner/playbooks/ensured-explanations) · [Alternatives concept guide](https://calibrated-explanations.readthedocs.io/en/latest/foundations/concepts/alternatives)

---

## Fairness-Aware Explanations

A model can be globally well-calibrated but systematically overconfident for a minority group. CE's **Mondrian/conditional calibration** conditions calibration and uncertainty on a per-instance group label (`bins`) (Löfström & Löfström, xAI 2024). The result: explanation uncertainty intervals are valid *within each group*, not only on average. Wider intervals for a group are a direct, auditable signal of data insufficiency — a concrete fairness artefact that can be reported to regulators or risk committees.

```python
explainer.explain_factual(X_query, bins=X_query[:, gender_col_index])
```

> **Read more:** [Mondrian / conditional calibration playbook](https://calibrated-explanations.readthedocs.io/en/latest/practitioner/playbooks/mondrian-calibration)

---

## Research and Citations

`calibrated-explanations` is the product of peer-reviewed research. If you use it, please cite the relevant paper(s).

1. Löfström, H., Löfström, T., Johansson, U., Sönströd, C. (2024). "Calibrated Explanations: with Uncertainty Information and Counterfactuals." *Expert Systems with Applications*. doi:[10.1016/j.eswa.2024.123154](https://doi.org/10.1016/j.eswa.2024.123154)

2. Löfström, T., Löfström, H., Johansson, U., Sönströd, C., Matela, R. (2025). "Calibrated Explanations for Regression." *Machine Learning* 114, 100. Springer Nature. doi:[10.1007/s10994-024-06642-8](https://doi.org/10.1007/s10994-024-06642-8)

3. Löfström, H., Löfström, T. (2024). "Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty." *xAI 2024*, Communications in Computer and Information Science, vol 2153. Springer, Cham. doi:[10.1007/978-3-031-63787-2_17](https://doi.org/10.1007/978-3-031-63787-2_17)

4. Löfström, T., Löfström, H., Johansson, U. (2024). "Calibrated Explanations for Multi-class." *Proceedings of the Thirteenth Symposium on Conformal and Probabilistic Prediction with Applications*, Proceedings of Machine Learning Research, PMLR 230:175–194.

5. Löfström, H., Löfström, T., Hallberg Szabadvary, J. (2024). "Ensured: Explanations for Decreasing the Epistemic Uncertainty in Predictions." *arXiv:2410.05479*. [https://arxiv.org/abs/2410.05479](https://arxiv.org/abs/2410.05479)

6. Löfström, H. et al. (2023). "Investigating the impact of calibration on the quality of explanations." *Annals of Mathematics and Artificial Intelligence*.

BibTeX entries are available in [`CITATION.cff`](CITATION.cff).

---

## Install & Requirements

```bash
pip install calibrated-explanations
```

- Python ≥ 3.8
- scikit-learn ≥ 1.3
- crepes ≥ 0.8.0 (conformal calibration backend)
- venn-abers ≥ 1.4.0 (Venn-Abers calibration)
- numpy ≥ 1.24, pandas ≥ 2.0 (standard data science stack)

Optional: `matplotlib` is required only for `.plot()` visualisation calls.

---

## Documentation

- [Full documentation](https://calibrated-explanations.readthedocs.io)
- [Getting started in 60 seconds](docs/getting_started_60s.md)
- [Contributing](CONTRIBUTING.md)

---

## License

Released under the [BSD 3-Clause License](LICENSE) — open for both academic and commercial use without restriction.

---

## Acknowledgements

Development of `calibrated-explanations` has been funded by the Swedish Knowledge Foundation together with industrial partners supporting the research and education environment on Knowledge Intensive Product Realization SPARK at Jönköping University, Sweden, through projects: AFAIR grant no. 20200223, ETIAI grant no. 20230040, and PREMACOP grant no. 20220187. Helena Löfström was initially a PhD student in the Industrial Graduate School in Digital Retailing (INSiDR) at the University of Borås, funded by the Swedish Knowledge Foundation, grant no. 20160035.
