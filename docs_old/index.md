# Calibrated Explanations documentation

{{ hero_calibrated_explanations }}

Welcome! These docs guide you from your **first calibrated explanation** through
advanced research and maintenance tasks. Start with the runnable quickstart
below, then follow the audience paths to dive deeper.

```{admonition} TL;DR mental model
:class: tip

1. Fit your preferred estimator.
2. Calibrate on held-out data.
3. Explain with `explain_factual` or `explore_alternatives`.
4. Interpret the calibrated probabilities and intervals before acting.
```

## Start here – run your first explanation

```bash
python -m pip install calibrated-explanations scikit-learn
```

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from calibrated_explanations import WrapCalibratedExplainer

dataset = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data,
    dataset.target,
    test_size=0.2,
    stratify=dataset.target,
    random_state=0,
)
X_proper, X_cal, y_proper, y_cal = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    stratify=y_train,
    random_state=0,
)

explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal, feature_names=dataset.feature_names)

factual = explainer.explain_factual(X_test[:1])
probability, (low, high) = explainer.predict(X_test[:1], uq_interval=True)

print(f"Calibrated probability: {probability[0]:.3f}")
print(factual[0])
```

```{admonition} What success looks like
:class: hint

```text
Prediction [ Low ,  High]
0.077 [0.000, 0.083]
Value : Feature                                  Weight [ Low  ,  High ]
0.07  : mean concave points > 0.05               -0.418 [-0.576, -0.256]
0.15  : worst concave points > 0.12              -0.308 [-0.548,  0.077]
```

Use the {doc}`how-to/interpret_explanations` guide to map these rule weights and
intervals to decisions, then explore the triangular plot walkthrough for
alternative explanations.
```

- The header row reports the calibrated probability with its uncertainty
  interval.
- Each subsequent line is a factual rule: the observed value, the associated
  feature, and its signed contribution with bounds.

Follow up with the {doc}`concepts/alternatives` tutorial to see how alternative
rules appear in the triangular plot and how to narrate their trade-offs.

## Audience journeys

### New practitioners

- Follow the {doc}`get-started/quickstart_classification` tutorial you just ran
  for a notebook-ready flow and interpretation checkpoints.
- Compare factual and alternative outputs with the
  {doc}`concepts/alternatives` triangular plot guide.
- Share the outcome with stakeholders via the
  {doc}`how-to/interpret_explanations` storytelling checklist.

### Practitioners

- Use the {doc}`practitioner/index` hub for production checklists, integration
  guides, and interpretation recipes across classification, regression, and
  probabilistic workflows.
- Need calibrated thresholds? Jump to the
  {doc}`get-started/quickstart_regression` notebook-friendly walkthrough.

### Researchers

- Start at the {doc}`researcher/index` hub for benchmark manifests, dataset
  splits, and replication workflow summaries.
- Find DOIs, arXiv IDs, and evaluation coverage in the
  {doc}`research/theory_and_literature` roundup.
- Reproduce experiments via the evaluation scripts and notebooks under the
  [evaluation/](https://github.com/Moffran/calibrated_explanations/tree/main/evaluation)
  directory; result archives (`*.pkl`, `.zip`) live alongside each run for quick
  diffing against the published tables.

### Contributors

- Spin up a development environment with the
  {doc}`contributing` workflow (venv, `pip install -e .[dev]`, required checks).
- Check the {doc}`contributor/index` hub for coding standards, plugin guardrails,
  and governance expectations.
- Review the {doc}`governance/release_checklist` before shipping changes to keep
  docs, quickstarts, and QA gates in sync.

### Maintainers

- Prioritise issues via the triage workflow outlined in the
  {doc}`contributor/index` hub and capture ADR impacts while reviewing pull
  requests.
- Align milestones with the {doc}`governance/nav_crosswalk` so README, RTD
  navigation, and notebooks stay consistent.

## Key references

- {doc}`Research hub <research/index>` – Publication summaries, benchmark
  coverage, and funding acknowledgements in one place.
- {doc}`Citing calibrated explanations <citing>` – Copy ready-to-use citations
  for binary & multiclass classification plus probabilistic and interval
  regression results when you publish your findings.

```{toctree}
:caption: Audience hubs
:maxdepth: 1

practitioner/index
researcher/index
contributor/index
```

```{toctree}
:caption: Overview
:maxdepth: 1

overview/index
```

```{toctree}
:caption: Get started
:maxdepth: 1

get-started/index
```

```{toctree}
:caption: How-to guides
:maxdepth: 1

how-to/index
```

```{toctree}
:caption: Concepts & architecture
:maxdepth: 1

concepts/index
```

```{toctree}
:caption: Reference
:maxdepth: 1

reference/index
```

```{toctree}
:caption: Research
:maxdepth: 1

research/index
```

```{toctree}
:caption: Extending the library
:maxdepth: 1

extending/index
```

```{toctree}
:caption: Governance & support
:maxdepth: 1

governance/index
```

{{ optional_extras_template }}
