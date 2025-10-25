# Get started

{{ hero_calibrated_explanations }}

Follow these curated paths to run calibrated explanations without any optional
telemetry or plugin prerequisites. Each quickstart mirrors the README flow and
links directly to the companion notebook.

| Binary & multiclass classification | Probabilistic regression |
| --- | --- |
| [Classification quickstart](quickstart_classification.md)<br>[Demo notebook](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_binary_classification.ipynb) | [Regression quickstart](quickstart_regression.md)<br>[Probabilistic regression notebook](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_probabilistic_regression.ipynb) |

> 📈 **Interval regression signpost:** The regression quickstart highlights how
> calibrated interval outputs complement probabilistic thresholds, with links to
> the interval regression walkthrough for deeper coverage.

After you finish a quickstart, read the
{doc}`../how-to/interpret_explanations` guide to interpret factual, alternative,
probabilistic, and interval outputs, and visit the
{doc}`../concepts/alternatives` explainer for the full alternatives walkthrough.

## Interpretation & research links

- {doc}`../how-to/interpret_explanations` – Deep dive on factual, alternative,
  probabilistic, and interval outputs mirrored in the notebooks.
- {doc}`../concepts/probabilistic_regression` – Concept guide explaining how
  probabilistic thresholds pair with interval regression.
- {doc}`../citing` – Cite the binary, multiclass, probabilistic, and interval
  regression research when you publish results.
- {doc}`../research/index` – Research hub with proofs, benchmarks, and funding context underpinning the quickstarts.

```{toctree}
:maxdepth: 1

installation
quickstart_classification
quickstart_regression
troubleshooting
```

{{ optional_extras_template }}
