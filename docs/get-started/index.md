# Get started

Follow these curated paths to run calibrated explanations without any optional
telemetry or plugin prerequisites. Each quickstart mirrors the README flow and
links directly to the companion notebook.

| Binary & multiclass classification | Probabilistic regression |
| --- | --- |
| [Classification quickstart](quickstart_classification.md)<br>[Demo notebook](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_binary_classification.ipynb) | [Regression quickstart](quickstart_regression.md)<br>[Probabilistic regression notebook](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_probabilistic_regression.ipynb) |

> 📈 **Interval regression signpost:** The regression quickstart highlights how
> calibrated interval outputs complement probabilistic thresholds, with links to
> the interval regression walkthrough for deeper coverage.

Regression in Calibrated Explanations is **conformal interval regression** implemented via
**Conformal Predictive Systems (CPS)**. Use `low_high_percentiles=(a, b)` to choose the CPS
percentiles for the returned interval; use `threshold=...` to switch to probabilistic (thresholded)
regression. See {doc}`../tasks/regression` and {doc}`../tasks/probabilistic_regression`.

After you finish a quickstart, read the
{doc}`../foundations/how-to/interpret_explanations` guide to interpret factual,
alternative, probabilistic, and interval outputs, and visit the
{doc}`../foundations/concepts/alternatives` explainer for the full alternatives
walkthrough.

Need CE-first guidance for agents and humans? See the
{doc}`ce_first_agent_guide` for a runnable, OSS-only workflow.
Need a faster start? Use the {doc}`../getting_started_60s` decision tree.

## Interpretation & research links

- {doc}`../foundations/how-to/interpret_explanations` – Deep dive on factual,
  alternative, and interval outputs mirrored in the notebooks.
- {doc}`../foundations/concepts/probabilistic_regression` – Concept guide explaining how
  probabilistic thresholds pair with interval regression.
- {doc}`../citing` – Cite the binary, multiclass, probabilistic, and interval
  regression research when you publish results.
- {doc}`../researcher/index` – Research hub with proofs, benchmarks, and funding context underpinning the quickstarts.

```{toctree}
:maxdepth: 1

installation
quickstart_classification
quickstart_regression
../getting_started_60s
ce_first_agent_guide
troubleshooting
```
