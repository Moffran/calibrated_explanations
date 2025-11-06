# API reference

Core modules are documented automatically via Sphinx autosummary.

```{toctree}
:maxdepth: 1
:caption: Explainers

../../_autosummary/calibrated_explanations.core.CalibratedExplainer
../../_autosummary/calibrated_explanations.core.WrapCalibratedExplainer
```

```{toctree}
:maxdepth: 1
:caption: Collections

../../_autosummary/calibrated_explanations.explanations.CalibratedExplanations
../../_autosummary/calibrated_explanations.explanations.AlternativeExplanations
```

```{toctree}
:maxdepth: 1
:caption: Explanations

../../_autosummary/calibrated_explanations.explanations.CalibratedExplanation
../../_autosummary/calibrated_explanations.explanations.FactualExplanation
../../_autosummary/calibrated_explanations.explanations.AlternativeExplanation
../../_autosummary/calibrated_explanations.explanations.FastExplanation
```

```{toctree}
:maxdepth: 1
:caption: Utilities

../../_autosummary/calibrated_explanations.utils.helper
../../_autosummary/calibrated_explanations.core.exceptions
../../_autosummary/calibrated_explanations.core.validation
../../_autosummary/calibrated_explanations.api.params
../../_autosummary/calibrated_explanations.api.config
```

## Parameter aliases and configuration

`calibrated_explanations.api.params.canonicalize_kwargs` maps known aliases to
canonical parameter names (for example `alpha` → `low_high_percentiles`) without
removing the original keys. Warnings encourage users to migrate while preserving
backwards compatibility.

`calibrated_explanations.api.config.ExplainerConfig` and the accompanying
builder expose typed configuration scaffolding. In v0.9.0 the helper remains a
private entry point for the wrapper (`WrapCalibratedExplainer._from_config`), but
telemetry records any supplied defaults or preprocessors.
