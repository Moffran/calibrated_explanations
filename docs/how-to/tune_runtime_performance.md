# Tune runtime performance (opt-in)

v0.9.0 introduces three performance controls that stay disabled by default:

the calibrator cache, a multiprocessing backend, and vectorised perturbations.
This guide shows how to enable each feature consciously and how to revert to the
baseline behaviour if they are not a fit for your deployment.

## Prerequisites

- Install ``calibrated-explanations`` as usual. Optional extras are only
  required when you enable the fast explanations plugin.
- Import :class:`calibrated_explanations.api.config.ExplainerBuilder` (or build an
  :class:`~calibrated_explanations.api.config.ExplainerConfig` manually) so you can
  toggle the cache and parallel backends without mutating the public
  ``WrapCalibratedExplainer`` API.
- Keep governance approvals handy—the release checklist treats these as opt-in
  features, so document who enabled them and why.

## Enable the calibrator cache

The cache saves intermediate calibration artefacts so repeated explanation runs
avoid recomputing identical payloads. It is disabled unless you flip the feature
flag when building your configuration.

```python
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder

builder = ExplainerBuilder(RandomForestClassifier(random_state=0))
config = builder.perf_cache(True, max_items=256).build_config()
explainer = WrapCalibratedExplainer._from_config(config)  # Private in v0.9.0
```

The ``max_items`` argument caps the number of cached entries; omit it to keep the
128-item default. To roll back, rebuild the configuration with
``perf_cache(False)`` or skip calling ``perf_cache`` entirely—no other code
changes are necessary.

## Enable multiprocessing for perturbations

The parallel backend runs perturbation-heavy steps across worker processes. Like
the cache, it remains off until you enable it on the configuration object.

```python
config = (
    builder.perf_parallel(True, backend="joblib")
    .perf_cache(True)
    .build_config()
)
explainer = WrapCalibratedExplainer._from_config(config)
```

- ``backend="joblib"`` forces the joblib executor. Use ``"auto"`` to let the
  runtime choose (it falls back to joblib today).
- The sequential fallback stays in place when ``perf_parallel_enabled`` is
  ``False``, so disabling the flag restores the single-process behaviour.
- Set the ``backend`` to ``"sequential"`` if you must keep the flag on but need
  to force deterministic single-worker execution temporarily.

## Use vectorised perturbations via FAST explanations

Vectorised perturbations ship as part of the opt-in FAST plugin bundle. Install
and register the bundle when you need the speed-up; otherwise the core runtime
uses the legacy perturbation loop.

```bash
pip install "calibrated-explanations[external-plugins]"
python -m external_plugins.fast_explanations register
```

After registration you can call ``WrapCalibratedExplainer.explain_fast`` to run
against the vectorised perturbation path provided by the plugin:

```python
fast = explainer.explain_fast(X_test, _use_plugin=True)
```

To roll back, skip installing the extra or run ``python -m
calibrated_explanations.plugins.cli deny core.explanation.fast`` to block the
 plugin, then fall back to ``explainer.explain_factual`` /
 ``explainer.explore_alternatives`` as before.

## Roll back to the baseline runtime

1. Rebuild any configuration objects with ``perf_cache(False)`` and
   ``perf_parallel(False)``.
2. Remove the FAST plugin bundle (``pip uninstall external-plugins``) or revoke
   trust via ``CE_DENY_PLUGIN``/``calibrated_explanations.plugins.cli``.
3. Restart long-lived services to clear cached artefacts and worker pools.

Document the change in your release notes or change log so operators know the
performance toggles returned to their v0.8.x defaults.

{{ optional_extras_template }}
