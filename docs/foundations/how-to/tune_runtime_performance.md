# Tune runtime performance (opt-in)

v0.9.0 introduces three performance controls that stay disabled by default: the
calibrator cache, a multiprocessing backend, and vectorised perturbations baked
into the core explainer. This guide shows how to enable each feature
consciously, how to tune the new configuration surface, and how to revert to the
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
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder

model = make_model()  # Replace with your fitted estimator
builder = ExplainerBuilder(model)
config = (
    builder.perf_cache(
        True,
        max_items=256,
        max_bytes=8 * 1024 * 1024,
        namespace="service-a",
        version="v2",
        ttl=600,
    )
    .build_config()
)
explainer = WrapCalibratedExplainer._from_config(config)  # Private in v0.9.0
```

- ``max_items`` caps the number of cached entries (defaults to 512).
- ``max_bytes`` imposes an approximate memory ceiling using array ``nbytes`` when
  available.
- ``namespace``/``version`` isolate callers so multiple services can safely
  share an in-memory cache.
- ``ttl`` (seconds) expires entries proactively; omit it to cache until evicted
  by LRU.

You can toggle the cache at runtime with the ``CE_CACHE`` environment variable.
The format accepts comma-separated directives:

```bash
CE_CACHE="enable,max_items=1024,ttl=900" python serve.py
```

Valid tokens include ``enable``/``on``/``off`` as well as ``namespace=``,
``version=``, ``max_items=``, ``max_bytes=``, and ``ttl=``. To roll back, rebuild
the configuration with ``perf_cache(False)`` or export ``CE_CACHE=off``.

## Enable multiprocessing for perturbations

The parallel backend runs perturbation-heavy steps across worker processes. Like
the cache, it remains off until you enable it on the configuration object.

```python
config = (
    builder.perf_parallel(True, backend="threads", workers=4, min_batch=8)
    .perf_cache(True)
    .build_config()
)
explainer = WrapCalibratedExplainer._from_config(config)
```

- ``backend`` accepts ``"threads"``, ``"processes"``, ``"joblib"``, or
  ``"auto"`` (chooses a strategy based on platform and CPU count).
- ``workers`` caps the worker pool; omit it to use all logical CPUs.
- ``min_batch`` skips the executor for very small workloads so sequential
  execution stays cheaper.

The ``CE_PARALLEL`` environment variable mirrors the builder options:

```bash
CE_PARALLEL="enable,threads,workers=8,min_batch=4" python serve.py
```

Set ``CE_PARALLEL=off`` to fall back to single-threaded execution without
touching code. The executor resets the calibrator cache after forking, so cached
payloads remain process safe.

## Use vectorised perturbations via FAST explanations

Vectorised perturbations now ship in the core explainer. ``explain_factual`` and
``explore_alternatives`` rely on numpy masking rather than deep Python loops, so
you benefit immediately when the cache or parallel executor is enabled. The
``explain_fast`` plugin continues to offer additional heuristics, but it is no
longer required for SIMD-friendly perturbation handling.

## Roll back to the baseline runtime

1. Rebuild any configuration objects with ``perf_cache(False)`` and
   ``perf_parallel(False)``.
2. Remove the FAST plugin bundle (``pip uninstall external-plugins``) or revoke
   trust via ``CE_DENY_PLUGIN``/``calibrated_explanations.plugins.cli`` if you
   previously enabled it for additional heuristics.
3. Restart long-lived services to clear cached artefacts, worker pools, and any
   process-level telemetry counters.

Document the change in your release notes or change log so operators know the
performance toggles returned to their v0.8.x defaults. Capture cache metrics via
``explainer._perf_cache.metrics.snapshot()`` or the telemetry callback if you
need before/after validation.
