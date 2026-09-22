# Tune runtime performance (opt-in)

CE provides three performance controls that stay disabled by default: the
calibrator cache, a parallel backend, and vectorised perturbations baked
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

When you already have a fitted wrapper, reuse its learner when constructing the
builder so the cached artefacts align with your deployed estimator:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder

# Prepare the same public quickstart flow inline so the snippet works from a wheel install.
dataset = load_breast_cancer()
x_train, _, y_train, _ = train_test_split(
    dataset.data,
    dataset.target,
    test_size=0.2,
    stratify=dataset.target,
    random_state=0,
)
model = RandomForestClassifier(random_state=0)
model.fit(x_train, y_train)

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
explainer = WrapCalibratedExplainer.from_config(config)
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
config_parallel = (
    builder.perf_parallel(True, backend="threads", workers=4, min_batch=8)
    .perf_cache(True)
    .build_config()
)
explainer_parallel = WrapCalibratedExplainer.from_config(config_parallel)
```

- ``backend`` accepts ``"threads"``, ``"processes"``, ``"joblib"``, or
  ``"sequential"``. v1 requires an explicit strategy; the former ``"auto"``
  strategy is no longer valid when parallel execution is enabled.
- ``workers`` caps the worker pool; omit it to use all logical CPUs.
- ``min_batch`` skips the executor for very small workloads so sequential
  execution stays cheaper.
- ``min_instances`` sets the floor for instance-parallel execution; defaults to
  ``max(8, chunk_size)`` so small-but-parallel-worthy batches are not forced
  to run serially.
- ``tiny_workload`` overrides the tiny-workload guard used before spinning a
  pool; omit it to rely on the adaptive per-granularity defaults (≈8–16 by
  default).

The ``CE_PARALLEL`` environment variable mirrors the builder options:

```bash
CE_PARALLEL="enable,threads,workers=8,min_batch=4,min_instances=8,tiny=12" python serve.py
```

Set ``CE_PARALLEL=off`` to fall back to single-threaded execution without
touching code. The executor resets the calibrator cache after forking, so cached
payloads remain process safe.

## What is guaranteed and what is a hint

The cache and the parallel backend are opt-in. CE never turns them on for you,
so enabling either one is always an explicit request. Some parts of that request
are guaranteed; others are hints that CE applies when it can.

**Guarantees**

- **Off by default.** Without ``perf_cache(True)``, ``perf_parallel(True)``,
  ``CE_CACHE`` or ``CE_PARALLEL``, no cache is built, work runs sequentially and
  CE emits no warnings about either feature.
- **A broken request fails closed.** If an enabled cache or parallel executor
  cannot be configured, ``build_config()``, ``WrapCalibratedExplainer.from_config()``
  or ``calibrate()`` raises ``ConfigurationError``. Invalid configuration values,
  a malformed ``CE_CACHE``/``CE_PARALLEL`` value such as ``workers=two``, and a
  failure while building the primitives all count. The error ``details`` name the
  ``capability`` (``cache`` or ``parallel``), the ``source`` that failed and the
  underlying ``cause``. CE does not quietly continue without the feature.
- **Precedence.** A ``perf_cache=``/``perf_parallel=`` argument to
  ``calibrate()`` wins over the environment, and ``CE_CACHE``/``CE_PARALLEL`` win
  over builder values. ``CE_CACHE=off`` or ``CE_PARALLEL=off`` therefore switches
  off a builder request without a warning; the INFO log records it as
  ``disabled_by_env``.
- **Same results.** Enabling the cache or the parallel backend changes timing,
  not explanation output or its ordering.

**Best-effort hints**

- ``workers``, ``min_batch``, ``min_instances`` and ``tiny_workload`` are
  thresholds and caps. Workloads below the thresholds run sequentially, which is
  logged once per executor.
- If the requested worker pool cannot start, for example because of operating
  system limits, CE runs sequentially and emits a ``UserWarning`` plus an INFO
  log.
- If ``backend="joblib"`` is requested but joblib is not installed, CE uses
  threads and emits a ``UserWarning`` plus an INFO log once per executor.
- ``force_serial=1`` in ``CE_PARALLEL`` retries a failed parallel batch
  sequentially and emits a ``UserWarning`` plus an INFO log.
- Without ``cachetools`` (install ``calibrated-explanations[perf]``), an enabled
  cache uses an in-package LRU/TTL backend with the same semantics and logs one
  warning per process.
- ``CE_CACHE`` is read when the configuration is built with
  ``ExplainerBuilder``/``ExplainerConfig``; a plain
  ``WrapCalibratedExplainer(model)`` does not read it.
- Speed-ups and cache hit rates are not guaranteed. Measure them with the
  telemetry callback before relying on them.

## Use vectorised perturbations via FAST explanations

Vectorised perturbations now ship in the core explainer. ``explain_factual`` and
``explore_alternatives`` rely on numpy masking rather than deep Python loops, so
you benefit immediately when the cache or parallel executor is enabled. The
``explain_fast`` plugin continues to offer additional heuristics, but it is no
longer required for SIMD-friendly perturbation handling.

## Filter features using internal FAST explanations

When the number of features is large, you can reduce compute by using internal
FAST explanations to discard unimportant features before running the full
factual/alternative explainers.

Enable this at build time:

```python
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder

builder = ExplainerBuilder(model)
config = (
    builder
    .perf_parallel(True, backend="threads", workers=4, granularity="instance")
    .perf_feature_filter(True, per_instance_top_k=8)
    .build_config()
)
wrapper = WrapCalibratedExplainer.from_config(config)
wrapper.fit(x_proper, y_proper)
wrapper.calibrate(x_cal, y_cal)
explanations = wrapper.explain_factual(x_test)
```

At runtime you can override or disable the filter via ``CE_FEATURE_FILTER``:

```bash
CE_FEATURE_FILTER="enable,top_k=8" python serve.py
```

Internally, each factual/alternative call:

- runs an internal FAST pass on the same batch to obtain per-instance weights,
- aggregates those weights and keeps at most ``top_k`` features for the batch,
- passes the resulting ``features_to_ignore`` to the existing execution plugins.

If the FAST plugin is not installed or fails, the filter is skipped with a
visible `UserWarning` and INFO log, and execution continues with the unfiltered
explainers.

## Roll back to the baseline runtime

1. Rebuild any configuration objects with ``perf_cache(False)`` and
   ``perf_parallel(False)``.
2. Remove the FAST plugin bundle (``pip uninstall external-plugins``) or revoke
   trust via ``CE_DENY_PLUGIN``/``calibrated_explanations.plugins.cli`` if you
   previously enabled it for additional heuristics.
3. Restart long-lived services to clear cached artefacts, worker pools, and any
   process-level telemetry counters.

Document the change in your release notes or change log so operators know the
performance toggles returned to their defaults. Capture cache hit/miss events
through the configured telemetry callback when you need before/after validation.

## See also

- [Configure runtime behaviour (ConfigManager)](configure_runtime.md) — full env-var
  reference table, pyproject.toml sections, and export diagnostics
- [Configure telemetry](configure_telemetry.md) — telemetry diagnostic mode and
  governance logging
