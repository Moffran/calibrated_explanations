"""Explain-specific parallel runtime wiring.

This module keeps explain-facing heuristics and chunking decisions co-located
with the explain executors while delegating low-level execution to the shared
``calibrated_explanations.parallel`` facade. It wraps the generic
``ParallelExecutor`` with explain-aware defaults (granularity, minimum
instance thresholds, and chunk sizing) so that explain strategies avoid
sprinkling parallel plumbing across sibling packages.
"""

from __future__ import annotations

import logging
import numbers
import time
import types
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np

from ...parallel import ParallelConfig, ParallelExecutor
from ...utils.int_utils import as_int_array
from ._helpers import merge_ignore_features, validate_and_prepare_input
from ._shared import ExplainConfig, ExplainRequest

if TYPE_CHECKING:
    from ...plugins import ExplanationRequest

_DEFAULT_PERCENTILES: Tuple[float, float] = (5, 95)
logger = logging.getLogger(__name__)


@dataclass
class ExplainParallelRuntime:
    """Bundle explain-specific parallel settings around the shared executor."""

    executor: ParallelExecutor | None
    min_instances_for_parallel: int
    chunk_size: int
    _start_time: float | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> "ExplainParallelRuntime":
        """Enter the runtime context, initializing the executor if present."""
        self._start_time = time.perf_counter()
        if self.executor:
            self.executor.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the runtime context, updating telemetry and handling cleanup."""
        if self.executor:
            self.executor.__exit__(exc_type, exc_val, exc_tb)

            # Update telemetry
            duration = time.perf_counter() - (self._start_time or 0)
            self.executor.metrics.total_duration += duration

            # Fallback warning
            if (
                self.executor.config.enabled
                and self.executor.active_strategy_name == "sequential"
                and self.executor.config.strategy != "sequential"
            ):
                warnings.warn(
                    "Parallel execution fell back to sequential. Check logs for details.",
                    UserWarning,
                    stacklevel=2,
                )

    def cancel(self) -> None:
        """Cancel any running parallel tasks."""
        if self.executor:
            self.executor.cancel()

    @property
    def telemetry(self) -> Dict[str, Any]:
        """Return current telemetry snapshot."""
        if self.executor:
            return self.executor.metrics.snapshot()
        return {}

    @classmethod
    def from_explainer(
        cls, explainer: Any, *, granularity: str | None = None
    ) -> "ExplainParallelRuntime":
        """Create a runtime wrapper using explainer-provided knobs.

        This inspects the explainer for a provisioned ``ParallelExecutor``
        (via ``_perf_parallel`` or ``executor``), resolves the effective
        granularity, and derives chunking heuristics from the executor
        configuration so explain strategies do not need to reach into the
        parallel package directly.
        """
        executor = getattr(explainer, "parallel_executor", None)
        if executor is None:
            if hasattr(explainer, "__dict__") and "_perf_parallel" in explainer.__dict__:
                executor = explainer.__dict__.get("_perf_parallel")
            if executor is None:
                executor = getattr(explainer, "executor", None)
        if executor is not None and not hasattr(executor, "config"):
            executor = None

        # If no executor is provided by the explainer, create a lightweight
        # in-process ParallelExecutor so instance-parallel plugins can opt-in
        # to their execution path during tests or simple usages. This avoids
        # requiring callers to always provision an executor when selecting the
        # instance-parallel plugin via the plugin config.
        auto_created_executor = False
        if executor is None:
            cfg = ParallelConfig.from_env()
            # Ensure instance granularity when created for explain runtime
            cfg.granularity = "instance"
            # Enable by default so executor-dependent strategies consider
            # the executor available; map() may still choose sequential.
            cfg.enabled = True
            executor = ParallelExecutor(cfg)
            auto_created_executor = True

        parallel_config = executor.config if executor is not None else ParallelConfig()
        # Granularity is deprecated and not captured here.

        chunk_size = max(1, parallel_config.instance_chunk_size or parallel_config.min_batch_size)
        default_min_instances = max(8, chunk_size)
        if auto_created_executor:
            # Tests and simple callers often do not intend to enforce large
            # instance thresholds; when we created the executor automatically
            # for explain runtime, lower the gate so small batches can still
            # exercise the instance-parallel plugin path (map() may still
            # internally choose sequential execution based on workload).
            min_instances = 1
        else:
            min_instances_candidate = getattr(explainer, "min_instances_for_parallel", None)
            if isinstance(min_instances_candidate, numbers.Number) and not isinstance(
                min_instances_candidate, bool
            ):
                min_instances_value = int(min_instances_candidate)
            else:
                min_instances_value = None

            min_instances = max(
                1,
                min_instances_value
                or parallel_config.min_instances_for_parallel
                or default_min_instances,
            )

        return cls(
            executor=executor,
            min_instances_for_parallel=min_instances,
            chunk_size=chunk_size,
        )

    def build_config(self, explainer: Any) -> ExplainConfig:
        """Materialize an :class:`ExplainConfig` for executor dispatch."""
        # Granularity is derived from executor config if present, else defaults to 'none'
        # or whatever ExplainConfig defaults to if we passed None, but ExplainConfig expects str.
        # We use executor config if available.
        granularity = "none"
        if self.executor:
            granularity = self.executor.config.granularity

        return ExplainConfig(
            executor=self.executor,
            granularity=granularity,
            min_instances_for_parallel=self.min_instances_for_parallel,
            chunk_size=self.chunk_size,
            num_features=explainer.num_features,
            features_to_ignore_default=getattr(explainer, "features_to_ignore", ()),
            categorical_features=getattr(explainer, "categorical_features", ()),
            feature_values=getattr(explainer, "feature_values", {}),
            mode=getattr(explainer, "mode", "classification"),
        )


def build_explain_execution_plan(
    explainer: Any, x: Any, request: "ExplanationRequest"
) -> Tuple[ExplainRequest, ExplainConfig, ExplainParallelRuntime]:
    """Prepare explain execution request and config using explain-local rules."""
    prepared_x = validate_and_prepare_input(explainer, x)
    features_to_ignore_array = merge_ignore_features(explainer, request.features_to_ignore)
    feature_filter_per_instance_ignore = getattr(
        request, "feature_filter_per_instance_ignore", None
    )
    if isinstance(feature_filter_per_instance_ignore, Iterable) and not isinstance(
        feature_filter_per_instance_ignore, (str, bytes)
    ):
        merged_per_instance: list[np.ndarray] = []
        for i, inst_mask in enumerate(feature_filter_per_instance_ignore):
            if i >= len(prepared_x):
                break
            inst_arr = as_int_array(inst_mask)
            merged = np.union1d(features_to_ignore_array, inst_arr)
            merged_per_instance.append(merged)
        feature_filter_per_instance_ignore = merged_per_instance
    low_high_percentiles = tuple(request.low_high_percentiles or _DEFAULT_PERCENTILES)

    explain_request = ExplainRequest(
        x=prepared_x,
        threshold=request.threshold,
        low_high_percentiles=low_high_percentiles,
        bins=request.bins,
        features_to_ignore=features_to_ignore_array,
        feature_filter_per_instance_ignore=feature_filter_per_instance_ignore,
        use_plugin=False,
        skip_instance_parallel=False,
    )

    runtime = ExplainParallelRuntime.from_explainer(explainer)
    explain_config = runtime.build_config(explainer)

    return explain_request, explain_config, runtime


__all__ = [
    "ExplainParallelRuntime",
    "build_explain_execution_plan",
]


def worker_init_from_explainer_spec(serialized_spec: dict) -> None:
    """Worker-side initializer that installs a module-global `explain_slice`.

    The function is intentionally small and picklable-friendly: it binds the
    compact explainer spec into a closure and exposes a simple callable that
    accepts `(start, stop, state)` and returns a serialisable payload.
    """
    # Try to rehydrate a full CalibratedExplainer in the worker process using
    # the supplied compact spec. If the spec is minimal, fall back to a small
    # fake explainer sufficient for sequential execution tests.
    try:
        from ...core import calibrated_explainer as ce_module  # type: ignore
        from ...core.explain import _shared as shared_module
        from ...core.explain import sequential as sequential_module
    except (
        Exception
    ):  # ADR002_ALLOW: worker bootstrap must not block on optional deps.  # pragma: no cover
        calibrated_explainer_cls = None  # type: ignore
        sequential_executor_cls = None  # type: ignore
        explain_config_cls = None  # type: ignore
        explain_request_cls = None  # type: ignore
    else:
        calibrated_explainer_cls = getattr(ce_module, "CalibratedExplainer", None)
        explain_config_cls = getattr(shared_module, "ExplainConfig", None)
        explain_request_cls = getattr(shared_module, "ExplainRequest", None)
        sequential_executor_cls = getattr(sequential_module, "SequentialExplainExecutor", None)

    def _build_explainer_from_spec(spec: dict):
        if calibrated_explainer_cls is None:
            return None
        # Support pickled learner bytes for better cross-process transport.
        learner = None
        learner_bytes = spec.get("learner_bytes")
        if learner_bytes is not None:
            try:
                import pickle  # noqa: S403 # nosec B403

                learner = pickle.loads(learner_bytes)  # noqa: S301 # nosec B301
            except Exception:  # ADR002_ALLOW: learner payload is best-effort.  # pragma: no cover
                learner = None
        else:
            learner = spec.get("learner")
        x_cal = spec.get("x_cal")
        y_cal = spec.get("y_cal")
        mode = spec.get("mode", "classification")
        kwargs = {}
        # Construct a CalibratedExplainer instance in the worker. This may be
        # more expensive but keeps worker tasks minimal and avoids shipping
        # the entire explainer object from the parent process.
        try:
            return calibrated_explainer_cls(learner, x_cal, y_cal, mode=mode, **kwargs)
        except (
            Exception
        ):  # ADR002_ALLOW: spec may be incomplete; fall back to None.  # pragma: no cover
            return None

    worker_explainer = _build_explainer_from_spec(serialized_spec or {})

    def explain_slice(start: int, stop: int, state: Any):
        # state is expected to contain 'subset' when running under the
        # pool-at-init initializer path; fallback to minimal behaviour if not.
        subset = state.get("subset") if isinstance(state, dict) else None
        cfg_state = state.get("config_state") if isinstance(state, dict) else {}

        if (
            worker_explainer is not None
            and sequential_executor_cls is not None
            and explain_config_cls is not None
            and explain_request_cls is not None
        ):
            # Build ExplainConfig and ExplainRequest locally and run sequential executor
            config = (
                explain_config_cls(**cfg_state)
                if cfg_state is not None
                else explain_config_cls(executor=None)
            )
            chunk_request = explain_request_cls(
                x=subset,
                threshold=state.get("threshold_slice") if isinstance(state, dict) else None,
                low_high_percentiles=state.get("low_high_percentiles")
                if isinstance(state, dict)
                else None,
                bins=state.get("bins_slice") if isinstance(state, dict) else None,
                features_to_ignore=state.get("features_to_ignore_array")
                if isinstance(state, dict)
                else None,
                feature_filter_per_instance_ignore=state.get("feature_filter_per_instance_ignore")
                if isinstance(state, dict)
                else None,
                use_plugin=False,
                skip_instance_parallel=True,
            )
            plugin = sequential_executor_cls()
            return plugin.execute(chunk_request, config, worker_explainer)

        # Fallback toy payload for environments where we can't rehydrate
        return {"spec": serialized_spec, "start": start, "stop": stop, "state": state}

    globals()["explain_slice"] = explain_slice
    globals()["_worker_harness"] = types.SimpleNamespace(explain_slice=explain_slice)
    globals()["worker_harness"] = globals()["_worker_harness"]
