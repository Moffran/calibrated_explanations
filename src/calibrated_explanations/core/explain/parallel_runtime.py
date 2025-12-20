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
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Tuple
import types

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
                and self.executor._active_strategy_name == "sequential"
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
        executor = getattr(explainer, "_perf_parallel", None) or getattr(
            explainer, "executor", None
        )
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
            min_instances = max(
                1,
                getattr(explainer, "min_instances_for_parallel", None)
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
    features_to_ignore_per_instance = getattr(request, "features_to_ignore_per_instance", None)
    if isinstance(features_to_ignore_per_instance, Iterable) and not isinstance(
        features_to_ignore_per_instance, (str, bytes)
    ):
        merged_per_instance: list[np.ndarray] = []
        for i, inst_mask in enumerate(features_to_ignore_per_instance):
            if i >= len(prepared_x):
                break
            inst_arr = as_int_array(inst_mask)
            merged = np.union1d(features_to_ignore_array, inst_arr)
            merged_per_instance.append(merged)
        features_to_ignore_per_instance = merged_per_instance
    low_high_percentiles = tuple(request.low_high_percentiles or _DEFAULT_PERCENTILES)

    explain_request = ExplainRequest(
        x=prepared_x,
        threshold=request.threshold,
        low_high_percentiles=low_high_percentiles,
        bins=request.bins,
        features_to_ignore=features_to_ignore_array,
        features_to_ignore_per_instance=features_to_ignore_per_instance,
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
        from ...core.calibrated_explainer import CalibratedExplainer  # type: ignore
        from ...core.explain.sequential import SequentialExplainExecutor
        from ...core.explain._shared import ExplainConfig, ExplainRequest
    except Exception:
        CalibratedExplainer = None  # type: ignore
        SequentialExplainExecutor = None  # type: ignore
        ExplainConfig = None  # type: ignore
        ExplainRequest = None  # type: ignore

    def _build_explainer_from_spec(spec: dict):
        if CalibratedExplainer is None:
            return None
        # Support pickled learner bytes for better cross-process transport.
        learner = None
        learner_bytes = spec.get("learner_bytes")
        if learner_bytes is not None:
            try:
                import pickle

                learner = pickle.loads(learner_bytes)
            except Exception:
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
            return CalibratedExplainer(learner, x_cal, y_cal, mode=mode, **kwargs)
        except Exception:
            return None

    worker_explainer = _build_explainer_from_spec(serialized_spec or {})

    def explain_slice(start: int, stop: int, state: Any):
        # state is expected to contain 'subset' when running under the
        # pool-at-init initializer path; fallback to minimal behaviour if not.
        subset = state.get("subset") if isinstance(state, dict) else None
        cfg_state = state.get("config_state") if isinstance(state, dict) else {}

        if worker_explainer is not None and SequentialExplainExecutor is not None:
            # Build ExplainConfig and ExplainRequest locally and run sequential executor
            config = ExplainConfig(**cfg_state) if cfg_state is not None else ExplainConfig(executor=None)
            chunk_request = ExplainRequest(
                x=subset,
                threshold=state.get("threshold_slice") if isinstance(state, dict) else None,
                low_high_percentiles=state.get("low_high_percentiles") if isinstance(state, dict) else None,
                bins=state.get("bins_slice") if isinstance(state, dict) else None,
                features_to_ignore=state.get("features_to_ignore_array") if isinstance(state, dict) else None,
                features_to_ignore_per_instance=state.get("features_to_ignore_per_instance") if isinstance(state, dict) else None,
                use_plugin=False,
                skip_instance_parallel=True,
            )
            plugin = SequentialExplainExecutor()
            return plugin.execute(chunk_request, config, worker_explainer)

        # Fallback toy payload for environments where we can't rehydrate
        return {"spec": serialized_spec, "start": start, "stop": stop, "state": state}

    globals()["explain_slice"] = explain_slice
    globals()["_worker_harness"] = types.SimpleNamespace(explain_slice=explain_slice)
