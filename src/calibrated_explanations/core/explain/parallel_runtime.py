"""Explain-specific parallel runtime wiring.

This module keeps explain-facing heuristics and chunking decisions co-located
with the explain executors while delegating low-level execution to the shared
``calibrated_explanations.parallel`` facade. It wraps the generic
``ParallelExecutor`` with explain-aware defaults (granularity, minimum
instance thresholds, and chunk sizing) so that explain strategies avoid
sprinkling parallel plumbing across sibling packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

from ...parallel import ParallelConfig, ParallelExecutor
from ._helpers import merge_ignore_features, validate_and_prepare_input
from ._shared import ExplainConfig, ExplainRequest

if TYPE_CHECKING:
    from ...plugins import ExplanationRequest

_DEFAULT_PERCENTILES: Tuple[float, float] = (5, 95)


@dataclass
class ExplainParallelRuntime:
    """Bundle explain-specific parallel settings around the shared executor."""

    executor: ParallelExecutor | None
    granularity: str
    min_instances_for_parallel: int
    chunk_size: int

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

        parallel_config = executor.config if executor is not None else ParallelConfig()
        resolved_granularity = (
            granularity or getattr(explainer, "granularity", None) or parallel_config.granularity
        )
        chunk_size = max(1, parallel_config.min_batch_size)
        min_instances = max(
            1,
            getattr(explainer, "min_instances_for_parallel", 0) or chunk_size,
        )

        return cls(
            executor=executor,
            granularity=resolved_granularity,
            min_instances_for_parallel=min_instances,
            chunk_size=chunk_size,
        )

    def build_config(self, explainer: Any) -> ExplainConfig:
        """Materialize an :class:`ExplainConfig` for executor dispatch."""
        return ExplainConfig(
            executor=self.executor,
            granularity=self.granularity if self.executor is not None else "none",
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
) -> Tuple[ExplainRequest, ExplainConfig]:
    """Prepare explain execution request and config using explain-local rules."""
    prepared_x = validate_and_prepare_input(explainer, x)
    features_to_ignore_array = merge_ignore_features(explainer, request.features_to_ignore)
    low_high_percentiles = tuple(request.low_high_percentiles or _DEFAULT_PERCENTILES)

    explain_request = ExplainRequest(
        x=prepared_x,
        threshold=request.threshold,
        low_high_percentiles=low_high_percentiles,
        bins=request.bins,
        features_to_ignore=features_to_ignore_array,
        use_plugin=False,
        skip_instance_parallel=False,
    )

    runtime = ExplainParallelRuntime.from_explainer(
        explainer, granularity=getattr(explainer, "granularity", None)
    )
    explain_config = runtime.build_config(explainer)

    return explain_request, explain_config


__all__ = [
    "ExplainParallelRuntime",
    "build_explain_execution_plan",
]
