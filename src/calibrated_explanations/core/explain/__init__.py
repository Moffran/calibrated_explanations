"""Explain plugin system for calibrated explanations.

This package provides a plugin-based architecture for explain execution strategies:
- Sequential: single-threaded feature-by-feature processing
- Feature-parallel: parallel processing across features  
- Instance-parallel: parallel processing across instances

The plugin system replaces branching logic in CalibratedExplainer.explain,
providing clean separation between orchestration and execution strategies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ._base import BaseExplainPlugin
from ._shared import ExplainConfig, ExplainRequest, ExplainResponse
from .parallel_feature import FeatureParallelExplainPlugin
from .parallel_instance import InstanceParallelExplainPlugin
from .sequential import SequentialExplainPlugin

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer

# Global plugin registry (initialized on first import)
_REGISTERED_PLUGINS: List[BaseExplainPlugin] = [
    InstanceParallelExplainPlugin(),  # Priority 30: check first
    FeatureParallelExplainPlugin(),   # Priority 20: check second
    SequentialExplainPlugin(),        # Priority 10: fallback
]


def select_plugin(
    request: ExplainRequest,
    config: ExplainConfig,
) -> BaseExplainPlugin:
    """Select the appropriate explain plugin based on request and config.
    
    Plugins are checked in priority order (highest first). The first plugin
    that returns True from its supports() method is selected.
    
    Parameters
    ----------
    request : ExplainRequest
        The explain request context
    config : ExplainConfig
        Execution configuration with executor and granularity
        
    Returns
    -------
    BaseExplainPlugin
        The selected plugin (sequential if no others support)
        
    Raises
    ------
    ValueError
        If conflicting parallelism settings are detected
    """
    # Validate configuration for conflicts
    if config.executor is not None and config.executor.config.enabled:
        # Check for mixed granularity (both feature and instance parallel requested)
        # This should be caught by plugin supports() methods, but add explicit check
        if config.granularity not in ("feature", "instance", "none"):
            raise ValueError(
                f"Invalid parallelism granularity: {config.granularity}. "
                "Must be 'feature', 'instance', or 'none'."
            )

    # Check plugins in priority order
    for plugin in sorted(_REGISTERED_PLUGINS, key=lambda p: p.priority, reverse=True):
        if plugin.supports(request, config):
            return plugin

    # This should never happen since sequential plugin always supports,
    # but provide a defensive fallback
    return SequentialExplainPlugin()


def explain(
    explainer: CalibratedExplainer,
    x,
    threshold=None,
    low_high_percentiles=(5, 95),
    bins=None,
    features_to_ignore=None,
    *,
    _use_plugin: bool = True,
    _skip_instance_parallel: bool = False,
):
    """Execute explain operation using the plugin system.
    
    This is the thin delegator that replaces the monolithic explain method.
    It builds request/config context and delegates to the selected plugin.
    
    Parameters
    ----------
    explainer : CalibratedExplainer
        The explainer instance providing model access
    x : array-like
        Test instances to explain
    threshold : optional
        Threshold for binary classification or regression intervals
    low_high_percentiles : tuple
        Percentile bounds for uncertainty intervals
    bins : optional
        Difficulty bins for conditional calibration
    features_to_ignore : optional
        Feature indices to skip during perturbation
    _use_plugin : bool, default=True
        Whether to use plugin system (for backward compatibility)
    _skip_instance_parallel : bool, default=False
        Prevent recursive instance parallelism
        
    Returns
    -------
    CalibratedExplanations
        The completed explanation collection
    """
    from ._helpers import merge_ignore_features

    # Merge features to ignore
    features_to_ignore_array = merge_ignore_features(explainer, features_to_ignore)

    # Validate and prepare input
    x_validated = explainer._validate_and_prepare_input(x)

    # Build request context
    request = ExplainRequest(
        x=x_validated,
        threshold=threshold,
        low_high_percentiles=low_high_percentiles,
        bins=bins,
        features_to_ignore=features_to_ignore_array,
        use_plugin=_use_plugin,
        skip_instance_parallel=_skip_instance_parallel,
    )

    # Build execution config
    executor = getattr(explainer, "_perf_parallel", None)
    granularity = "none"
    if executor is not None and executor.config.enabled:
        granularity = getattr(executor.config, "granularity", "feature")

    config = ExplainConfig(
        executor=executor,
        granularity=granularity,
        min_instances_for_parallel=4,  # Could be configurable
        chunk_size=getattr(executor.config, "min_batch_size", 100) if executor else 100,
        num_features=explainer.num_features,
        features_to_ignore_default=list(explainer.features_to_ignore),
        categorical_features=list(explainer.categorical_features),
        feature_values=dict(explainer.feature_values),
        mode=explainer.mode,
    )

    # Select and execute plugin
    plugin = select_plugin(request, config)
    return plugin.execute(request, config, explainer)


__all__ = [
    "BaseExplainPlugin",
    "ExplainConfig",
    "ExplainRequest",
    "ExplainResponse",
    "FeatureParallelExplainPlugin",
    "InstanceParallelExplainPlugin",
    "SequentialExplainPlugin",
    "explain",
    "select_plugin",
]
