"""Base plugin interface for explain execution strategies.

This module defines the abstract protocol that all explain plugins
(sequential, feature-parallel, instance-parallel) must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...explanations import CalibratedExplanations
    from ..calibrated_explainer import CalibratedExplainer
    from ._shared import ExplainConfig, ExplainRequest


class BaseExplainPlugin(ABC):
    """Abstract base for explain execution strategy plugins.

    Each plugin implements a specific execution strategy:
    - Sequential: single-threaded feature-by-feature processing
    - Feature-parallel: parallel processing across features
    - Instance-parallel: parallel processing across instances

    The plugin system replaces branching logic in CalibratedExplainer.explain,
    providing clean separation between orchestration and execution.
    """

    @abstractmethod
    def supports(self, request: ExplainRequest, config: ExplainConfig) -> bool:
        """Return True if this plugin can handle the given request and config.

        Plugins use this predicate to declare their applicability based on:
        - Executor availability and configuration
        - Parallelism granularity settings
        - Minimum instance/feature thresholds
        - Control flags (skip_instance_parallel, etc.)

        Parameters
        ----------
        request : ExplainRequest
            The explain request context
        config : ExplainConfig
            Execution configuration with executor and granularity

        Returns
        -------
        bool
            True if plugin can execute this request
        """
        ...

    @abstractmethod
    def execute(
        self,
        request: ExplainRequest,
        config: ExplainConfig,
        explainer: CalibratedExplainer,
    ) -> CalibratedExplanations:
        """Execute the explain operation using this plugin's strategy.

        The plugin is responsible for:
        1. Validating input compatibility
        2. Orchestrating prediction and perturbation phases
        3. Managing parallelism (if applicable)
        4. Assembling the final CalibratedExplanations result
        5. Updating explainer state (latest_explanation, timers)

        Parameters
        ----------
        request : ExplainRequest
            The explain request with input data and parameters
        config : ExplainConfig
            Execution configuration with executor and settings
        explainer : CalibratedExplainer
            The explainer instance providing model access and helpers

        Returns
        -------
        CalibratedExplanations
            The completed explanation collection

        Raises
        ------
        ValueError
            If request parameters are invalid
        RuntimeError
            If execution encounters unrecoverable errors
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the plugin's identifying name.

        Used for logging, telemetry, and debugging.
        """
        ...

    @property
    @abstractmethod
    def priority(self) -> int:
        """Return the plugin's selection priority.

        Higher priority plugins are checked first during plugin selection.
        Recommended values:
        - Instance-parallel: 30 (check first, most specific)
        - Feature-parallel: 20 (check second, specific)
        - Sequential: 10 (fallback, always supports)
        """
        ...


__all__ = ["BaseExplainPlugin"]
