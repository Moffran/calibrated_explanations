"""SHAP integration pipeline for calibrated explanations.

This module provides the ShapPipeline class which orchestrates SHAP-based
explanation generation through delegated execution.
"""

# pylint: disable=protected-access

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from calibrated_explanations.utils.exceptions import ConfigurationError
from calibrated_explanations.integrations.shap import ShapHelper

if TYPE_CHECKING:
    from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


class ShapPipeline:
    """Pipeline for generating SHAP-based explanations.

    This class handles the orchestration of SHAP explanation generation,
    including SHAP explainer initialization and preloading.

    Attributes
    ----------
    explainer : CalibratedExplainer
        The parent explainer instance.
    """

    def __init__(self, explainer: CalibratedExplainer) -> None:
        """Initialize the SHAP explanation pipeline.

        Parameters
        ----------
        explainer : CalibratedExplainer
            The parent explainer instance.
        """
        self.explainer = explainer
        self._shap_helper: ShapHelper | None = None

    def _is_shap_enabled(self, is_enabled: bool | None = None) -> bool:
        """Return whether SHAP export is enabled.

        Parameters
        ----------
        is_enabled : bool, optional
            If provided, set the enabled state.

        Returns
        -------
        bool
            Whether SHAP is currently enabled.
        """
        if self._shap_helper is None:
            self._shap_helper = ShapHelper(self.explainer)
        if is_enabled is not None:
            self._shap_helper.set_enabled(bool(is_enabled))
        return self._shap_helper.is_enabled()

    def _preload_shap(self, num_test: int | None = None) -> tuple[Any, Any]:
        """Eagerly compute SHAP explanations to amortize repeated requests.

        Parameters
        ----------
        num_test : int, optional
            Number of test instances to explain. If not provided, uses a single
            reference instance from the calibration data.

        Returns
        -------
        tuple
            A tuple of (shap_explainer, reference_explanation) or (None, None)
            if SHAP is not available.

        Raises
        ------
        ConfigurationError
            If SHAP is requested but the optional dependency is missing.
        """
        if self._shap_helper is None:
            self._shap_helper = ShapHelper(self.explainer)
        return self._shap_helper.preload(num_test=num_test)

    def explain(self, x_test: Any, **kwargs) -> Any:
        """Generate SHAP explanations for the given instances.

        Parameters
        ----------
        x_test : array-like
            A set with n_samples of test objects to explain.
        **kwargs
            Additional keyword arguments passed through to SHAP.

        Returns
        -------
        Any
            SHAP explanation object.

        Raises
        ------
        ConfigurationError
            If SHAP is not properly configured or dependencies are missing.
        """
        shap_explainer, _ = self._preload_shap(num_test=len(x_test))

        if shap_explainer is None:
            raise ConfigurationError(
                "SHAP integration requested but the optional dependency is missing."
            )

        # Return SHAP explanations for all instances
        return shap_explainer(x_test, **kwargs)
