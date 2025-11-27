"""LIME integration pipeline for calibrated explanations.

This module provides the LimePipeline class which orchestrates LIME-based
explanation generation through delegated execution.
"""

# pylint: disable=protected-access

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from calibrated_explanations.core.exceptions import (
    ConfigurationError,
    DataShapeError,
    ValidationError,
)
from calibrated_explanations.explanations import CalibratedExplanations
from calibrated_explanations.integrations.lime import LimeHelper
from calibrated_explanations.utils.helper import (
    assert_threshold,
    safe_isinstance,
)

if TYPE_CHECKING:
    from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


class LimePipeline:
    """Pipeline for generating LIME-based explanations.

    This class handles the orchestration of LIME explanation generation,
    including LIME explainer initialization and local approximation.

    Attributes
    ----------
    explainer : CalibratedExplainer
        The parent explainer instance.
    """

    def __init__(self, explainer: CalibratedExplainer) -> None:
        """Initialize the LIME explanation pipeline.

        Parameters
        ----------
        explainer : CalibratedExplainer
            The parent explainer instance.
        """
        self.explainer = explainer
        self._lime_helper: LimeHelper | None = None

    def _is_lime_enabled(self, is_enabled: bool | None = None) -> bool:
        """Return whether LIME export is enabled.

        Parameters
        ----------
        is_enabled : bool, optional
            If provided, set the enabled state.

        Returns
        -------
        bool
            Whether LIME is currently enabled.
        """
        if self._lime_helper is None:
            self._lime_helper = LimeHelper(self.explainer)
        if is_enabled is not None:
            self._lime_helper.set_enabled(bool(is_enabled))
        return self._lime_helper.is_enabled()

    def _preload_lime(self, x_cal: Any = None) -> tuple[Any, Any]:
        """Materialize LIME explainer artifacts when the dependency is available.

        Parameters
        ----------
        x_cal : array-like, optional
            Calibration data to use for LIME initialization. If not provided,
            uses the explainer's calibration data.

        Returns
        -------
        tuple
            A tuple of (lime_explainer, reference_explanation) or (None, None)
            if LIME is not available.
        """
        if self._lime_helper is None:
            self._lime_helper = LimeHelper(self.explainer)
        return self._lime_helper.preload(x_cal=x_cal)

    def explain(
        self,
        x_test: Any,
        threshold: float | None = None,
        low_high_percentiles: tuple[float, float] = (5, 95),
        bins: Any = None,
    ) -> CalibratedExplanations:
        """Generate LIME explanations for the given instances.

        Parameters
        ----------
        x_test : array-like
            A set with n_samples of test objects to predict.
        threshold : float, int or array-like of shape (n_samples,), default=None
            Values for which p-values should be returned. Only used for
            probabilistic explanations for regression.
        low_high_percentiles : tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval.
            Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories.

        Returns
        -------
        CalibratedExplanations
            A `CalibratedExplanations` containing one explanation for
            each instance.

        Raises
        ------
        ValidationError
            If parameters are invalid or inconsistent with the explainer's
            configuration.
        DataShapeError
            If the input shape doesn't match the explainer's feature count.
        ConfigurationError
            If LIME is not properly configured or dependencies are missing.
        """
        # Preload LIME explainer using pipeline's helper
        lime_explainer, _ = self._preload_lime()

        # Measure total execution time
        total_time = time()
        instance_time = []

        # Normalize input
        if safe_isinstance(x_test, "pandas.core.frame.DataFrame"):
            x_test = x_test.values  # pylint: disable=invalid-name
        if len(x_test.shape) == 1:
            x_test = x_test.reshape(1, -1)

        # Validate input shape
        if x_test.shape[1] != self.explainer.num_features:
            raise DataShapeError(
                "The number of features in the test data must be the same as in the "
                "calibration data."
            )

        # Validate Mondrian categories if applicable
        if self.explainer._is_mondrian():
            if bins is None:
                raise ValidationError(
                    "The bins parameter must be specified for Mondrian explanations."
                )
            if len(bins) != len(x_test):
                raise DataShapeError(
                    "The length of the bins parameter must be the same as the "
                    "number of instances in x."
                )

        # Create explanation object
        explanation = CalibratedExplanations(self.explainer, x_test, threshold, bins)

        # Validate and set threshold if provided
        if threshold is not None:
            if "regression" not in self.explainer.mode:
                raise ValidationError(
                    "The threshold parameter is only supported for mode='regression'."
                )
            assert_threshold(threshold, x_test)
        elif "regression" in self.explainer.mode:
            explanation.low_high_percentiles = low_high_percentiles

        # Initialize feature containers
        feature_weights: Dict[str, List[np.ndarray]] = {
            "predict": [],
            "low": [],
            "high": [],
        }
        feature_predict: Dict[str, List[np.ndarray]] = {
            "predict": [],
            "low": [],
            "high": [],
        }
        prediction: Dict[str, Any] = {"predict": [], "low": [], "high": [], "classes": []}

        instance_weights = [
            {
                "predict": np.zeros(self.explainer.num_features),
                "low": np.zeros(self.explainer.num_features),
                "high": np.zeros(self.explainer.num_features),
            }
            for _ in range(len(x_test))
        ]
        instance_predict = [
            {
                "predict": np.zeros(self.explainer.num_features),
                "low": np.zeros(self.explainer.num_features),
                "high": np.zeros(self.explainer.num_features),
            }
            for _ in range(len(x_test))
        ]

        # Get predictions for the instances
        predict, low, high, predicted_class = self.explainer._predict(
            x_test, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins
        )
        prediction["predict"] = predict
        prediction["low"] = low
        prediction["high"] = high
        if self.explainer.is_multiclass():
            prediction["classes"] = predicted_class
        else:
            prediction["classes"] = np.ones(x_test.shape[0])

        # Verify LIME is available
        if lime_explainer is None:
            raise ConfigurationError(
                "LIME integration requested but the optional dependency is missing."
            )

        # Define probability functions for LIME
        def low_proba(x_data):
            _, low_vals, _, _ = self.explainer._predict(
                x_data,
                threshold=threshold,
                low_high_percentiles=low_high_percentiles,
                bins=bins,
            )
            return np.asarray([[1 - l, l] for l in low_vals])  # noqa: E741

        def high_proba(x_data):
            _, _, high_vals, _ = self.explainer._predict(
                x_data,
                threshold=threshold,
                low_high_percentiles=low_high_percentiles,
                bins=bins,
            )
            return np.asarray([[1 - h, h] for h in high_vals])  # noqa: E741

        # Initialize result structure
        res_struct: Dict[str, Dict[str, Any]] = {}
        res_struct["low"] = {}
        res_struct["high"] = {}
        res_struct["low"]["explanation"], res_struct["high"]["explanation"] = [], []
        res_struct["low"]["abs_rank"], res_struct["high"]["abs_rank"] = [], []
        res_struct["low"]["values"], res_struct["high"]["values"] = [], []

        # Generate LIME explanations for each instance
        for idx, instance in enumerate(x_test):
            instance_timer = time()

            assert lime_explainer is not None
            low_explanation = lime_explainer.explain_instance(
                instance, predict_fn=low_proba, num_features=len(instance)
            )
            high_explanation = lime_explainer.explain_instance(
                instance, predict_fn=high_proba, num_features=len(instance)
            )

            res_struct["low"]["explanation"].append(low_explanation)
            res_struct["high"]["explanation"].append(high_explanation)
            res_struct["low"]["abs_rank"], res_struct["high"]["abs_rank"] = (
                np.zeros(len(instance)),
                np.zeros(len(instance)),
            )
            res_struct["low"]["values"], res_struct["high"]["values"] = (
                np.zeros(len(instance)),
                np.zeros(len(instance)),
            )

            # Extract feature importances from LIME
            for j, feat_info in enumerate(low_explanation.local_exp[1]):
                res_struct["low"]["abs_rank"][feat_info[0]] = low_explanation.local_exp[1][j][0]
                res_struct["low"]["values"][feat_info[0]] = feat_info[1]

            for j, feat_info in enumerate(high_explanation.local_exp[1]):
                res_struct["high"]["abs_rank"][feat_info[0]] = high_explanation.local_exp[1][j][0]
                res_struct["high"]["values"][feat_info[0]] = feat_info[1]

            # Aggregate feature importances into weights
            for feat_idx in range(self.explainer.num_features):
                tmp_low = res_struct["low"]["values"][feat_idx]
                tmp_high = res_struct["high"]["values"][feat_idx]
                instance_weights[idx]["low"][feat_idx] = np.min([tmp_low, tmp_high])
                instance_weights[idx]["high"][feat_idx] = np.max([tmp_low, tmp_high])
                instance_weights[idx]["predict"][feat_idx] = instance_weights[idx]["high"][
                    feat_idx
                ] / (
                    1
                    - instance_weights[idx]["low"][feat_idx]
                    + instance_weights[idx]["high"][feat_idx]
                )

                instance_predict[idx]["low"][feat_idx] = (
                    low_explanation.predict_proba[-1] - instance_weights[idx]["low"][feat_idx]
                )
                instance_predict[idx]["high"][feat_idx] = (
                    high_explanation.predict_proba[-1] - instance_weights[idx]["high"][feat_idx]
                )
                instance_predict[idx]["predict"][feat_idx] = instance_predict[idx]["high"][
                    feat_idx
                ] / (
                    1
                    - instance_predict[idx]["low"][feat_idx]
                    + instance_predict[idx]["high"][feat_idx]
                )

            feature_weights["predict"].append(instance_weights[idx]["predict"])
            feature_weights["low"].append(instance_weights[idx]["low"])
            feature_weights["high"].append(instance_weights[idx]["high"])

            feature_predict["predict"].append(instance_predict[idx]["predict"])
            feature_predict["low"].append(instance_predict[idx]["low"])
            feature_predict["high"].append(instance_predict[idx]["high"])
            instance_time.append(time() - instance_timer)

        # Finalize explanation with computed features
        explanation.finalize_fast(
            feature_weights,
            feature_predict,
            prediction,
            instance_time=instance_time,
            total_time=total_time,
        )

        # Update explainer state
        self.explainer.latest_explanation = explanation

        return explanation
