"""Fast explanation pipeline for delegated execution from CalibratedExplainer.

This module contains the FastExplanationPipeline class which orchestrates
fast explanation generation, including perturbation, noise injection,
discretization, and rule extraction.
"""

# pylint: disable=protected-access

from __future__ import annotations

from contextlib import suppress
from time import time
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from calibrated_explanations.core.explain._helpers import compute_feature_effects
from calibrated_explanations.explanations import CalibratedExplanations
from calibrated_explanations.utils import assert_threshold, safe_isinstance
from calibrated_explanations.utils.exceptions import (
    ConfigurationError,
    DataShapeError,
    ValidationError,
)

if TYPE_CHECKING:
    from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


class FastExplanationPipeline:
    """Pipeline for generating fast explanations.

    This class handles the orchestration of fast explanation generation,
    delegating computation of feature effects and prediction intervals.

    Attributes
    ----------
    explainer : CalibratedExplainer
        The parent explainer instance.
    """

    def __init__(self, explainer: CalibratedExplainer) -> None:
        """Initialize the fast explanation pipeline.

        Parameters
        ----------
        explainer : CalibratedExplainer
            The parent explainer instance.
        """
        self.explainer = explainer

    def explain(
        self,
        x_test: Any,
        threshold: float | None = None,
        low_high_percentiles: tuple[float, float] = (5, 95),
        bins: Any = None,
    ) -> CalibratedExplanations:
        """Generate fast explanations for the given instances.

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
            A `CalibratedExplanations` containing one `FastExplanation` for
            each instance.

        Raises
        ------
        ValidationError
            If parameters are invalid or inconsistent with the explainer's
            configuration.
        DataShapeError
            If the input shape doesn't match the explainer's feature count.
        ConfigurationError
            If fast explanations are not properly configured.
        """
        # Initialize fast mode if not already done
        if not self.explainer.is_fast():
            try:
                self.explainer.enable_fast_mode()
            except Exception as exc:
                raise ConfigurationError(
                    "Fast explanations are only possible if the explainer is a "
                    "Fast Calibrated Explainer."
                ) from exc

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
        if self.explainer.is_mondrian():  # pylint: disable=protected-access
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
        explanation = CalibratedExplanations(
            self.explainer,
            x_test,
            threshold,
            bins,
            condition_source=getattr(self.explainer, "condition_source", "observed"),
        )

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

        # Measure feature computation time
        feature_time = time()

        # Get predictions for the instances
        predict, low, high, predicted_class = self.explainer.predict_calibrated(  # pylint: disable=protected-access
            x_test,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            bins=bins,
        )
        prediction: Dict[str, Any] = {
            "predict": predict,
            "low": low,
            "high": high,
            "classes": (
                predicted_class if self.explainer.is_multiclass() else np.ones(x_test.shape[0])
            ),
        }

        if self.explainer.mode == "classification":
            with suppress(Exception):
                if self.explainer.is_multiclass():
                    full_probs = self.explainer.interval_learner[  # pylint: disable=protected-access
                        self.explainer.num_features
                    ].predict_proba(x_test, bins=bins)
                else:
                    full_probs = self.explainer.interval_learner[  # pylint: disable=protected-access
                        self.explainer.num_features
                    ].predict_proba(x_test, bins=bins)
                prediction["__full_probabilities__"] = full_probs

        # Temporarily swap calibration targets for feature computation
        y_cal = self.explainer.y_cal
        self.explainer.y_cal = self.explainer.scaled_y_cal

        # Get features to process (excluding ignored features)
        features_to_process = [
            feat
            for feat in range(self.explainer.num_features)
            if feat not in self.explainer.features_to_ignore
        ]

        # Compute feature effects with optional parallelization
        executor = getattr(self.explainer, "_perf_parallel", None)
        if (
            executor is not None
            and executor.config.enabled
            and getattr(executor.config, "granularity", "feature") == "feature"
        ):
            feature_results = compute_feature_effects(
                self.explainer,
                features_to_process,
                x_test,
                threshold,
                low_high_percentiles,
                bins,
                prediction,
                executor,
            )
        else:
            feature_results = compute_feature_effects(
                self.explainer,
                features_to_process,
                x_test,
                threshold,
                low_high_percentiles,
                bins,
                prediction,
                None,
            )

        # Aggregate feature results into instance-level weights
        for (
            feature_index,
            weights_predict,
            weights_low,
            weights_high,
            local_predict,
            local_low,
            local_high,
        ) in feature_results:
            for i in range(len(x_test)):
                instance_weights[i]["predict"][feature_index] = weights_predict[i]
                instance_weights[i]["low"][feature_index] = weights_low[i]
                instance_weights[i]["high"][feature_index] = weights_high[i]
                instance_predict[i]["predict"][feature_index] = local_predict[i]
                instance_predict[i]["low"][feature_index] = local_low[i]
                instance_predict[i]["high"][feature_index] = local_high[i]

        # Restore original calibration targets
        self.explainer.y_cal = y_cal

        # Flatten results for explanation object
        for i in range(len(x_test)):
            feature_weights["predict"].append(instance_weights[i]["predict"])
            feature_weights["low"].append(instance_weights[i]["low"])
            feature_weights["high"].append(instance_weights[i]["high"])

            feature_predict["predict"].append(instance_predict[i]["predict"])
            feature_predict["low"].append(instance_predict[i]["low"])
            feature_predict["high"].append(instance_predict[i]["high"])

        # Compute timing
        feature_time = time() - feature_time
        instance_time = [feature_time / x_test.shape[0]] * x_test.shape[0]

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
        self.explainer.last_explanation_mode = "fast"  # pylint: disable=protected-access

        return explanation

    def preprocess(self) -> None:
        """Identify constant calibration features that can be ignored downstream.

        Updates the explainer's `features_to_ignore` list with indices of
        features that have constant values in the calibration data.
        """
        constant_columns = [
            f
            for f in range(self.explainer.num_features)
            if np.all(self.explainer.x_cal[:, f] == self.explainer.x_cal[0, f])
        ]
        self.explainer.features_to_ignore = constant_columns

    def discretize(self, x_data: Any) -> np.ndarray:
        """Apply the discretizer to the data sample x_data.

        For new data samples and missing values, the nearest bin is used.

        Parameters
        ----------
        x_data : array-like
            The data sample to discretize.

        Returns
        -------
        np.ndarray
            The discretized data sample.
        """
        x_data = np.array(x_data, copy=True)  # Ensure x_data is a numpy array
        for feature_idx in self.explainer.discretizer.to_discretize:
            bins = np.concatenate(
                ([-np.inf], self.explainer.discretizer.mins[feature_idx][1:], [np.inf])
            )
            bin_indices = np.digitize(x_data[:, feature_idx], bins, right=True) - 1
            means = np.asarray(self.explainer.discretizer.means[feature_idx])
            bin_indices = np.clip(bin_indices, 0, len(means) - 1)
            x_data[:, feature_idx] = means[bin_indices]
        return x_data

    def rule_boundaries(
        self,
        instances: Any,
        perturbed_instances: Any = None,
    ) -> np.ndarray:
        """Extract the rule boundaries for a set of instances.

        Computes the minimum and maximum feature values for each instance
        based on discretized bins.

        Parameters
        ----------
        instances : array-like
            The instances to extract boundaries for.
        perturbed_instances : array-like, optional
            Discretized versions of instances. Defaults to None.

        Returns
        -------
        np.ndarray
            Min and max values for each feature for each instance.
        """
        # Backward compatibility for single instance
        if len(instances.shape) == 1:
            min_max = []
            if perturbed_instances is None:
                perturbed_instances = self.discretize(instances.reshape(1, -1))
            for feature_idx in range(self.explainer.num_features):
                if feature_idx not in self.explainer.discretizer.to_discretize:
                    min_max.append([instances[feature_idx], instances[feature_idx]])
                else:
                    bins = np.concatenate(
                        (
                            [-np.inf],
                            self.explainer.discretizer.mins[feature_idx][1:],
                            [np.inf],
                        )
                    )
                    min_max.append(
                        [
                            self.explainer.discretizer.mins[feature_idx][
                                np.digitize(perturbed_instances[0, feature_idx], bins, right=True)
                                - 1
                            ],
                            self.explainer.discretizer.maxs[feature_idx][
                                np.digitize(perturbed_instances[0, feature_idx], bins, right=True)
                                - 1
                            ],
                        ]
                    )
            return min_max

        # Batch processing
        instances = np.array(instances)  # Ensure instances is a numpy array
        if perturbed_instances is None:
            perturbed_instances = self.discretize(instances)
        else:
            perturbed_instances = np.array(
                perturbed_instances
            )  # Ensure perturbed_instances is a numpy array

        all_min_max = []
        for instance, perturbed_instance in zip(instances, perturbed_instances, strict=False):
            min_max = []
            for feature_idx in range(self.explainer.num_features):
                if feature_idx not in self.explainer.discretizer.to_discretize:
                    min_max.append([instance[feature_idx], instance[feature_idx]])
                else:
                    bins = np.concatenate(
                        (
                            [-np.inf],
                            self.explainer.discretizer.mins[feature_idx][1:],
                            [np.inf],
                        )
                    )
                    min_max.append(
                        [
                            self.explainer.discretizer.mins[feature_idx][
                                np.digitize(perturbed_instance[feature_idx], bins, right=True) - 1
                            ],
                            self.explainer.discretizer.maxs[feature_idx][
                                np.digitize(perturbed_instance[feature_idx], bins, right=True) - 1
                            ],
                        ]
                    )
            all_min_max.append(min_max)
        return np.array(all_min_max)
