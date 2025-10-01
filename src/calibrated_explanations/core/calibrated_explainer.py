"""Calibrated Explanations for Black-Box Predictions (calibrated-explanations).

The calibrated explanations explanation method is based on the paper
"Calibrated Explanations: with Uncertainty Information and Counterfactuals"
by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

Calibrated explanations are a way to explain the predictions of a black-box learner
using Venn-Abers predictors (classification & regression) or
conformal predictive systems (regression).
"""

# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-lines, too-many-positional-arguments, too-many-public-methods
from __future__ import annotations

import warnings as _warnings
from time import time

import numpy as np
from typing import Any, Dict, List, Optional
from crepes import ConformalClassifier
from crepes.extras import hinge
from sklearn.metrics import confusion_matrix

from .._plots import _plot_global
from .._VennAbers import VennAbers
from ..explanations import AlternativeExplanations, CalibratedExplanations
from ..utils.discretizers import (
    BinaryEntropyDiscretizer,
    BinaryRegressorDiscretizer,
    EntropyDiscretizer,
    RegressorDiscretizer,
)
from ..utils.helper import (
    assert_threshold,
    check_is_fitted,
    concatenate_thresholds,
    convert_targets_to_numeric,
    immutable_array,
    safe_import,
    safe_mean,
    safe_isinstance,
)
from ..api.params import canonicalize_kwargs, validate_param_combination, warn_on_aliases
from .exceptions import (
    ValidationError,
    DataShapeError,
    ConfigurationError,
    NotFittedError,
)


class CalibratedExplainer:
    """The :class:`.CalibratedExplainer` class is used for explaining machine learning learners with calibrated predictions.

    The calibrated explanations are based on the paper
    "Calibrated Explanations for Black-Box Predictions"
    by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

    Calibrated explanations provides a way to explain the predictions of a black-box learner
    using Venn-Abers predictors (classification) or
    conformal predictive systems (regression).
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def __init__(
        self,
        learner,
        X_cal,
        y_cal,
        mode="classification",
        feature_names=None,
        categorical_features=None,
        categorical_labels=None,
        class_labels=None,
        bins=None,
        difficulty_estimator=None,
        **kwargs,
    ) -> None:
        """The :class:`.CalibratedExplainer` class is used for explaining machine learning learners with calibrated predictions.

        The calibrated explanations are based on the paper
        "Calibrated Explanations for Black-Box Predictions"
        by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

        Calibrated explanations provides a way to explain the predictions of a black-box learner
        using Venn-Abers predictors (classification) or
        conformal predictive systems (regression).

        Notes
        -----
        Minimal lifecycle logging is available at INFO level. To enable, configure:

        >>> import logging; logging.getLogger('calibrated_explanations').setLevel(logging.INFO)

        Initialize the :class:`.CalibratedExplainer` object for explaining the predictions of a black-box learner.

            Parameters
            ----------
            learner : predictive learner
                A predictive learner that can be used to predict the target variable. The learner must be fitted and have a predict_proba method (for classification) or a predict method (for regression).
            X_cal : array-like of shape (n_calibrations_samples, n_features)
                The calibration input data for the learner.
            y_cal : array-like of shape (n_calibrations_samples,)
                The calibration target data for the learner.
            mode : str, default="classification"
                The mode parameter specifies the type of problem being solved.
            feature_names : list of str, default=None
                A list of feature names for the input data. Each feature name should be a string. If not
                provided, the feature names will be assigned as "0", "1", "2", etc.
            categorical_features : list of int, default=None
                A list of indices for categorical features. These are the features that have discrete values
                and are not continuous.
            categorical_labels : dict(int, dict(int, str)), default=None
                A nested dictionary that maps the index of categorical features to another dictionary. The
                inner dictionary maps each feature value to a feature label. This is used for categorical
                feature encoding in the explanations. If None, the feature values will be used as labels.
            class_labels : dict(int, str), default=None
                A dictionary mapping numerical target values to class names. This parameter is only applicable
                for classification learners. If None, the numerical target values will be used as labels.
            bins : array-like of shape (n_samples,), default=None
                Mondrian categories
            difficulty_estimator : :class:`crepes.extras.DifficultyEstimator`, default=None
                A `DifficultyEstimator` object from the `crepes` package. It is used to estimate the difficulty of
                explaining a prediction. If None, no difficulty estimation is used. This parameter is only used
                for regression learners.
            sample_percentiles : list of int, default=[25, 50, 75]
                An array-like object that specifies the percentiles used to sample values for evaluation of
                numerical features. For example, if `sample_percentiles = [25, 50, 75]`, then the values at the
                25th, 50th, and 75th percentiles within each discretized group will be sampled from the calibration
                data for each numerical feature.
            seed : int, default=42
                The seed parameter is an integer that is used to set the random state for
                reproducibility. It is used in various parts of the code where randomization is involved, such
                as sampling values for evaluation of numerical features or initializing the random state for
                certain operations.
            verbose : bool, default=False
                A boolean parameter that determines whether additional printouts should be enabled during the
                operation of the class. If set to True, it will print out additional information during the
                execution of the code. If set to False, it will not print out any additional information.
            fast : bool, default=False
                A boolean parameter that determines whether the explainer should initiate the Fast Calibrated Explanations.
            reject : bool, default=False
                A boolean parameter that determines whether the explainer should reject explanations that are
                deemed too difficult to explain. If set to True, the explainer will reject explanations that are
                deemed too difficult to explain. If set to False, the explainer will not reject any explanations.
            oob : bool, default=False
                A boolean parameter that determines whether the explainer should use out-of-bag samples for calibration.
                If set to True, the explainer will use out-of-bag samples for calibration. If set to False, the explainer
                will not use out-of-bag samples for calibration. This requires the learner to be a RandomForestClassifier
            predict_function : function handle
                A function handle that takes an array-like input and returns an array-like output of probabilities
                for classification or predictions for regression. If not provided, defaults to predict_proba for
                classification mode or predict for regression mode. This allows customizing how predictions are
                generated from the learner.

            Returns
            -------
            :class:`.CalibratedExplainer` : A :class:`.CalibratedExplainer` object that can be used to explain predictions from a predictive learner.
        """
        init_time = time()
        self.__initialized = False
        check_is_fitted(learner)
        self.learner = learner
        self.predict_function = kwargs.get("predict_function")
        if self.predict_function is None:
            self.predict_function = (
                learner.predict_proba if mode == "classification" else learner.predict
            )
        # Optionally suppress or convert low-level crepes errors into clearer messages.
        # Caller can pass suppress_crepes_errors=True via kwargs to avoid raising on
        # crepes broadcasting/shape errors (useful for synthetic tiny datasets).
        self.suppress_crepes_errors = bool(kwargs.get("suppress_crepes_errors", False))
        self.oob = kwargs.get("oob", False)
        if self.oob:
            try:
                if mode == "classification":
                    y_oob_proba = self.learner.oob_decision_function_
                    if (
                        len(y_oob_proba.shape) == 1 or y_oob_proba.shape[1] == 1
                    ):  # Binary classification
                        y_oob = (y_oob_proba > 0.5).astype(np.dtype(y_cal.dtype))
                    else:  # Multiclass classification
                        y_oob = np.argmax(y_oob_proba, axis=1)
                        if safe_isinstance(y_cal, "pandas.core.arrays.categorical.Categorical"):
                            y_oob = y_cal.categories[y_oob]
                        else:
                            y_oob = y_oob.astype(np.dtype(y_cal.dtype))
                else:
                    y_oob = self.learner.oob_prediction_
            except Exception as exc:
                raise exc
            if len(X_cal) != len(y_oob):
                raise DataShapeError(
                    "The length of the out-of-bag predictions does not match the length of X_cal."
                )
            y_cal = y_oob
        self.X_cal = X_cal
        self.y_cal = y_cal

        self.set_seed(kwargs.get("seed", 42))
        self.sample_percentiles = kwargs.get("sample_percentiles", [25, 50, 75])
        self.verbose = kwargs.get("verbose", False)
        self.bins = bins

        self.__fast = kwargs.get("fast", False)
        self.__noise_type = kwargs.get("noise_type", "uniform")
        self.__scale_factor = kwargs.get("scale_factor", 5)
        self.__severity = kwargs.get("severity", 1)

        self.categorical_labels = categorical_labels
        self.class_labels = class_labels
        if categorical_features is None:
            if categorical_labels is not None:
                categorical_features = categorical_labels.keys()
            else:
                categorical_features = []
        self.categorical_features = list(categorical_features)
        self.features_to_ignore = kwargs.get("features_to_ignore", [])
        self._preprocess()

        if feature_names is None:
            feature_names = (
                self._X_cal[0].keys()
                if isinstance(self._X_cal[0], dict)
                else [str(i) for i in range(self.num_features)]
            )
        self._feature_names = list(feature_names)

        if mode == "classification":
            if any(isinstance(val, str) for val in self.y_cal) or any(
                isinstance(val, (np.str_, np.object_)) for val in self.y_cal
            ):
                self.y_cal_numeric, self.label_map = convert_targets_to_numeric(self.y_cal)
                self.y_cal = self.y_cal_numeric  # save to _y_cal to avoid append
                if self.class_labels is None:
                    self.class_labels = {v: k for k, v in self.label_map.items()}
            else:
                self.label_map = None
                if self.class_labels is None:
                    self.class_labels = {int(label): str(label) for label in np.unique(self.y_cal)}
        else:
            self.label_map = None
            self.class_labels = None

        self.discretizer: Any = None
        self.discretized_X_cal: Optional[np.ndarray] = None
        # Predeclare attributes for fast mode to satisfy type checkers
        self.fast_X_cal: Optional[np.ndarray] = None
        self.scaled_X_cal: Optional[np.ndarray] = None
        self.scaled_y_cal: Optional[np.ndarray] = None

        self.feature_values: Dict[int, List[Any]] = {}
        self.feature_frequencies: Dict[int, np.ndarray] = {}
        self.latest_explanation: Optional[CalibratedExplanations] = None
        self.__shap_enabled = False
        self.__lime_enabled = False
        self.lime: Any = None
        self.lime_exp: Any = None
        self.shap: Any = None
        self.shap_exp: Any = None
        self.reject = kwargs.get("reject", False)

        self.set_difficulty_estimator(difficulty_estimator, initialize=False)
        self.__set_mode(str.lower(mode), initialize=False)

        self.interval_learner: Any = None
        # Phase 1A delegation: interval learner initialization via helper
        from .calibration_helpers import initialize_interval_learner as _init_il

        _init_il(self)
        self.reject_learner = (
            self.initialize_reject_learner() if kwargs.get("reject", False) else None
        )

        self.init_time = time() - init_time

    @property
    def X_cal(self):
        """Get the calibration input data.

        Returns
        -------
        array-like
            The calibration input data.
        """
        return self.__X_cal if isinstance(self._X_cal[0], dict) else self._X_cal

    @X_cal.setter
    def X_cal(self, value):
        """Set the calibration input data.

        Parameters
        ----------
        value : array-like of shape (n_samples, n_features)
            The new calibration input data.

        Raises
        ------
        ValueError
            If the number of features in value does not match the existing calibration data.
        """
        if safe_isinstance(value, "pandas.core.frame.DataFrame"):
            value = value.values

        if len(value.shape) == 1:
            value = value.reshape(1, -1)

        self._X_cal = value

        if isinstance(self._X_cal[0], dict):
            self.__X_cal = np.array([[x[f] for f in x] for x in self._X_cal])

    @property
    def y_cal(self):
        """Get the calibration target data.

        Returns
        -------
        array-like
            The calibration target data.
        """
        return self._y_cal

    @y_cal.setter
    def y_cal(self, value):
        """Set the calibration target data.

        Parameters
        ----------
        value : array-like of shape (n_samples,)
            The new calibration target data.
        """
        if safe_isinstance(value, "pandas.core.frame.DataFrame"):
            self._y_cal = np.asarray(value.values)
        else:
            if len(value.shape) == 2 and value.shape[1] == 1:
                value = value.ravel()
            self._y_cal = np.asarray(value)

    def append_cal(self, X, y):
        """Append new calibration data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The new calibration input data to append.
        y : array-like of shape (n_samples,)
            The new calibration target data to append.
        """
        if X.shape[1] != self.num_features:
            raise DataShapeError("Number of features must match existing calibration data")
        self.X_cal = np.vstack((self.X_cal, X))
        self.y_cal = np.concatenate((self.y_cal, y))

    @property
    def num_features(self):
        """Get the number of features in the calibration data.

        Returns
        -------
        int
            The number of features in the calibration data. For dictionary input,
            returns the number of keys. For array input, returns the number of columns.
        """
        return (
            len(self._X_cal[0].keys())
            if isinstance(self._X_cal[0], dict)
            else len(self._X_cal[0, :])
        )

    @property
    def feature_names(self):
        """Get the feature names.

        Returns
        -------
        list
            The list of feature names. If no feature names were provided during initialization,
            returns None.
        """
        return self._feature_names

    def reinitialize(self, learner, xs=None, ys=None, bins=None):
        """Reinitialize the explainer with a new learner.

        This is useful when the learner is updated or retrained and the explainer needs to be reinitialized.

        Parameters
        ----------
        learner : predictive learner
            A predictive learner that can be used to predict the target variable. The learner must be fitted and have a predict_proba method (for classification) or a predict method (for regression).
        xs : array-like, optional
            New calibration input data to append
        ys : array-like, optional
            New calibration target data to append

        Returns
        -------
        :class:`.CalibratedExplainer`
            A :class:`.CalibratedExplainer` object that can be used to explain predictions from a predictive learner.
        """
        self.__initialized = False
        check_is_fitted(learner)
        self.learner = learner
        if xs is not None and ys is not None:
            self.append_cal(xs, ys)
            if bins is not None:
                if self.bins is None:
                    raise ValidationError("Cannot mix calibration instances with and without bins.")
                if len(bins) != len(ys):
                    raise DataShapeError(
                        "The length of bins must match the number of added instances."
                    )
                self.bins = np.concatenate((self.bins, bins)) if self.bins is not None else bins
            # Phase 1A delegation: update interval learner via helper
            from .calibration_helpers import update_interval_learner as _upd_il

            _upd_il(self, xs, ys, bins=bins)
        else:
            from .calibration_helpers import initialize_interval_learner as _init_il

            _init_il(self)
        self.__initialized = True

    def __repr__(self):
        """Return the string representation of the CalibratedExplainer."""
        # pylint: disable=line-too-long
        disp_str = f"CalibratedExplainer(mode={self.mode}{', conditional=True' if self.bins is not None else ''}{f', discretizer={self.discretizer}' if self.discretizer is not None else ''}, learner={self.learner}{f', difficulty_estimator={self.difficulty_estimator})' if self.mode == 'regression' else ')'}"
        if self.verbose:
            disp_str += f"\n\tinit_time={self.init_time}"
            if self.latest_explanation is not None:
                disp_str += f"\n\ttotal_explain_time={self.latest_explanation.total_explain_time}"
            disp_str += f"\n\tsample_percentiles={self.sample_percentiles}\
                        \n\tseed={self.seed}\
                        \n\tverbose={self.verbose}"
            if self.feature_names is not None:
                disp_str += f"\n\tfeature_names={self.feature_names}"
            if self.categorical_features is not None:
                disp_str += f"\n\tcategorical_features={self.categorical_features}"
            if self.categorical_labels is not None:
                disp_str += f"\n\tcategorical_labels={self.categorical_labels}"
            if self.class_labels is not None:
                disp_str += f"\n\tclass_labels={self.class_labels}"
        return disp_str

    # pylint: disable=invalid-name, too-many-return-statements
    def _predict(
        self,
        X_test,
        threshold=None,  # The same meaning as threshold has for cps in crepes.
        low_high_percentiles=(5, 95),
        classes=None,
        bins=None,
        feature=None,
        **kwargs,
    ):
        """Internal prediction method that handles both classification and regression cases.

        For classification:
        - Returns probabilities and intervals for binary/multiclass
        - Handles Mondrian categories via bins parameter

        For regression:
        - Returns predictions and uncertainty intervals
        - Can return probability predictions when threshold is provided

        Parameters
        ----------
        X_test : A set of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        classes : None or array-like of shape (n_samples,), default=None
            The classes predicted for the original instance. None if not multiclass or regression.

        Raises
        ------
        ValueError: The length of the threshold-parameter must be either a constant or the same as the number of
            instances in X_test.

        Returns
        -------
        predict : ndarray of shape (n_samples,)
            The prediction for the test data. For classification, this is the regularized probability
            of the positive class, derived using the intervals from VennAbers. For regression, this is the
            median prediction from the ConformalPredictiveSystem.
        low : ndarray of shape (n_samples,)
            The lower bound of the prediction interval. For classification, this is derived using
            VennAbers. For regression, this is the lower percentile given as parameter, derived from the
            ConformalPredictiveSystem.
        high : ndarray of shape (n_samples,)
            The upper bound of the prediction interval. For classification, this is derived using
            VennAbers. For regression, this is the upper percentile given as parameter, derived from the
            ConformalPredictiveSystem.
        classes : ndarray of shape (n_samples,)
            The classes predicted for the original instance. None if not multiclass or regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        """
        # strip plotting-only keys that callers may pass
        kwargs.pop("show", None)
        kwargs.pop("style_override", None)
        if not self.__initialized:
            raise NotFittedError("The learner must be initialized before calling predict.")
        if feature is None and self.is_fast():
            feature = self.num_features  # Use the calibrator defined using X_cal
        if self.mode == "classification":
            if self.is_multiclass():
                if self.is_fast():
                    predict, low, high, new_classes = self.interval_learner[feature].predict_proba(
                        X_test, output_interval=True, classes=classes, bins=bins
                    )
                else:
                    predict, low, high, new_classes = self.interval_learner.predict_proba(
                        X_test, output_interval=True, classes=classes, bins=bins
                    )
                if classes is None:
                    return (
                        [predict[i, c] for i, c in enumerate(new_classes)],
                        [low[i, c] for i, c in enumerate(new_classes)],
                        [high[i, c] for i, c in enumerate(new_classes)],
                        new_classes,
                    )
                if type(classes) not in (list, np.ndarray):
                    classes = [classes]
                return [predict[i, c] for i, c in enumerate(classes)], low, high, None

            if self.is_fast():
                predict, low, high = self.interval_learner[feature].predict_proba(
                    X_test, output_interval=True, bins=bins
                )
            else:
                predict, low, high = self.interval_learner.predict_proba(
                    X_test, output_interval=True, bins=bins
                )
            return predict[:, 1], low, high, None
        if "regression" in self.mode:
            # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            if threshold is None:  # normal regression
                if not (low_high_percentiles[0] <= low_high_percentiles[1]):
                    raise ValidationError(
                        "The low percentile must be smaller than (or equal to) the high percentile."
                    )
                if not (
                    (
                        (low_high_percentiles[0] > 0 and low_high_percentiles[0] <= 50)
                        and (low_high_percentiles[1] >= 50 and low_high_percentiles[1] < 100)
                    )
                    or low_high_percentiles[0] == -np.inf
                    or low_high_percentiles[1] == np.inf
                    and not (
                        low_high_percentiles[0] == -np.inf and low_high_percentiles[1] == np.inf
                    )
                ):
                    raise ValidationError(
                        "The percentiles must be between 0 and 100 (exclusive). \
                            The lower percentile can be -np.inf and the higher percentile can \
                            be np.inf (but not at the same time) to allow one-sided intervals."
                    )
                low = (
                    [low_high_percentiles[0], 50]
                    if low_high_percentiles[0] != -np.inf
                    else [50, 50]
                )
                high = (
                    [low_high_percentiles[1], 50] if low_high_percentiles[1] != np.inf else [50, 50]
                )

                try:
                    if self.is_fast():
                        return self.interval_learner[feature].predict_uncertainty(
                            X_test, low_high_percentiles, bins=bins
                        )
                    return self.interval_learner.predict_uncertainty(
                        X_test, low_high_percentiles, bins=bins
                    )
                except Exception:  # typically crepes broadcasting/shape errors
                    if self.suppress_crepes_errors:
                        # Log and return placeholder arrays (caller should handle downstream)
                        _warnings.warn(
                            "crepes produced an unexpected result (likely too-small calibration set); returning zeros as a degraded fallback.",
                            UserWarning,
                            stacklevel=2,
                        )
                        n = X_test.shape[0]
                        # produce zero-length or zero arrays consistent with expected shape
                        return np.zeros(n), np.zeros(n), np.zeros(n), None
                    # Preserve prior behavior: re-raise the original exception so callers/tests
                    # see the original error (instead of converting it to DataShapeError).
                    raise

            # regression with threshold condition
            assert_threshold(threshold, X_test)
            try:
                if self.is_fast():
                    return self.interval_learner[feature].predict_probability(
                        X_test, threshold, bins=bins
                    )
                # pylint: disable=unexpected-keyword-arg
                return self.interval_learner.predict_probability(X_test, threshold, bins=bins)
            except Exception as exc:
                if self.suppress_crepes_errors:
                    _warnings.warn(
                        "crepes produced an unexpected result while computing probabilities; returning zeros as a degraded fallback.",
                        UserWarning,
                        stacklevel=2,
                    )
                    n = X_test.shape[0]
                    return np.zeros(n), np.zeros(n), np.zeros(n), None
                    # Re-raise as a clearer DataShapeError with guidance
                    raise DataShapeError(
                        "Error while computing prediction intervals from the underlying crepes library. "
                        "This commonly occurs when the calibration set is too small for the requested percentiles. "
                        "Consider using a larger calibration set, or instantiate the explainer with suppress_crepes_errors=True to return a degraded fallback. "
                        f"(original error: {exc})"
                    ) from exc

        return None, None, None, None  # Should never happen

    def explain_factual(
        self,
        X_test,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
    ) -> CalibratedExplanations:
        """Create a :class:`.CalibratedExplanations` object for the test data with the discretizer automatically assigned for factual explanations.

        Parameters
        ----------
        X_test : array-like
            A set with n_samples of test objects to predict.
        threshold : float, int or array-like, default=None
            Values for which p-values should be returned. Only used for probabilistic explanations for regression.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of
            instances in X_test.

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A `CalibratedExplanations` containing one :class:`.FactualExplanation` for each instance.
        """
        discretizer = "binaryRegressor" if "regression" in self.mode else "binaryEntropy"
        self.set_discretizer(discretizer, features_to_ignore=features_to_ignore)
        return self.explain(X_test, threshold, low_high_percentiles, bins, features_to_ignore)

    def explain_counterfactual(
        self,
        X_test,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
    ) -> AlternativeExplanations:
        """See documentation for the `explore_alternatives` method.

        See Also
        --------
        :meth:`.CalibratedExplainer.explore_alternatives` : Refer to the documentation for `explore_alternatives` for more details.

        Warnings
        --------
        Deprecated: This method is deprecated and may be removed in future versions. Use `explore_alternatives` instead.
        """
        _warnings.warn(
            "The `explain_counterfactual` method is deprecated and may be removed in future versions. Use `explore_alternatives` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.explore_alternatives(
            X_test, threshold, low_high_percentiles, bins, features_to_ignore
        )

    def explore_alternatives(
        self,
        X_test,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
    ) -> AlternativeExplanations:
        """Create a :class:`.AlternativeExplanations` object for the test data with the discretizer automatically assigned for alternative explanations.

        Parameters
        ----------
        X_test : array-like
            A set with n_samples of test objects to predict.
        threshold : float, int or array-like, default=None
            Values for which p-values should be returned. Only used for probabilistic explanations for regression.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of
            instances in X_test.

        Returns
        -------
        AlternativeExplanations : :class:`.AlternativeExplanations`
            An `AlternativeExplanations` containing one :class:`.AlternativeExplanation` for each instance.

        Notes
        -----
        The `explore_alternatives` will eventually be used instead of the `explain_counterfactual` method.
        """
        discretizer = "regressor" if "regression" in self.mode else "entropy"
        self.set_discretizer(discretizer, features_to_ignore=features_to_ignore)
        # At runtime, explain() will return an AlternativeExplanations when an alternative discretizer is set.
        # Help mypy with a narrow cast here without changing behavior.
        return self.explain(X_test, threshold, low_high_percentiles, bins, features_to_ignore)  # type: ignore[return-value]

    def __call__(
        self,
        X_test,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
    ) -> CalibratedExplanations:
        """Call self as a function to create a :class:`.CalibratedExplanations` object for the test data with the already assigned discretizer.

        Since v0.4.0, this method is equivalent to the `explain` method.
        """
        return self.explain(X_test, threshold, low_high_percentiles, bins, features_to_ignore)

    def explain(
        self,
        X_test,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
    ) -> CalibratedExplanations:
        """Generate explanations for test instances by analyzing feature effects.

        This method:
        1. Makes predictions on original test instances
        2. Creates perturbed versions by varying feature values
        3. Analyzes how predictions change with feature perturbations
        4. Generates feature importance weights and prediction intervals

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A :class:`.CalibratedExplanations` containing one :class:`.CalibratedExplanation` for each instance.

        See Also
        --------
        :meth:`.CalibratedExplainer.explain_factual` : Refer to the documentation for `explain_factual` for more details.
        :meth:`.CalibratedExplainer.explore_alternatives` : Refer to the documentation for `explore_alternatives` for more details.
        """
        # Track total explanation time
        total_time = time()

        features_to_ignore = (
            self.features_to_ignore
            if features_to_ignore is None
            else np.union1d(self.features_to_ignore, features_to_ignore)
        )

        # Validate inputs and initialize explanation object
        X_test = self._validate_and_prepare_input(X_test)
        explanation = self._initialize_explanation(
            X_test, low_high_percentiles, threshold, bins, features_to_ignore
        )

        instance_time = time()

        # Step 1: Get predictions for original test instances
        (
            predict,
            low,
            high,
            prediction,
            perturbed_feature,
            rule_boundaries,
            lesser_values,
            greater_values,
            covered_values,
            X_cal,
        ) = self._explain_predict_step(
            X_test, threshold, low_high_percentiles, bins, features_to_ignore
        )

        # Step 2: Initialize data structures to store feature-level results
        # Dictionaries to store aggregated results across all instances
        feature_weights: Dict[str, List[np.ndarray]] = {"predict": [], "low": [], "high": []}
        feature_predict: Dict[str, List[np.ndarray]] = {"predict": [], "low": [], "high": []}
        binned_predict: Dict[str, List[Any]] = {  # Results for discretized feature values
            "predict": [],
            "low": [],
            "high": [],
            "current_bin": [],
            "rule_values": [],
            "counts": [],
            "fractions": [],
        }

        # Initialize per-instance storage
        rule_values: Dict[int, Dict[int, Any]] = {}
        instance_weights: Dict[int, Dict[str, np.ndarray]] = {}
        instance_predict: Dict[int, Dict[str, np.ndarray]] = {}
        instance_binned: Dict[int, Dict[str, Dict[int, Any]]] = {}

        # Initialize data structures for each test instance
        for i, x in enumerate(X_test):
            rule_values[i] = {}
            instance_weights[i] = {
                "predict": np.zeros(x.shape[0]),
                "low": np.zeros(x.shape[0]),
                "high": np.zeros(x.shape[0]),
            }
            instance_predict[i] = {
                "predict": np.zeros(x.shape[0]),
                "low": np.zeros(x.shape[0]),
                "high": np.zeros(x.shape[0]),
            }
            instance_binned[i] = {
                "predict": {},
                "low": {},
                "high": {},
                "current_bin": {},
                "rule_values": {},
                "counts": {},
                "fractions": {},
            }

        # Step 3: Process each feature to analyze its effects
        for f in range(self.num_features):
            if f in features_to_ignore:
                for i in range(len(X_test)):
                    rule_values[i][f] = (self.feature_values[f], X_test[i, f], X_test[i, f])
                    instance_binned[i]["predict"][f] = predict[i]
                    instance_binned[i]["low"][f] = low[i]
                    instance_binned[i]["high"][f] = high[i]
                continue

            # Get discretized values for this feature
            feature_values = self.feature_values[f]
            perturbed = [v[1] for v in perturbed_feature if v[0] == f]

            # Handle categorical and numerical features differently
            if f in self.categorical_features:
                # Process categorical feature - analyze effect of each possible value
                for i in np.unique(perturbed):
                    current_bin = -1
                    # Initialize arrays to store predictions for each feature value
                    average_predict = np.zeros(len(feature_values))
                    low_predict = np.zeros(len(feature_values))
                    high_predict = np.zeros(len(feature_values))
                    counts = np.zeros(len(feature_values))

                    # Calculate predictions for each possible feature value
                    for bin_value, value in enumerate(feature_values):
                        # Find predictions where this feature was set to this value
                        feature_index = [
                            perturbed_feature[j, 0] == f
                            and perturbed_feature[j, 1] == i
                            and perturbed_feature[j, 2] == value
                            for j in range(len(perturbed_feature))
                        ]

                        # Track original feature value's bin
                        if X_test[i, f] == value:
                            current_bin = bin_value

                        # Store predictions for this value
                        average_predict[bin_value] = (
                            predict[feature_index][0] if len(predict[feature_index]) > 0 else 0
                        )
                        low_predict[bin_value] = (
                            low[feature_index][0] if len(low[feature_index]) > 0 else 0
                        )
                        high_predict[bin_value] = (
                            high[feature_index][0] if len(high[feature_index]) > 0 else 0
                        )
                        counts[bin_value] = len(np.where(X_cal[:, f] == value)[0])

                    # Store results for this instance
                    rule_values[i][f] = (feature_values, X_test[i, f], X_test[i, f])
                    uncovered = np.setdiff1d(np.arange(len(average_predict)), current_bin)
                    fractions = counts[uncovered] / np.sum(counts[uncovered])

                    # Store binned predictions
                    instance_binned[i]["predict"][f] = average_predict
                    instance_binned[i]["low"][f] = low_predict
                    instance_binned[i]["high"][f] = high_predict
                    instance_binned[i]["current_bin"][f] = current_bin
                    instance_binned[i]["counts"][f] = counts
                    instance_binned[i]["fractions"][f] = fractions

                    # Handle special case where current bin is the only bin
                    if len(uncovered) == 0:
                        instance_predict[i]["predict"][f] = 0
                        instance_predict[i]["low"][f] = 0
                        instance_predict[i]["high"][f] = 0

                        instance_weights[i]["predict"][f] = 0
                        instance_weights[i]["low"][f] = 0
                        instance_weights[i]["high"][f] = 0
                    else:
                        # Calculate the weighted average (only makes a difference for categorical features)
                        # instance_predict['predict'][f] = np.sum(average_predict[uncovered]*fractions[uncovered])
                        # instance_predict['low'][f] = np.sum(low_predict[uncovered]*fractions[uncovered])
                        # instance_predict['high'][f] = np.sum(high_predict[uncovered]*fractions[uncovered])
                        # Calculate the average predictions
                        instance_predict[i]["predict"][f] = safe_mean(average_predict[uncovered])
                        instance_predict[i]["low"][f] = safe_mean(low_predict[uncovered])
                        instance_predict[i]["high"][f] = safe_mean(high_predict[uncovered])

                        # Calculate feature importance weights
                        instance_weights[i]["predict"][f] = self._assign_weight(
                            instance_predict[i]["predict"][f], prediction["predict"][i]
                        )
                        tmp_low = self._assign_weight(
                            instance_predict[i]["low"][f], prediction["predict"][i]
                        )
                        tmp_high = self._assign_weight(
                            instance_predict[i]["high"][f], prediction["predict"][i]
                        )
                        instance_weights[i]["low"][f] = np.min([tmp_low, tmp_high])
                        instance_weights[i]["high"][f] = np.max([tmp_low, tmp_high])
            else:
                # Get unique feature values and boundaries for this feature
                feature_values = np.unique(np.array(X_cal[:, f]))
                lower_boundary = rule_boundaries[:, f, 0]
                upper_boundary = rule_boundaries[:, f, 1]

                # Initialize dictionaries to store predictions and counts for each instance
                avg_predict_map: Dict[int, np.ndarray] = {}
                low_predict_map: Dict[int, np.ndarray] = {}
                high_predict_map: Dict[int, np.ndarray] = {}
                counts_map: Dict[int, np.ndarray] = {}
                rule_value_map: Dict[int, List[np.ndarray]] = {}
                for i in range(len(X_test)):
                    # Set boundary values and initialize arrays based on number of bins
                    lower_boundary[i] = (
                        lower_boundary[i] if np.any(feature_values < lower_boundary[i]) else -np.inf
                    )
                    upper_boundary[i] = (
                        upper_boundary[i] if np.any(feature_values > upper_boundary[i]) else np.inf
                    )
                    num_bins = 1 + (1 if lower_boundary[i] != -np.inf else 0)
                    num_bins += 1 if upper_boundary[i] != np.inf else 0
                    avg_predict_map[i] = np.zeros(num_bins)
                    low_predict_map[i] = np.zeros(num_bins)
                    high_predict_map[i] = np.zeros(num_bins)
                    counts_map[i] = np.zeros(num_bins)
                    rule_value_map[i] = []

                # Track bin assignments
                bin_value = np.zeros(len(X_test), dtype=int)
                current_bin = -np.ones(len(X_test), dtype=int)

                # Process instances below lower boundary
                for j, val in enumerate(np.unique(lower_boundary)):
                    if lesser_values[f][j][0].shape[0] == 0:
                        continue
                    for i in np.where(lower_boundary == val)[0]:
                        # Find relevant perturbed feature indices
                        index = [
                            p_i
                            for p_i in range(len(perturbed_feature))
                            if perturbed_feature[p_i, 0] == f
                            and perturbed_feature[p_i, 1] == i
                            and perturbed_feature[p_i, 2] == j
                            and perturbed_feature[p_i, 3]
                        ]

                        # Store predictions and counts for values below boundary
                        avg_predict_map[i][bin_value[i]] = (
                            safe_mean(predict[index]) if len(index) > 0 else 0
                        )
                        low_predict_map[i][bin_value[i]] = (
                            safe_mean(low[index]) if len(index) > 0 else 0
                        )
                        high_predict_map[i][bin_value[i]] = (
                            safe_mean(high[index]) if len(index) > 0 else 0
                        )
                        counts_map[i][bin_value[i]] = len(np.where(X_cal[:, f] < val)[0])
                        rule_value_map[i].append(lesser_values[f][j][0])
                        bin_value[i] += 1

                # Process instances above upper boundary
                for j, val in enumerate(np.unique(upper_boundary)):
                    if greater_values[f][j][0].shape[0] == 0:
                        continue
                    for i in np.where(upper_boundary == val)[0]:
                        # Find relevant perturbed feature indices
                        index = [
                            p_i
                            for p_i in range(len(perturbed_feature))
                            if perturbed_feature[p_i, 0] == f
                            and perturbed_feature[p_i, 1] == i
                            and perturbed_feature[p_i, 2] == j
                            and not perturbed_feature[p_i, 3]
                        ]

                        # Store predictions and counts for values above boundary
                        avg_predict_map[i][bin_value[i]] = (
                            safe_mean(predict[index]) if len(index) > 0 else 0
                        )
                        low_predict_map[i][bin_value[i]] = (
                            safe_mean(low[index]) if len(index) > 0 else 0
                        )
                        high_predict_map[i][bin_value[i]] = (
                            safe_mean(high[index]) if len(index) > 0 else 0
                        )
                        counts_map[i][bin_value[i]] = len(np.where(X_cal[:, f] > val)[0])
                        rule_value_map[i].append(greater_values[f][j][0])
                        bin_value[i] += 1

                # Process instances between boundaries
                indices = range(len(X_test))
                for i in indices:
                    for j, (_lower, _upper) in enumerate(
                        np.unique(list(zip(lower_boundary, upper_boundary)), axis=0)
                    ):
                        # Find relevant perturbed feature indices
                        index = [
                            p_i
                            for p_i in range(len(perturbed_feature))
                            if perturbed_feature[p_i, 0] == f
                            and perturbed_feature[p_i, 1] == i
                            and perturbed_feature[p_i, 2] == j
                            and perturbed_feature[p_i, 3] is None
                        ]

                        # Store predictions and counts for values between boundaries
                        avg_predict_map[i][bin_value[i]] = (
                            safe_mean(predict[index]) if len(index) > 0 else 0
                        )
                        low_predict_map[i][bin_value[i]] = (
                            safe_mean(low[index]) if len(index) > 0 else 0
                        )
                        high_predict_map[i][bin_value[i]] = (
                            safe_mean(high[index]) if len(index) > 0 else 0
                        )
                        counts_map[i][bin_value[i]] = len(
                            np.where((X_cal[:, f] >= _lower) & (X_cal[:, f] <= _upper))[0]
                        )
                        rule_value_map[i].append(covered_values[f][j][0])
                        current_bin[i] = bin_value[i]
                # For each test instance
                for i in range(len(X_test)):
                    # Store rule values for this feature and instance
                    rule_values[i][f] = (rule_value_map[i], X_test[i, f], X_test[i, f])

                    # Get indices of bins not containing current value
                    uncovered = np.setdiff1d(np.arange(len(avg_predict_map[i])), current_bin[i])

                    # Calculate fractions for uncovered bins
                    fractions = counts_map[i][uncovered] / np.sum(counts_map[i][uncovered])

                    # Store binned prediction results for this feature and instance
                    instance_binned[i]["predict"][f] = avg_predict_map[i]
                    instance_binned[i]["low"][f] = low_predict_map[i]
                    instance_binned[i]["high"][f] = high_predict_map[i]
                    instance_binned[i]["current_bin"][f] = current_bin[i]
                    instance_binned[i]["counts"][f] = counts_map[i]
                    instance_binned[i]["fractions"][f] = fractions

                    # Handle the situation where the current bin is the only bin
                    if len(uncovered) == 0:
                        instance_predict[i]["predict"][f] = 0
                        instance_predict[i]["low"][f] = 0
                        instance_predict[i]["high"][f] = 0

                        instance_weights[i]["predict"][f] = 0
                        instance_weights[i]["low"][f] = 0
                        instance_weights[i]["high"][f] = 0
                    else:
                        # Calculate the weighted average (only makes a difference for categorical features)
                        # instance_predict['predict'][f] = np.sum(average_predict[uncovered]*fractions[uncovered])
                        # instance_predict['low'][f] = np.sum(low_predict[uncovered]*fractions[uncovered])
                        # instance_predict['high'][f] = np.sum(high_predict[uncovered]*fractions[uncovered])
                        instance_predict[i]["predict"][f] = safe_mean(avg_predict_map[i][uncovered])
                        instance_predict[i]["low"][f] = safe_mean(low_predict_map[i][uncovered])
                        instance_predict[i]["high"][f] = safe_mean(high_predict_map[i][uncovered])

                        instance_weights[i]["predict"][f] = self._assign_weight(
                            instance_predict[i]["predict"][f], prediction["predict"][i]
                        )
                        tmp_low = self._assign_weight(
                            instance_predict[i]["low"][f], prediction["predict"][i]
                        )
                        tmp_high = self._assign_weight(
                            instance_predict[i]["high"][f], prediction["predict"][i]
                        )
                        instance_weights[i]["low"][f] = np.min([tmp_low, tmp_high])
                        instance_weights[i]["high"][f] = np.max([tmp_low, tmp_high])

        for i in range(len(X_test)):
            binned_predict["predict"].append(instance_binned[i]["predict"])
            binned_predict["low"].append(instance_binned[i]["low"])
            binned_predict["high"].append(instance_binned[i]["high"])
            binned_predict["current_bin"].append(instance_binned[i]["current_bin"])
            binned_predict["rule_values"].append(rule_values[i])
            binned_predict["counts"].append(instance_binned[i]["counts"])
            binned_predict["fractions"].append(instance_binned[i]["fractions"])

            feature_weights["predict"].append(instance_weights[i]["predict"])
            feature_weights["low"].append(instance_weights[i]["low"])
            feature_weights["high"].append(instance_weights[i]["high"])

            feature_predict["predict"].append(instance_predict[i]["predict"])
            feature_predict["low"].append(instance_predict[i]["low"])
            feature_predict["high"].append(instance_predict[i]["high"])
        elapsed_time = time() - instance_time
        list_instance_time = [elapsed_time / len(X_test) for _ in range(len(X_test))]

        explanation = explanation.finalize(
            binned_predict,
            feature_weights,
            feature_predict,
            prediction,
            instance_time=list_instance_time,
            total_time=total_time,
        )
        self.latest_explanation = explanation
        return explanation

    def _validate_and_prepare_input(self, X_test):
        """Delegate to extracted helper (Phase 1A)."""
        from .prediction_helpers import validate_and_prepare_input as _vh

        return _vh(self, X_test)

    def _initialize_explanation(
        self, X_test, low_high_percentiles, threshold, bins, features_to_ignore
    ):
        """Delegate to extracted helper (Phase 1A)."""
        from .prediction_helpers import initialize_explanation as _ih

        return _ih(self, X_test, low_high_percentiles, threshold, bins, features_to_ignore)

    def _explain_predict_step(
        self, X_test, threshold, low_high_percentiles, bins, features_to_ignore
    ):
        # Phase 1A: delegate initial setup to prediction_helpers to lock behavior
        from .prediction_helpers import explain_predict_step as _eps

        (
            _base_predict,
            _base_low,
            _base_high,
            prediction,
            perturbed_feature,
            rule_boundaries,
            lesser_values,
            greater_values,
            covered_values,
            X_cal,
            perturbed_threshold,
            perturbed_bins,
            perturbed_X,
            perturbed_class,
        ) = _eps(self, X_test, threshold, low_high_percentiles, bins, features_to_ignore)

        # Sub-step 1.b: prepare and add the perturbed test instances (unchanged logic)
        # pylint: disable=too-many-nested-blocks
        for f in range(self.num_features):
            if f in features_to_ignore:
                continue
            if f in self.categorical_features:
                feature_values = self.feature_values[f]
                X_copy = np.array(X_test, copy=True)
                for value in feature_values:
                    X_copy[:, f] = value
                    perturbed_X = np.concatenate((perturbed_X, np.array(X_copy)))
                    perturbed_feature = np.concatenate(
                        (perturbed_feature, [(f, i, value, None) for i in range(X_test.shape[0])])
                    )
                    perturbed_bins = (
                        np.concatenate((perturbed_bins, bins)) if bins is not None else None
                    )
                    perturbed_class = np.concatenate((perturbed_class, prediction["predict"]))
                    perturbed_threshold = concatenate_thresholds(
                        perturbed_threshold, threshold, list(range(X_test.shape[0]))
                    )
            else:
                X_copy = np.array(X_test, copy=True)
                feature_values = np.unique(np.array(X_cal[:, f]))
                lower_boundary = rule_boundaries[:, f, 0]
                upper_boundary = rule_boundaries[:, f, 1]
                for i in range(len(X_test)):
                    lower_boundary[i] = (
                        lower_boundary[i] if np.any(feature_values < lower_boundary[i]) else -np.inf
                    )
                    upper_boundary[i] = (
                        upper_boundary[i] if np.any(feature_values > upper_boundary[i]) else np.inf
                    )

                lesser_values[f] = {}
                greater_values[f] = {}
                covered_values[f] = {}
                for j, val in enumerate(np.unique(lower_boundary)):
                    lesser_values[f][j] = (np.unique(self.__get_lesser_values(f, val)), val)
                    indices = np.where(lower_boundary == val)[0]
                    for value in lesser_values[f][j][0]:
                        X_local = X_copy[indices, :]
                        X_local[:, f] = value
                        perturbed_X = np.concatenate((perturbed_X, np.array(X_local)))
                        perturbed_feature = np.concatenate(
                            (perturbed_feature, [(f, i, j, True) for i in indices])
                        )
                        if bins is not None:
                            perturbed_bins = np.concatenate(
                                (
                                    perturbed_bins,
                                    bins[indices] if len(indices) > 1 else [bins[indices[0]]],
                                )
                            )
                        perturbed_class = np.concatenate(
                            (perturbed_class, prediction["classes"][indices])
                        )
                        perturbed_threshold = concatenate_thresholds(
                            perturbed_threshold, threshold, indices
                        )
                for j, val in enumerate(np.unique(upper_boundary)):
                    greater_values[f][j] = (np.unique(self.__get_greater_values(f, val)), val)
                    indices = np.where(upper_boundary == val)[0]
                    for value in greater_values[f][j][0]:
                        X_local = X_copy[indices, :]
                        X_local[:, f] = value
                        perturbed_X = np.concatenate((perturbed_X, np.array(X_local)))
                        perturbed_feature = np.concatenate(
                            (perturbed_feature, [(f, i, j, False) for i in indices])
                        )
                        if bins is not None:
                            perturbed_bins = np.concatenate(
                                (
                                    perturbed_bins,
                                    bins[indices] if len(indices) > 1 else [bins[indices[0]]],
                                )
                            )
                        perturbed_class = np.concatenate(
                            (perturbed_class, prediction["classes"][indices])
                        )
                        perturbed_threshold = concatenate_thresholds(
                            perturbed_threshold, threshold, indices
                        )
                indices = range(len(X_test))
                for i in indices:
                    covered_values[f][i] = (
                        self.__get_covered_values(f, lower_boundary[i], upper_boundary[i]),
                        (lower_boundary[i], upper_boundary[i]),
                    )
                    for value in covered_values[f][i][0]:
                        X_local = X_copy[i, :]
                        X_local[f] = value
                        perturbed_X = np.concatenate(
                            (perturbed_X, np.array(X_local.reshape(1, -1)))
                        )
                        perturbed_feature = np.concatenate((perturbed_feature, [(f, i, i, None)]))
                        perturbed_bins = (
                            np.concatenate((perturbed_bins, [bins[i]]))
                            if bins is not None
                            else None
                        )
                        perturbed_class = np.concatenate(
                            (perturbed_class, [prediction["classes"][i]])
                        )
                        if threshold is not None and isinstance(threshold, (list, np.ndarray)):
                            if isinstance(threshold[0], tuple) and len(perturbed_threshold) == 0:
                                perturbed_threshold = [threshold[i]]
                            else:
                                perturbed_threshold = np.concatenate(
                                    (perturbed_threshold, [threshold[i]])
                                )
        # Sub-step 1.c: Predict and convert to numpy arrays to allow boolean indexing
        if (
            threshold is not None
            and isinstance(threshold, (list, np.ndarray))
            and isinstance(threshold[0], tuple)
        ):
            perturbed_threshold = [tuple(pair) for pair in perturbed_threshold]
        predict, low, high, _ = self._predict(
            perturbed_X,
            threshold=perturbed_threshold,
            low_high_percentiles=low_high_percentiles,
            classes=perturbed_class,
            bins=perturbed_bins,
        )
        predict = np.array(predict)
        low = np.array(low)
        high = np.array(high)
        # predicted_class = np.array(perturbed_class)
        return (
            predict,
            low,
            high,
            prediction,
            perturbed_feature,
            rule_boundaries,
            lesser_values,
            greater_values,
            covered_values,
            X_cal,
        )

    def explain_fast(
        self,
        X_test,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
    ) -> CalibratedExplanations:
        """Create a :class:`.CalibratedExplanations` object for the test data.

        Parameters
        ----------
        X_test : array-like
            A set with n_samples of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of
            instances in X_test.
        RuntimeError: Fast explanations are only possible if the explainer is a Fast Calibrated Explainer.

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A `CalibratedExplanations` containing one :class:`.FastExplanation` for each instance.
        """
        if not self.is_fast():
            try:
                self.__fast = True
                self.__initialize_interval_learner_for_fast_explainer()
            except Exception as exc:
                self.__fast = False
                raise ConfigurationError(
                    "Fast explanations are only possible if the explainer is a Fast Calibrated Explainer."
                ) from exc
        total_time = time()
        instance_time = []
        if safe_isinstance(X_test, "pandas.core.frame.DataFrame"):
            X_test = X_test.values  # pylint: disable=invalid-name
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(1, -1)
        if X_test.shape[1] != self.num_features:
            raise DataShapeError(
                "The number of features in the test data must be the same as in the \
                            calibration data."
            )
        if self._is_mondrian():
            if bins is None:
                raise ValidationError(
                    "The bins parameter must be specified for Mondrian explanations."
                )
            if len(bins) != len(X_test):
                raise DataShapeError(
                    "The length of the bins parameter must be the same as the number of instances in X_test."
                )
        explanation = CalibratedExplanations(self, X_test, threshold, bins)

        if threshold is not None:
            if "regression" not in self.mode:
                raise ValidationError(
                    "The threshold parameter is only supported for mode='regression'."
                )
            assert_threshold(threshold, X_test)
        # explanation.low_high_percentiles = low_high_percentiles
        elif "regression" in self.mode:
            explanation.low_high_percentiles = low_high_percentiles

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
                "predict": np.zeros(self.num_features),
                "low": np.zeros(self.num_features),
                "high": np.zeros(self.num_features),
            }
            for _ in range(len(X_test))
        ]
        instance_predict = [
            {
                "predict": np.zeros(self.num_features),
                "low": np.zeros(self.num_features),
                "high": np.zeros(self.num_features),
            }
            for _ in range(len(X_test))
        ]

        feature_time = time()

        predict, low, high, predicted_class = self._predict(
            X_test, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins
        )
        prediction: Dict[str, Any] = {
            "predict": predict,
            "low": low,
            "high": high,
            "classes": (predicted_class if self.is_multiclass() else np.ones(X_test.shape[0])),
        }
        y_cal = self.y_cal
        self.y_cal = self.scaled_y_cal
        for f in range(self.num_features):
            if f in self.features_to_ignore:
                continue

            predict, low, high, predicted_class = self._predict(
                X_test,
                threshold=threshold,
                low_high_percentiles=low_high_percentiles,
                bins=bins,
                feature=f,
            )

            for i in range(len(X_test)):
                instance_weights[i]["predict"][f] = self._assign_weight(
                    predict[i], prediction["predict"][i]
                )
                tmp_low = self._assign_weight(low[i], prediction["predict"][i])
                tmp_high = self._assign_weight(high[i], prediction["predict"][i])
                instance_weights[i]["low"][f] = np.min([tmp_low, tmp_high])
                instance_weights[i]["high"][f] = np.max([tmp_low, tmp_high])

                instance_predict[i]["predict"][f] = predict[i]
                instance_predict[i]["low"][f] = low[i]
                instance_predict[i]["high"][f] = high[i]
        self.y_cal = y_cal

        for i in range(len(X_test)):
            feature_weights["predict"].append(instance_weights[i]["predict"])
            feature_weights["low"].append(instance_weights[i]["low"])
            feature_weights["high"].append(instance_weights[i]["high"])

            feature_predict["predict"].append(instance_predict[i]["predict"])
            feature_predict["low"].append(instance_predict[i]["low"])
            feature_predict["high"].append(instance_predict[i]["high"])
        feature_time = time() - feature_time
        instance_time = [feature_time / X_test.shape[0]] * X_test.shape[0]

        explanation.finalize_fast(
            feature_weights,
            feature_predict,
            prediction,
            instance_time=instance_time,
            total_time=total_time,
        )
        self.latest_explanation = explanation
        return explanation

    def explain_lime(
        self,
        X_test,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
    ) -> CalibratedExplanations:
        """Create a :class:`.CalibratedExplanations` object for the test data.

        Parameters
        ----------
        X_test : array-like
            A set with n_samples of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of
            instances in X_test.
        RuntimeError: Fast explanations are only possible if the explainer is a Fast Calibrated Explainer.

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A `CalibratedExplanations` containing one :class:`.FastExplanation` for each instance.
        """
        if not self.__lime_enabled:
            self._preload_lime()
        total_time = time()
        instance_time = []
        if safe_isinstance(X_test, "pandas.core.frame.DataFrame"):
            X_test = X_test.values  # pylint: disable=invalid-name
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(1, -1)
        if X_test.shape[1] != self.num_features:
            raise DataShapeError(
                "The number of features in the test data must be the same as in the \
                            calibration data."
            )
        if self._is_mondrian():
            if bins is None:
                raise ValidationError(
                    "The bins parameter must be specified for Mondrian explanations."
                )
            if len(bins) != len(X_test):
                raise DataShapeError(
                    "The length of the bins parameter must be the same as the number of instances in X_test."
                )
        explanation = CalibratedExplanations(self, X_test, threshold, bins)

        if threshold is not None:
            if "regression" not in self.mode:
                raise ValidationError(
                    "The threshold parameter is only supported for mode='regression'."
                )
            assert_threshold(threshold, X_test)
        # explanation.low_high_percentiles = low_high_percentiles
        elif "regression" in self.mode:
            explanation.low_high_percentiles = low_high_percentiles

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
                "predict": np.zeros(self.num_features),
                "low": np.zeros(self.num_features),
                "high": np.zeros(self.num_features),
            }
            for _ in range(len(X_test))
        ]
        instance_predict = [
            {
                "predict": np.zeros(self.num_features),
                "low": np.zeros(self.num_features),
                "high": np.zeros(self.num_features),
            }
            for _ in range(len(X_test))
        ]

        predict, low, high, predicted_class = self._predict(
            X_test, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins
        )
        prediction["predict"] = predict
        prediction["low"] = low
        prediction["high"] = high
        if self.is_multiclass():
            prediction["classes"] = predicted_class
        else:
            prediction["classes"] = np.ones(X_test.shape[0])

            explainer = self.lime

        def low_proba(x):
            _, low, _, _ = self._predict(
                x, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins
            )
            return np.asarray([[1 - l, l] for l in low])  # noqa E741

        def high_proba(x):
            _, _, high, _ = self._predict(
                x, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins
            )
            return np.asarray([[1 - h, h] for h in high])  # noqa E741

        res_struct: Dict[str, Dict[str, Any]] = {}
        res_struct["low"] = {}
        res_struct["high"] = {}
        res_struct["low"]["explanation"], res_struct["high"]["explanation"] = [], []
        res_struct["low"]["abs_rank"], res_struct["high"]["abs_rank"] = [], []
        res_struct["low"]["values"], res_struct["high"]["values"] = [], []

        for i, x in enumerate(X_test):
            instance_timer = time()

            assert explainer is not None
            low = explainer.explain_instance(x, predict_fn=low_proba, num_features=len(x))
            high = explainer.explain_instance(x, predict_fn=high_proba, num_features=len(x))

            res_struct["low"]["explanation"].append(low)
            res_struct["high"]["explanation"].append(high)
            res_struct["low"]["abs_rank"], res_struct["high"]["abs_rank"] = (
                np.zeros(len(x)),
                np.zeros(len(x)),
            )
            res_struct["low"]["values"], res_struct["high"]["values"] = (
                np.zeros(len(x)),
                np.zeros(len(x)),
            )

            for j, f in enumerate(low.local_exp[1]):
                res_struct["low"]["abs_rank"][f[0]] = low.local_exp[1][j][0]
                res_struct["low"]["values"][f[0]] = f[1]
            for j, f in enumerate(high.local_exp[1]):
                res_struct["high"]["abs_rank"][f[0]] = high.local_exp[1][j][0]
                res_struct["high"]["values"][f[0]] = f[1]

            for f in range(self.num_features):
                tmp_low = res_struct["low"]["values"][f]
                tmp_high = res_struct["high"]["values"][f]
                instance_weights[i]["low"][f] = np.min([tmp_low, tmp_high])
                instance_weights[i]["high"][f] = np.max([tmp_low, tmp_high])
                instance_weights[i]["predict"][f] = instance_weights[i]["high"][f] / (
                    1 - instance_weights[i]["low"][f] + instance_weights[i]["high"][f]
                )

                instance_predict[i]["low"][f] = (
                    low.predict_proba[-1] - instance_weights[i]["low"][f]
                )
                instance_predict[i]["high"][f] = (
                    high.predict_proba[-1] - instance_weights[i]["high"][f]
                )
                instance_predict[i]["predict"][f] = instance_predict[i]["high"][f] / (
                    1 - instance_predict[i]["low"][f] + instance_predict[i]["high"][f]
                )

            feature_weights["predict"].append(instance_weights[i]["predict"])
            feature_weights["low"].append(instance_weights[i]["low"])
            feature_weights["high"].append(instance_weights[i]["high"])

            feature_predict["predict"].append(instance_predict[i]["predict"])
            feature_predict["low"].append(instance_predict[i]["low"])
            feature_predict["high"].append(instance_predict[i]["high"])
            instance_time.append(time() - instance_timer)

        explanation.finalize_fast(
            feature_weights,
            feature_predict,
            prediction,
            instance_time=instance_time,
            total_time=total_time,
        )
        self.latest_explanation = explanation
        return explanation

    def assign_threshold(self, threshold):
        """Assign the threshold for the explainer.

        The threshold is used to calculate the p-values for the predictions.
        """
        if threshold is None:
            return None
        if isinstance(threshold, (list, np.ndarray)):
            return (
                np.empty((0,), dtype=tuple) if isinstance(threshold[0], tuple) else np.empty((0,))
            )
        return threshold

    def _assign_weight(self, instance_predict, prediction):
        return (
            prediction - instance_predict
            if np.isscalar(prediction)
            else [prediction[i] - ip for i, ip in enumerate(instance_predict)]
        )  # probabilistic regression

    def is_multiclass(self):
        """Test if it is a multiclass problem.

        Returns
        -------
        bool
            True if multiclass.
        """
        return self.num_classes > 2

    def is_fast(self):
        """Test if the explainer is fast.

        Returns
        -------
        bool
            True if fast.
        """
        return self.__fast

    def rule_boundaries(self, instances, perturbed_instances=None):
        """Extract the rule boundaries for a set of instances.

        Parameters
        ----------
        instances : array-like
            The instances to extract boundaries for.
        perturbed_instances : array-like, optional
            Discretized versions of instances. Defaults to None.

        Returns
        -------
        array-like
            Min and max values for each feature for each instance.
        """
        # backwards compatibility
        if len(instances.shape) == 1:
            min_max = []
            if perturbed_instances is None:
                perturbed_instances = self._discretize(instances.reshape(1, -1))
            for f in range(self.num_features):
                if f not in self.discretizer.to_discretize:
                    min_max.append([instances[f], instances[f]])
                else:
                    bins = np.concatenate(([-np.inf], self.discretizer.mins[f][1:], [np.inf]))
                    min_max.append(
                        [
                            self.discretizer.mins[f][
                                np.digitize(perturbed_instances[0, f], bins, right=True) - 1
                            ],
                            self.discretizer.maxs[f][
                                np.digitize(perturbed_instances[0, f], bins, right=True) - 1
                            ],
                        ]
                    )
            return min_max
        instances = np.array(instances)  # Ensure instances is a numpy array
        if perturbed_instances is None:
            perturbed_instances = self._discretize(instances)
        else:
            perturbed_instances = np.array(
                perturbed_instances
            )  # Ensure perturbed_instances is a numpy array

        all_min_max = []
        for instance, perturbed_instance in zip(instances, perturbed_instances):
            min_max = []
            for f in range(self.num_features):
                if f not in self.discretizer.to_discretize:
                    min_max.append([instance[f], instance[f]])
                else:
                    bins = np.concatenate(([-np.inf], self.discretizer.mins[f][1:], [np.inf]))
                    min_max.append(
                        [
                            self.discretizer.mins[f][
                                np.digitize(perturbed_instance[f], bins, right=True) - 1
                            ],
                            self.discretizer.maxs[f][
                                np.digitize(perturbed_instance[f], bins, right=True) - 1
                            ],
                        ]
                    )
            all_min_max.append(min_max)
        return np.array(all_min_max)

    def __get_greater_values(self, f: int, greater: float):
        """Get sample values greater than the given threshold for a numerical feature.
        Uses percentile sampling from calibration data."""
        if not np.any(self.X_cal[:, f] > greater):
            return np.array([])
        return np.percentile(self.X_cal[self.X_cal[:, f] > greater, f], self.sample_percentiles)

    def __get_lesser_values(self, f: int, lesser: float):
        """Get sample values less than the given threshold for a numerical feature.
        Uses percentile sampling from calibration data."""
        if not np.any(self.X_cal[:, f] < lesser):
            return np.array([])
        return np.percentile(self.X_cal[self.X_cal[:, f] < lesser, f], self.sample_percentiles)

    def __get_covered_values(self, f: int, lesser: float, greater: float):
        """Get sample values between lower and upper bounds for a numerical feature.
        Uses percentile sampling from calibration data."""
        covered = np.where((self.X_cal[:, f] >= lesser) & (self.X_cal[:, f] <= greater))[0]
        return np.percentile(self.X_cal[covered, f], self.sample_percentiles)

    def set_seed(self, seed: int) -> None:
        """Change the seed used in the random number generator.

        Parameters
        ----------
        seed : int
            The seed to be used in the random number generator.
        """
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def set_difficulty_estimator(self, difficulty_estimator, initialize=True) -> None:
        """Assign or update the difficulty estimator.

        If initialized to a difficulty estimator, the explainer can be used to reject explanations that are deemed too difficult.

        Parameters
        ----------
        difficulty_estimator : :class:`crepes.extras.DifficultyEstimator` or None):
            A :class:`crepes.extras.DifficultyEstimator` object from the crepes package. To remove the :class:`crepes.extras.DifficultyEstimator`, set to None.
        initialize (bool, optional):
            If true, then the interval learner is initialized once done. Defaults to True.
        """
        if difficulty_estimator is not None:
            try:
                if not difficulty_estimator.fitted:
                    raise NotFittedError(
                        "The difficulty estimator is not fitted. Please fit the estimator first."
                    )
            except AttributeError as e:
                raise NotFittedError(
                    "The difficulty estimator is not fitted. Please fit the estimator first."
                ) from e
        self.__initialized = False
        self.difficulty_estimator = difficulty_estimator
        if initialize:
            self.__initialize_interval_learner()

    def __constant_sigma(self, X: np.ndarray, learner=None, beta=None) -> np.ndarray:  # pylint: disable=unused-argument
        return np.ones(X.shape[0]) if isinstance(X, (np.ndarray, list, tuple)) else np.ones(1)

    def _get_sigma_test(self, X: np.ndarray) -> np.ndarray:
        """Return the difficulty (sigma) of the test instances."""
        if self.difficulty_estimator is None:
            return self.__constant_sigma(X)
        return self.difficulty_estimator.apply(X)

    def __set_mode(self, mode, initialize=True) -> None:
        """Assign the mode of the explainer. The mode can be either 'classification' or 'regression'.

        Parameters
        ----------
            mode (str): The mode can be either 'classification' or 'regression'.
            initialize (bool, optional): If true, then the interval learner is initialized once done. Defaults to True.

        Raises
        ------
            ValueError: The mode can be either 'classification' or 'regression'.
        """
        self.__initialized = False
        if mode == "classification":
            # assert 'predict_proba' in dir(self.learner), "The learner must have a predict_proba method."
            self.num_classes = len(np.unique(self.y_cal))
        elif mode == "regression":
            # assert 'predict' in dir(self.learner), "The learner must have a predict method."
            self.num_classes = 0
        else:
            raise ValidationError("The mode must be either 'classification' or 'regression'.")
        self.mode = mode
        if initialize:
            self.__initialize_interval_learner()

    def __update_interval_learner(self, xs, ys, bins=None) -> None:  # pylint: disable=unused-argument
        if self.is_fast():
            raise ConfigurationError("Fast explanations are not supported in this update path.")
        if self.mode == "classification":
            # pylint: disable=fixme
            # TODO: change so that existing calibrators are extended with new calibration instances
            self.interval_learner = VennAbers(
                self.X_cal,
                self.y_cal,
                self.learner,
                self.bins,
                difficulty_estimator=self.difficulty_estimator,
                predict_function=self.predict_function,
            )
        elif "regression" in self.mode:
            if isinstance(self.interval_learner, list):
                raise ConfigurationError("Fast explanations are not supported in this update path.")
            # update the IntervalRegressor
            self.interval_learner.insert_calibration(xs, ys, bins=bins)
        self.__initialized = True

    def __initialize_interval_learner(self) -> None:
        # Thin delegator kept for backward-compatibility internal calls
        from .calibration_helpers import initialize_interval_learner as _init_il

        _init_il(self)

    # pylint: disable=attribute-defined-outside-init
    def __initialize_interval_learner_for_fast_explainer(self):
        from .calibration_helpers import (
            initialize_interval_learner_for_fast_explainer as _init_fast,
        )

        _init_fast(self)

    def initialize_reject_learner(self, calibration_set=None, threshold=None):
        """Initialize the reject learner with a threshold value.

        The reject learner is a :class:`crepes.base.ConformalClassifier`
        that is trained on the calibration data. The reject learner is used to determine whether a test
        instance is within the calibration data distribution. The reject learner is only available for
        classification, unless a threshold is assigned.

        Parameters
        ----------
        calibration_set : array-like, optional
            The calibration set to use. Defaults to None.
        threshold : float, optional
            The threshold value. Defaults to None.
        """
        if calibration_set is None:
            X_cal, y_cal = self.X_cal, self.y_cal
        elif calibration_set is tuple:
            X_cal, y_cal = calibration_set
        else:
            X_cal, y_cal = calibration_set[0], calibration_set[1]
        self.reject_threshold = None
        if self.mode in "regression":
            proba_1, _, _, _ = self.interval_learner.predict_probability(
                X_cal, y_threshold=threshold, bins=self.bins
            )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            classes = (y_cal < threshold).astype(int)
            self.reject_threshold = threshold
        elif self.is_multiclass():  # pylint: disable=protected-access
            proba, classes = self.interval_learner.predict_proba(X_cal, bins=self.bins)
            proba = np.array([[1 - proba[i, c], proba[i, c]] for i, c in enumerate(classes)])
            classes = (classes == y_cal).astype(int)
        else:
            proba = self.interval_learner.predict_proba(X_cal, bins=self.bins)
            classes = y_cal
        alphas_cal = hinge(proba, np.unique(classes), classes)
        self.reject_learner = ConformalClassifier().fit(alphas=alphas_cal, bins=classes)
        return self.reject_learner

    def predict_reject(self, X_test, bins=None, confidence=0.95):
        """Predict whether to reject the explanations for the test data.

        Use conformal classifier to identify test instances that may be too different from calibration data.

        Parameters
        ----------
        X_test : array-like
            The test data.
        bins : array-like, optional
            Mondrian categories. Defaults to None.
        confidence : float, default=0.95
            The confidence level.

        Returns
        -------
        array-like
            Returns rejection decisions and error/rejection rates.
        """
        if self.mode in "regression":
            if self.reject_threshold is None:
                raise ValidationError(
                    "The reject learner is only available for regression with a threshold."
                )
            proba_1, _, _, _ = self.interval_learner.predict_probability(
                X_test, y_threshold=self.reject_threshold, bins=bins
            )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            classes = [0, 1]
        elif self.is_multiclass():  # pylint: disable=protected-access
            proba, classes = self.interval_learner.predict_proba(X_test, bins=bins)
            proba = np.array([[1 - proba[i, c], proba[i, c]] for i, c in enumerate(classes)])
            classes = [0, 1]
        else:
            proba = self.interval_learner.predict_proba(X_test, bins=bins)
            classes = np.unique(self.y_cal)
        alphas_test = hinge(proba)

        prediction_set = np.array(
            [
                self.reject_learner.predict_set(
                    alphas_test, np.full(len(alphas_test), classes[c]), confidence=confidence
                )[:, c]
                for c in range(len(classes))
            ]
        ).T
        singleton = np.sum(np.sum(prediction_set, axis=1) == 1)
        empty = np.sum(np.sum(prediction_set, axis=1) == 0)
        n = len(X_test)

        epsilon = 1 - confidence
        error_rate = (n * epsilon - empty) / singleton
        reject_rate = 1 - singleton / n

        rejected = np.sum(prediction_set, axis=1) != 1
        return rejected, error_rate, reject_rate

    def _preprocess(self):
        constant_columns = [
            f for f in range(self.num_features) if np.all(self.X_cal[:, f] == self.X_cal[0, f])
        ]
        self.features_to_ignore = constant_columns

    def _discretize(self, x):
        """Apply the discretizer to the data sample x.

        For new data samples and missing values, the nearest bin is used.

        Parameters
        ----------
        x : array-like
            The data sample to discretize.

        Returns
        -------
        array-like
            The discretized data sample.
        """
        x = np.array(x)  # Ensure x is a numpy array
        for f in self.discretizer.to_discretize:
            bins = np.concatenate(([-np.inf], self.discretizer.mins[f][1:], [np.inf]))
            x[:, f] = [
                self.discretizer.means[f][np.digitize(x[i, f], bins, right=True) - 1]
                for i in range(len(x))
            ]
        return x

    # pylint: disable=too-many-branches
    def set_discretizer(self, discretizer, X_cal=None, y_cal=None, features_to_ignore=None) -> None:
        """Assign the discretizer to be used.

        Parameters
        ----------
        discretizer : str or discretizer object
            The discretizer to be used.
        X_cal : array-like, optional
            The calibration data for the discretizer.
        y_cal : array-like, optional
            The calibration target data for the discretizer.
        """
        if X_cal is None:
            X_cal = self.X_cal
        if y_cal is None:
            y_cal = self.y_cal

        if discretizer is None:
            discretizer = "binaryRegressor" if "regression" in self.mode else "binaryEntropy"
        elif "regression" in self.mode:
            if not (
                discretizer is None
                or discretizer
                in {
                    "regressor",
                    "binaryRegressor",
                }
            ):
                raise ValidationError(
                    "The discretizer must be 'binaryRegressor' (default for factuals) or 'regressor' (default for alternatives) for regression."
                )
        else:
            if not (
                discretizer is None
                or discretizer
                in {
                    "entropy",
                    "binaryEntropy",
                }
            ):
                raise ValidationError(
                    "The discretizer must be 'binaryEntropy' (default for factuals) or 'entropy' (default for alternatives) for classification."
                )

        if features_to_ignore is None:
            features_to_ignore = []
        not_to_discretize = np.union1d(
            np.union1d(self.categorical_features, self.features_to_ignore), features_to_ignore
        )
        if discretizer == "binaryEntropy":
            if isinstance(self.discretizer, BinaryEntropyDiscretizer):
                return
            self.discretizer = BinaryEntropyDiscretizer(
                X_cal, not_to_discretize, self.feature_names, labels=y_cal, random_state=self.seed
            )
        elif discretizer == "binaryRegressor":
            if isinstance(self.discretizer, BinaryRegressorDiscretizer):
                return
            self.discretizer = BinaryRegressorDiscretizer(
                X_cal, not_to_discretize, self.feature_names, labels=y_cal, random_state=self.seed
            )

        elif discretizer == "entropy":
            if isinstance(self.discretizer, EntropyDiscretizer):
                return
            self.discretizer = EntropyDiscretizer(
                X_cal, not_to_discretize, self.feature_names, labels=y_cal, random_state=self.seed
            )
        elif discretizer == "regressor":
            if isinstance(self.discretizer, RegressorDiscretizer):
                return
            self.discretizer = RegressorDiscretizer(
                X_cal, not_to_discretize, self.feature_names, labels=y_cal, random_state=self.seed
            )
        self.discretized_X_cal = self._discretize(immutable_array(self.X_cal))

        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in range(self.num_features):
            assert self.discretized_X_cal is not None
            column = self.discretized_X_cal[:, feature]
            feature_count: Dict[Any, int] = {}
            for item in column:
                feature_count[item] = feature_count.get(item, 0) + 1
            values, frequencies = map(list, zip(*(sorted(feature_count.items()))))

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = np.array(frequencies) / float(sum(frequencies))

    def _is_mondrian(self):
        """Return whether the explainer is a Mondrian explainer.

        Returns
        -------
            bool: True if Mondrian
        """
        return self.bins is not None

    # pylint: disable=too-many-return-statements
    def predict(self, X_test, uq_interval=False, calibrated=True, **kwargs):
        """Generate predictions for the test data.

        Parameters
        ----------
        X_test : array-like
            The test data.
        uq_interval : bool, default=False
            Whether to return uncertainty intervals.
        calibrated : bool, default=True
            If True, the calibrator is used for prediction. If False, the underlying learner is used for prediction.
        **kwargs : Various types, optional
            Additional parameters to customize the explanation process. Supported parameters include:

            - threshold : float, int, or array-like of shape (n_samples,), optional, default=None
                Specifies the threshold(s) to get a thresholded prediction for regression tasks (prediction labels: `y_hat<=threshold-value` | `y_hat>threshold-value`). This parameter is ignored for classification tasks.

            - low_high_percentiles : tuple of two floats, optional, default=(5, 95)
                The lower and upper percentiles used to calculate the prediction interval for regression tasks. Determines the breadth of the interval based on the distribution of the predictions. This parameter is ignored for classification tasks.

        Raises
        ------
        RuntimeError
            If the learner has not been fitted prior to making predictions.

        Warning
            If the learner is not calibrated.

        Returns
        -------
        calibrated_prediction : float or array-like, or str
            The calibrated prediction. For regression tasks, this is the median of the conformal predictive system or a thresholded prediction if `threshold`is set. For classification tasks, it is the class label with the highest calibrated probability.
        interval : tuple of floats, optional
            A tuple (low, high) representing the lower and upper bounds of the uncertainty interval. This is returned only if `uq_interval=True`.

        Examples
        --------
        For a prediction without prediction intervals:

        .. code-block:: python

            w.predict(X_test)

        For a prediction with uncertainty quantification intervals:

        .. code-block:: python

            w.predict(X_test, uq_interval=True)

        Notes
        -----
        The `threshold` and `low_high_percentiles` parameters are only used for regression tasks.
        """
        # Phase 1B: emit deprecation warnings for aliases and normalize kwargs
        warn_on_aliases(kwargs)
        kwargs = canonicalize_kwargs(kwargs)
        validate_param_combination(kwargs)

        if not calibrated:
            if "threshold" in kwargs:
                raise ValidationError(
                    "A thresholded prediction is not possible for uncalibrated predictions."
                )
            if uq_interval:
                predict = self.learner.predict(X_test)
                return predict, (predict, predict)
            return self.learner.predict(X_test)

        if self.mode in "regression":
            predict, low, high, _ = self._predict(X_test, **kwargs)
            if "threshold" in kwargs:

                def get_label(predict, threshold):
                    if np.isscalar(threshold):
                        return f"y_hat <= {threshold}" if predict >= 0.5 else f"y_hat > {threshold}"
                    if isinstance(threshold, tuple):
                        return (
                            f"{threshold[0]} < y_hat <= {threshold[1]}"
                            if predict >= 0.5
                            else f"y_hat <= {threshold[0]} || y_hat > {threshold[1]}"
                        )
                    return (
                        "Error in CalibratedExplainer.predict.get_label()"  # should not reach here
                    )

                threshold = kwargs["threshold"]
                if np.isscalar(threshold) or isinstance(threshold, tuple):
                    new_classes = [get_label(predict[i], threshold) for i in range(len(predict))]
                else:
                    new_classes = [get_label(predict[i], threshold[i]) for i in range(len(predict))]
                return (new_classes, (low, high)) if uq_interval else new_classes
            return (predict, (low, high)) if uq_interval else predict

        predict, low, high, new_classes = self._predict(X_test, **kwargs)
        if new_classes is None:
            new_classes = (predict >= 0.5).astype(int)
        if self.label_map is not None or self.class_labels is not None:
            new_classes = np.array([self.class_labels[c] for c in new_classes])
        return (new_classes, (low, high)) if uq_interval else new_classes

    def predict_proba(self, X_test, uq_interval=False, calibrated=True, threshold=None, **kwargs):
        """Generate probability predictions for the test data.

        This is a wrapper around the predict_proba method which is more similar to the scikit-learn predict_proba method for classification.
        As opposed to predict_proba, this method may output uncertainty intervals.

        Parameters
        ----------
        X_test : array-like
            The test data for which predictions are to be made. This should be in a format compatible with sklearn (e.g., numpy arrays, pandas DataFrames).
        uq_interval : bool, default=False
            If true, then the prediction interval is returned as well.
        calibrated : bool, default=True
            If True, the calibrator is used for prediction. If False, the underlying learner is used for prediction.
        threshold : float, int or array-like of shape (n_samples,), optional, default=None
            Threshold values used with regression to get probability of being below the threshold. Only applicable to regression.

        Raises
        ------
        RuntimeError
            If the learner is not fitted before predicting.

        ValueError
            If the `threshold` parameter's length does not match the number of instances in `X_test`, or if it is not a single constant value applicable to all instances.

        RuntimeError
            If the learner is not fitted before predicting.

        Warning
            If the learner is not calibrated.

        Returns
        -------
        calibrated probability :
            The calibrated probability of the positive class (or the predicted class for multiclass).
        (low, high) : tuple of float lists, corresponding to the lower and upper bound of each prediction interval.

        Examples
        --------
        For a prediction without uncertainty quantification intervals:

        .. code-block:: python

            w.predict_proba(X_test)

        For a prediction with uncertainty quantification intervals:

        .. code-block:: python

            w.predict_proba(X_test, uq_interval=True)

        Notes
        -----
        The `threshold` parameter is only used for regression tasks.
        """
        # strip plotting-only keys that callers may pass
        kwargs.pop("show", None)
        kwargs.pop("style_override", None)
        # Phase 1B: emit deprecation warnings for aliases and normalize kwargs
        warn_on_aliases(kwargs)
        kwargs = canonicalize_kwargs(kwargs)
        validate_param_combination(kwargs)
        if not calibrated:
            if threshold is not None:
                raise ValidationError(
                    "A thresholded prediction is not possible for uncalibrated learners."
                )
            if uq_interval:
                proba = self.learner.predict_proba(X_test)
                if proba.shape[1] > 2:
                    return proba, (proba, proba)
                return proba, (proba[:, 1], proba[:, 1])
            return self.learner.predict_proba(X_test)
        if self.mode in "regression":
            if isinstance(self.interval_learner, list):
                proba_1, low, high, _ = self.interval_learner[-1].predict_probability(
                    X_test, y_threshold=threshold, **kwargs
                )
            else:
                proba_1, low, high, _ = self.interval_learner.predict_probability(
                    X_test, y_threshold=threshold, **kwargs
                )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            return (proba, (low, high)) if uq_interval else proba
        if self.is_multiclass():  # pylint: disable=protected-access
            if isinstance(self.interval_learner, list):
                proba, low, high, _ = self.interval_learner[-1].predict_proba(
                    X_test, output_interval=True, **kwargs
                )
            else:
                proba, low, high, _ = self.interval_learner.predict_proba(
                    X_test, output_interval=True, **kwargs
                )
            return (proba, (low, high)) if uq_interval else proba
        if isinstance(self.interval_learner, list):
            proba, low, high = self.interval_learner[-1].predict_proba(
                X_test, output_interval=True, **kwargs
            )
        else:
            proba, low, high = self.interval_learner.predict_proba(
                X_test, output_interval=True, **kwargs
            )
        return (proba, (low, high)) if uq_interval else proba

    def _is_lime_enabled(self, is_enabled=None) -> bool:
        """Return whether lime export is enabled.

        If is_enabled is not None, then the lime export is enabled/disabled according to the value of is_enabled.

        Parameters
        ----------
            is_enabled (bool, optional): is used to assign whether lime export is enabled or not. Defaults to None.

        Returns
        -------
            bool: returns whether lime export is enabled
        """
        if is_enabled is not None:
            self.__lime_enabled = is_enabled
        return self.__lime_enabled

    def _is_shap_enabled(self, is_enabled=None) -> bool:
        """Return whether shap export is enabled.

        If is_enabled is not None, then the shap export is enabled/disabled according to the value of is_enabled.

        Parameters
        ----------
            is_enabled (bool, optional): is used to assign whether shap export is enabled or not. Defaults to None.

        Returns
        -------
            bool: returns whether shap export is enabled
        """
        if is_enabled is not None:
            self.__shap_enabled = is_enabled
        return self.__shap_enabled

    def _preload_lime(self, X_cal=None):
        if not (lime := safe_import("lime.lime_tabular", "LimeTabularExplainer")):
            return None, None
        if not self._is_lime_enabled():
            if self.mode == "classification":
                self.lime = lime(
                    self.X_cal[:1, :] if X_cal is None else X_cal,
                    feature_names=self.feature_names,
                    class_names=["0", "1"],
                    mode=self.mode,
                )
                self.lime_exp = self.lime.explain_instance(
                    self.X_cal[0, :], self.learner.predict_proba, num_features=self.num_features
                )
            elif "regression" in self.mode:
                self.lime = lime(
                    self.X_cal[:1, :] if X_cal is None else X_cal,
                    feature_names=self.feature_names,
                    mode="regression",
                )
                self.lime_exp = self.lime.explain_instance(
                    self.X_cal[0, :], self.learner.predict, num_features=self.num_features
                )
            self._is_lime_enabled(True)
        return self.lime, self.lime_exp

    def _preload_shap(self, num_test=None):
        if shap := safe_import("shap"):
            if (
                not self._is_shap_enabled()
                or num_test is not None
                and self.shap_exp.shape[0] != num_test
            ):

                def f(x):
                    return self._predict(x)[0]

                self.shap = shap.Explainer(f, self.X_cal, feature_names=self.feature_names)
                self.shap_exp = (
                    self.shap(self.X_cal[0, :].reshape(1, -1))
                    if num_test is None
                    else self.shap(self.X_cal[:num_test, :])
                )
                self._is_shap_enabled(True)
            return self.shap, self.shap_exp
        return None, None

    # pylint: disable=duplicate-code, too-many-branches, too-many-statements, too-many-locals
    def plot(self, X_test, y_test=None, threshold=None, **kwargs):
        """Generate plots for the test data."""
        # Pass any style overrides along to the plotting function
        style_override = kwargs.pop("style_override", None)
        kwargs["style_override"] = style_override
        _plot_global(self, X_test, y_test=y_test, threshold=threshold, **kwargs)

    def calibrated_confusion_matrix(self):
        """Generate a calibrated confusion matrix.

        Generates a confusion matrix for the calibration set to provide insights about model behavior.
        The confusion matrix is only available for classification tasks. Leave-one-out cross-validation is
        used on the calibration set to generate the confusion matrix.

        Returns
        -------
        array-like
            The calibrated confusion matrix.
        """
        if not (self.mode == "classification"):
            raise ValidationError(
                "The confusion matrix is only available for classification tasks."
            )
        cal_predicted_classes = np.zeros(len(self.y_cal))
        for i in range(len(self.y_cal)):
            va = VennAbers(
                np.concatenate((self.X_cal[:i], self.X_cal[i + 1 :]), axis=0),
                np.concatenate((self.y_cal[:i], self.y_cal[i + 1 :])),
                self.learner,
                bins=np.concatenate((self.bins[:i], self.bins[i + 1 :]))
                if self.bins is not None
                else None,
            )
            _, _, _, predict = va.predict_proba(
                [self.X_cal[i]],
                output_interval=True,
                bins=[self.bins[i]] if self.bins is not None else None,
            )
            cal_predicted_classes[i] = predict[0]
        return confusion_matrix(self.y_cal, cal_predicted_classes)

    def predict_calibration(self):
        """Predict the target values for the calibration data.

        Returns
        -------
        array-like
            Predicted values for the calibration data. For models that expose a hat matrix,
            this returns updated predictions using that matrix; otherwise it uses the
            predict_function on the calibration data.
        """
        return self.predict_function(self.X_cal)


__all__ = ["CalibratedExplainer"]
