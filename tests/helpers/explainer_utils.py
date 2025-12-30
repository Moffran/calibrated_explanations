"""Helper fixtures and mocks for explainer tests."""

from __future__ import annotations

import pytest
import numpy as np
from typing import Any, Iterable, Sequence, Tuple, Union, Mapping

from calibrated_explanations import CalibratedExplainer
from tests.helpers.model_utils import (
    get_classification_model,
    get_regression_model,
    DummyLearner,
    DummyIntervalLearner,
)


def patch_interval_initializers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch the interval learner factories to return the dummy implementation."""

    def initialize(explainer: CalibratedExplainer, *args: Any, **kwargs: Any) -> None:
        explainer.interval_learner = DummyIntervalLearner()
        explainer.initialized = True

    monkeypatch.setattr(
        "calibrated_explanations.calibration.interval_learner.initialize_interval_learner",
        initialize,
    )
    monkeypatch.setattr(
        "calibrated_explanations.calibration.interval_learner.initialize_interval_learner_for_fast_explainer",
        initialize,
    )


def make_mock_explainer(
    monkeypatch: pytest.MonkeyPatch,
    learner: DummyLearner,
    x_cal: np.ndarray,
    y_cal: Any,
    **kwargs: Any,
) -> CalibratedExplainer:
    """Build a CalibratedExplainer using monkeypatched interval learners."""
    patch_interval_initializers(monkeypatch)
    return CalibratedExplainer(learner, x_cal, y_cal, **kwargs)


def make_explainer_from_dataset(dataset, mode="classification", **overrides):
    """Create a calibrated explainer from the dataset fixture.

    Parameters
    ----------
    dataset : tuple
        Fixture tuple produced by `make_binary_dataset` or `make_regression_dataset`.
    mode : str
        "classification" or "regression".
    **overrides : Any
        Overrides to pass to the `CalibratedExplainer` constructor.

    Returns
    -------
    tuple[CalibratedExplainer, np.ndarray]
        Explainer and the test inputs.
    """
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _num_classes,
        _num_features,
        categorical_features,
        feature_names,
    ) = dataset

    if mode == "classification":
        model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
        explainer = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
            mode="classification",
            feature_names=feature_names,
            categorical_features=categorical_features,
            class_labels=["No", "Yes"],
            seed=overrides.pop("seed", 42),
            **overrides,
        )
    else:
        model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
        explainer = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
            mode="regression",
            feature_names=feature_names,
            categorical_features=categorical_features,
            seed=overrides.pop("seed", 42),
            **overrides,
        )
    return explainer, x_test


def make_multiclass_explainer_from_dataset(multiclass_dataset, **overrides):
    """Create a multiclass calibrated explainer using the provided fixture."""
    from tests.helpers.model_utils import get_classification_model

    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,  # y_test
        _,  # num_classes
        _,  # num_features
        categorical_features,
        _,  # categorical_labels
        _,  # target_labels
        feature_names,
    ) = multiclass_dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        mode="classification",
        feature_names=feature_names,
        categorical_features=categorical_features,
        class_labels=["Class0", "Class1", "Class2"],
        **overrides,
    )
    return explainer, x_test


def make_regression_explainer_from_dataset(regression_dataset, **overrides):
    """Create a regression calibrated explainer using the provided fixture."""
    from tests.helpers.model_utils import get_regression_model

    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,  # y_test
        _,  # num_features
        categorical_features,
        feature_names,
    ) = regression_dataset

    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        mode="regression",
        feature_names=feature_names,
        categorical_features=categorical_features,
        **overrides,
    )
    return explainer, x_test


def assert_explanation_collections_equal(lhs, rhs):
    """Assert that two explanation collections are equal down to prediction arrays."""
    lhs_items = getattr(lhs, "explanations", lhs)
    rhs_items = getattr(rhs, "explanations", rhs)
    assert len(lhs_items) == len(rhs_items)
    assert repr(lhs) == repr(rhs)
    for left, right in zip(lhs_items, rhs_items):
        for key in ("predict", "low", "high"):
            np.testing.assert_allclose(left.prediction[key], right.prediction[key])
            np.testing.assert_allclose(left.feature_weights[key], right.feature_weights[key])


class FakeCalibrationEnvelope:
    """Deterministic calibration envelope used by fake explanations."""

    def __init__(self, confidence: int = 90, percentiles: Tuple[float, float] = None):
        self.confidence = confidence
        self.low_high_percentiles = percentiles

    def get_confidence(self) -> int:
        """Return the configured confidence level."""
        return self.confidence


class FakeExplainer:
    """Minimal explainer stub for plotting tests."""

    _plot_plugin_fallbacks: Mapping[str, tuple[str, ...]] = {"alternative": ()}
    _last_explanation_mode: str = "alternative"

    def __init__(self, is_multiclass_flag: bool = False) -> None:
        self.learner = object()
        self.y_cal = np.array([0.1, 0.2, 0.3])
        self.is_multiclass_flag = is_multiclass_flag

    def predict(self, x, uq_interval=False, bins=None, **kwargs):
        """Return deterministic predictions with zero/one bounds."""
        preds = np.zeros(len(x))
        low = np.zeros(len(x))
        high = np.ones(len(x))
        return preds, (low, high)

    def is_multiclass(self) -> bool:
        """Report whether this fake explainer is marked as multiclass."""
        return self.is_multiclass_flag


class FakeNonProbExplainer:
    """Explainer without ``predict_proba`` to drive the non-probabilistic path."""

    def __init__(self) -> None:
        self.learner = object()
        self.y_cal = np.array([0.1, 0.2, 0.3])

    def predict(self, x, uq_interval=False, **kwargs):
        """Return a linear ramp of predictions plus fixed intervals."""
        size = len(x)
        preds = np.linspace(0.2, 0.8, size)
        low = preds - 0.1
        high = preds + 0.1
        return preds, (low, high)

    def is_multiclass(self) -> bool:
        """Return False for this non-probabilistic fake."""
        return False


class FakeProbExplainer:
    """Explainer exposing ``predict_proba`` to trigger the probabilistic branch."""

    def __init__(self, *, classes: Iterable[int] = (0, 1)) -> None:
        self.learner = self
        self.class_labels = dict(enumerate(classes))

    def predict_proba(self, x, uq_interval=False, threshold=None, **kwargs):
        """Return constant probability arrays and placeholder intervals."""
        proba = np.full((len(x), len(self.class_labels)), 0.5)
        low = np.zeros_like(proba)
        high = np.ones_like(proba)
        return proba, (low, high)

    def is_multiclass(self) -> bool:
        """Indicate whether the configured class labels represent multiclass data."""
        return len(self.class_labels) > 2

    def predict(self, x, uq_interval=False, **kwargs):
        """Return ramped predictions with fixed uncertainty bounds."""
        preds = np.linspace(0.1, 0.9, len(x))
        low = preds - 0.05
        high = preds + 0.05
        return preds, (low, high)


class FakeExplanation:
    """Light-weight explanation carrying attributes accessed by the plots."""

    def __init__(
        self,
        mode: str = "classification",
        *,
        thresholded: bool = False,
        y_threshold: Union[float, Tuple[float, float]] = 0.5,
        class_labels: Sequence[str] | None = None,
        is_multiclass: bool = False,
        y_minmax: Tuple[float, float] = (0.0, 1.0),
        percentiles: Tuple[float, float] = None,
        confidence: int = 90,
    ) -> None:
        self.mode = mode
        self.thresholded = thresholded
        self.y_threshold = y_threshold
        self.class_labels = list(class_labels) if class_labels is not None else None
        self.y_minmax = y_minmax
        self.low_high_percentiles = percentiles
        self.prediction = {"classes": 1}
        self.is_multiclass = is_multiclass
        self.explainer = FakeExplainer(is_multiclass_flag=is_multiclass)
        self.calibrated_explanations = FakeCalibrationEnvelope(confidence, percentiles)

    def get_explainer(self) -> FakeExplainer:
        """Return the embedded fake explainer."""
        return self.explainer

    def get_mode(self) -> str:
        """Return the explanation mode that was set."""
        return self.mode

    def get_class_labels(self) -> Sequence[str] | None:
        """Return the configured class labels or None."""
        return self.class_labels

    def is_thresholded(self) -> bool:
        """Indicate whether the explanation is thresholded."""
        return self.thresholded

    def is_one_sided(self) -> bool:
        """Indicate whether the explanation produces one-sided intervals."""
        return False


def generic_test(cal_exp, x_prop_train, y_prop_train, x, y):
    """Run a generic calibrated explainer test routine.

    This function encapsulates repeated assertions used across several
    test modules. Tests should import and call this helper rather than
    importing from another test module.
    """
    cal_exp.fit(x_prop_train, y_prop_train)
    assert cal_exp.fitted
    assert cal_exp.calibrated

    learner = cal_exp.learner

    from calibrated_explanations.core import WrapCalibratedExplainer

    new_exp = WrapCalibratedExplainer(learner)
    assert new_exp.fitted
    assert not new_exp.calibrated
    assert new_exp.learner == learner

    explainer = getattr(cal_exp, "explainer", None)
    if explainer is not None:
        new_exp = WrapCalibratedExplainer(explainer)
        assert new_exp.fitted
        assert new_exp.calibrated
        assert new_exp.explainer == explainer
        assert new_exp.learner == learner

    cal_exp.plot(x, show=False)
    cal_exp.plot(x, y, show=False)
    return cal_exp


def initiate_explainer(
    model,
    x_cal,
    y_cal,
    feature_names,
    categorical_features,
    mode,
    class_labels=None,
    difficulty_estimator=None,
    bins=None,
    fast=False,
    verbose=False,
    **kwargs,
):
    """Initialize a CalibratedExplainer instance."""
    from calibrated_explanations.core import CalibratedExplainer

    return CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        feature_names=feature_names,
        categorical_features=categorical_features,
        mode=mode,
        class_labels=class_labels,
        bins=bins,
        fast=fast,
        difficulty_estimator=difficulty_estimator,
        verbose=verbose,
        **kwargs,
    )
