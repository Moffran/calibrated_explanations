"""Helper fixtures and mocks for explainer tests."""

from __future__ import annotations

import pytest
import numpy as np
from typing import Any, Iterable, Sequence, Tuple, Union, Mapping

from calibrated_explanations import CalibratedExplainer
from tests.helpers.model_utils import get_classification_model, DummyLearner, DummyIntervalLearner


def patch_interval_initializers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch the interval learner factories to return the dummy implementation."""

    def _initialize(explainer: CalibratedExplainer, *args: Any, **kwargs: Any) -> None:
        explainer.interval_learner = DummyIntervalLearner()
        explainer._CalibratedExplainer__initialized = True

    monkeypatch.setattr(
        "calibrated_explanations.calibration.interval_learner.initialize_interval_learner",
        _initialize,
    )
    monkeypatch.setattr(
        "calibrated_explanations.calibration.interval_learner.initialize_interval_learner_for_fast_explainer",
        _initialize,
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


def make_explainer_from_dataset(binary_dataset, **overrides):
    """Create a calibrated explainer from the binary_dataset fixture.

    Parameters
    ----------
    binary_dataset : tuple
        Fixture tuple produced by `make_binary_dataset`.
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
    ) = binary_dataset

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
        self._mode = mode
        self._thresholded = thresholded
        self.y_threshold = y_threshold
        self._class_labels = list(class_labels) if class_labels is not None else None
        self.y_minmax = y_minmax
        self.low_high_percentiles = percentiles
        self.prediction = {"classes": 1}
        self.is_multiclass = is_multiclass
        self._explainer = FakeExplainer(is_multiclass_flag=is_multiclass)
        self.calibrated_explanations = FakeCalibrationEnvelope(confidence, percentiles)

    def _get_explainer(self) -> FakeExplainer:
        """Return the embedded fake explainer."""
        return self._explainer

    def get_mode(self) -> str:
        """Return the explanation mode that was set."""
        return self._mode

    def get_class_labels(self) -> Sequence[str] | None:
        """Return the configured class labels or None."""
        return self._class_labels

    def is_thresholded(self) -> bool:
        """Indicate whether the explanation is thresholded."""
        return self._thresholded

    def is_one_sided(self) -> bool:
        """Indicate whether the explanation produces one-sided intervals."""
        return False
