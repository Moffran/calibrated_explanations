import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from unittest.mock import MagicMock

from calibrated_explanations.core import prediction_helpers as ph
from calibrated_explanations.core.explain._computation import explain_predict_step
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.utils.exceptions import DataShapeError, ValidationError


def test_prediction_helpers_round_trip():
    data = load_iris()
    x_train, x_cal, y_train, y_cal = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    clf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    clf.fit(x_train, y_train)
    explainer = CalibratedExplainer(clf, x_cal, y_cal, mode="classification", seed=42)

    x_test = x_cal[:2]
    # Delegated validation should produce identical array
    x_valid = ph.validate_and_prepare_input(explainer, x_test)
    assert np.array_equal(x_valid, x_test)

    explanation = ph.initialize_explanation(
        explainer, x_valid, (5, 95), None, None, features_to_ignore=None
    )
    assert explanation is not None

    # Need a discretizer assigned prior to predict step
    explainer.set_discretizer("binaryEntropy")
    # Predict step returns tuple; sanity check shape length invariants
    predict, low, high, prediction, *_rest = explain_predict_step(
        explainer, x_valid, None, (5, 95), None, features_to_ignore=[]
    )
    # The internal predict step includes perturbed instances, so length should be >= original
    assert len(predict) >= len(x_valid)
    assert len(low) == len(predict)
    assert len(high) == len(predict)
    # The base prediction vector stored in prediction dict should align with original test size
    base_pred_len = (
        len(prediction["predict"])
        if hasattr(prediction["predict"], "__len__")
        else prediction["predict"].shape[0]
    )  # type: ignore[index]
    assert base_pred_len == len(x_valid)
    assert "__full_probabilities__" in prediction
    assert len(prediction["__full_probabilities__"]) == len(x_valid)


class StubExplainer:
    """Lightweight implementation of ``_ExplainerProtocol`` for unit tests."""

    def __init__(
        self,
        *,
        num_features: int = 3,
        mode: str = "classification",
        mondrian: bool = False,
        fast: bool = False,
        multiclass: bool = True,
    ) -> None:
        self.num_features = num_features
        self.mode = mode
        self.mondrian_flag = mondrian
        self.fast_flag = fast
        self.multiclass_flag = multiclass
        self.x_cal = np.zeros((2, num_features))
        self.interval_learner = self.build_interval_learner()
        self.learner_instance = (
            self.interval_learner[0]
            if isinstance(self.interval_learner, list)
            else self.interval_learner
        )
        self.predict_calls: list = []
        self.discretizer = None  # Required by explain_predict_step

    def build_interval_learner(self):
        class Learner:
            def __init__(self, *, as_list: bool) -> None:
                self.as_list = as_list
                self.calls = []

            def predict_proba(self, x, bins=None):
                self.calls.append((np.asarray(x), bins))
                return np.full((len(x), 2), 0.5)

        learner = Learner(as_list=self.fast_flag)
        if self.fast_flag:
            # ``explain_predict_step`` indexes the learner list by ``num_features``
            return [learner for _ in range(self.num_features + 1)]
        return learner

    @property
    def plugin_manager(self):
        from calibrated_explanations.plugins.manager import PluginManager

        if not hasattr(self, "_plugin_manager"):
            self.plugin_manager = PluginManager(self)
        return self.plugin_manager

    # ``_ExplainerProtocol`` API -------------------------------------------------
    def is_mondrian(self) -> bool:  # noqa: D401 - protocol implementation
        return self.mondrian_flag

    def infer_explanation_mode(self) -> str:
        return "factual"

    def is_multiclass(self) -> bool:  # noqa: D401 - protocol implementation
        return self.multiclass_flag

    def is_fast(self) -> bool:  # noqa: D401 - protocol implementation
        return self.fast_flag

    def predict(self, x, **kwargs):  # noqa: D401 - protocol implementation
        self.predict_calls.append((np.asarray(x), kwargs))
        size = np.asarray(x).shape[0]
        classes = np.arange(size) if self.multiclass_flag else np.zeros(size, dtype=int)
        return (
            np.full((size,), 0.1),
            np.full((size,), -0.1),
            np.full((size,), 0.2),
            classes,
        )

    def predict_calibrated(self, x, **kwargs):
        return self.predict(x, **kwargs)

    def _predict(self, *args, **kwargs):
        """Internal alias for predict."""
        return self.predict(*args, **kwargs)

    def discretize(self, x):  # noqa: D401 - protocol implementation
        return np.asarray(x) + 1

    def rule_boundaries(self, x, x_perturbed):  # noqa: D401 - protocol implementation
        return {"rule": "boundaries", "shape": np.asarray(x_perturbed).shape}


def test_validate_and_prepare_input_reshapes_vector():
    explainer = StubExplainer(num_features=3)
    vector = np.array([1.0, 2.0, 3.0])

    prepared = ph.validate_and_prepare_input(explainer, vector)

    assert prepared.shape == (1, 3)
    assert isinstance(prepared, np.ndarray)


def test_validate_and_prepare_input_rejects_wrong_width():
    explainer = StubExplainer(num_features=4)
    matrix = np.ones((2, 3))

    with pytest.raises(DataShapeError):
        ph.validate_and_prepare_input(explainer, matrix)


def test_initialize_explanation_requires_bins_for_mondrian(monkeypatch):
    explainer = StubExplainer(mondrian=True)
    x = np.ones((2, explainer.num_features))

    # Avoid constructing the heavy ``CalibratedExplanations`` implementation.
    class Collection:
        def __init__(self, *_args, **_kwargs) -> None:
            self.low_high_percentiles = None

    # Patch the import inside the initialize_explanation function
    import calibrated_explanations.explanations as exp_module  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(exp_module, "CalibratedExplanations", Collection)

    with pytest.raises(ValidationError):
        ph.initialize_explanation(
            explainer,
            x,
            (10, 90),
            threshold=None,
            bins=None,
            features_to_ignore=None,
        )


def test_initialize_explanation_validates_bin_length(monkeypatch):
    explainer = StubExplainer(mondrian=True)
    x = np.ones((2, explainer.num_features))

    class Collection:
        def __init__(self, *_args, **_kwargs) -> None:
            self.low_high_percentiles = None

    import calibrated_explanations.explanations as exp_module  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(exp_module, "CalibratedExplanations", Collection)

    with pytest.raises(DataShapeError):
        ph.initialize_explanation(
            explainer,
            x,
            (5, 95),
            threshold=None,
            bins=np.ones((1,)),
            features_to_ignore=None,
        )


def test_initialize_explanation_rejects_threshold_for_classification(monkeypatch):
    explainer = StubExplainer(mode="classification")
    x = np.ones((2, explainer.num_features))

    class Collection:
        def __init__(self, *_args, **_kwargs) -> None:
            self.low_high_percentiles = None

    import calibrated_explanations.explanations as exp_module  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(exp_module, "CalibratedExplanations", Collection)

    with pytest.raises(ValidationError):
        ph.initialize_explanation(
            explainer,
            x,
            (5, 95),
            threshold=0.5,
            bins=None,
            features_to_ignore=None,
        )


def test_initialize_explanation_handles_regression_thresholds(monkeypatch):
    explainer = StubExplainer(mode="regression")
    x = np.ones((2, explainer.num_features))
    bins = np.ones((2,))
    threshold = [(0.1, 0.2), (0.3, 0.4)]

    recorded_calls: list[tuple] = []

    def fake_assert_mock(thresh, data):
        recorded_calls.append((thresh, np.asarray(data).shape))
        return thresh

    class Collection:
        def __init__(self, *_args, **_kwargs) -> None:
            self.low_high_percentiles = None

    import calibrated_explanations.explanations as exp_module  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(exp_module, "CalibratedExplanations", Collection)
    monkeypatch.setattr(ph, "assert_threshold", fake_assert_mock)

    with pytest.warns(UserWarning, match="list of interval thresholds"):
        explanation = ph.initialize_explanation(
            explainer,
            x,
            (5, 95),
            threshold=threshold,
            bins=bins,
            features_to_ignore=None,
        )

    assert explanation.low_high_percentiles is None
    assert recorded_calls == [(threshold, x.shape)]


def test_initialize_explanation_sets_percentiles_without_threshold(monkeypatch):
    explainer = StubExplainer(mode="regression")
    x = np.ones((3, explainer.num_features))

    class Collection:
        def __init__(self, *_args, **_kwargs) -> None:
            self.low_high_percentiles = None

    import calibrated_explanations.explanations as exp_module  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(exp_module, "CalibratedExplanations", Collection)

    explanation = ph.initialize_explanation(
        explainer,
        x,
        (25, 75),
        threshold=None,
        bins=None,
        features_to_ignore=None,
    )

    assert explanation.low_high_percentiles == (25, 75)


def test_predict_internal_delegates_to_underlying_protocol():
    explainer = StubExplainer(multiclass=False)
    x = np.ones((2, explainer.num_features))

    result = ph.predict_internal(
        explainer,
        x,
        threshold=0.3,
        low_high_percentiles=(1, 99),
        classes=[1],
        bins=np.array([0, 1]),
        feature=0,
    )

    assert explainer.predict_calls
    call_x, kwargs = explainer.predict_calls[-1]
    np.testing.assert_array_equal(call_x, x)
    assert kwargs["threshold"] == 0.3
    assert kwargs["low_high_percentiles"] == (1, 99)
    assert kwargs["classes"] == [1]
    np.testing.assert_array_equal(kwargs["bins"], np.array([0, 1]))
    assert kwargs["feature"] == 0
    assert isinstance(result, tuple)


def test_format_regression_prediction_handles_thresholds():
    predict = np.array([1.0, 2.0])
    low = np.array([0.5, 1.5])
    high = np.array([1.5, 2.5])

    labels = ph.format_regression_prediction(predict, low, high, threshold=1.5)
    assert isinstance(labels, list)
    assert all("y_hat" in label for label in labels)

    multi_labels = ph.format_regression_prediction(
        predict, low, high, threshold=[(0.5, 1.5), (0.1, 0.9)]
    )
    assert isinstance(multi_labels, list)

    interval_result = ph.format_regression_prediction(
        predict, low, high, threshold=None, uq_interval=True
    )
    assert isinstance(interval_result, tuple)


def test_format_classification_prediction_maps_labels():
    predict = np.array([0.6, 0.4])
    low = np.zeros_like(predict)
    high = np.ones_like(predict)
    new_classes = None
    class_labels = np.array(["neg", "pos"])

    mapped = ph.format_classification_prediction(
        predict,
        low,
        high,
        new_classes,
        is_multiclass_val=False,
        class_labels=class_labels,
        uq_interval=True,
    )
    assert isinstance(mapped, tuple)
    assert mapped[0].tolist() == ["pos", "neg"]


def test_handle_uncalibrated_regression_prediction():
    learner = MagicMock()
    learner.predict.return_value = np.array([1.0, 2.0])
    x = np.ones((2, 2))

    with pytest.raises(ValidationError):
        ph.handle_uncalibrated_regression_prediction(learner, x, threshold=1.0)

    result = ph.handle_uncalibrated_regression_prediction(learner, x, uq_interval=True)
    assert isinstance(result, tuple)


def test_handle_uncalibrated_classification_prediction():
    learner = MagicMock()
    learner.predict.return_value = np.array([0, 1])
    x = np.ones((2, 2))

    with pytest.raises(ValidationError):
        ph.handle_uncalibrated_classification_prediction(learner, x, threshold=0.5)

    result = ph.handle_uncalibrated_classification_prediction(learner, x, uq_interval=True)
    assert isinstance(result, tuple)
