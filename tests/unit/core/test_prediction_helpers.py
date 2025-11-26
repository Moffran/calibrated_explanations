import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations.core import prediction_helpers as ph
from calibrated_explanations.core.explain._computation import explain_predict_step
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.exceptions import DataShapeError, ValidationError


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


class _StubExplainer:
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
        self._mondrian = mondrian
        self._fast = fast
        self._multiclass = multiclass
        self.x_cal = np.zeros((2, num_features))
        self.interval_learner = self._build_interval_learner()
        self._learner = (
            self.interval_learner[0]
            if isinstance(self.interval_learner, list)
            else self.interval_learner
        )
        self.predict_calls: list = []
        self.discretizer = None  # Required by explain_predict_step

    def _build_interval_learner(self):
        class _Learner:
            def __init__(self, *, as_list: bool) -> None:
                self.as_list = as_list
                self.calls = []

            def predict_proba(self, x, bins=None):
                self.calls.append((np.asarray(x), bins))
                return np.full((len(x), 2), 0.5)

        learner = _Learner(as_list=self._fast)
        if self._fast:
            # ``explain_predict_step`` indexes the learner list by ``num_features``
            return [learner for _ in range(self.num_features + 1)]
        return learner

    # ``_ExplainerProtocol`` API -------------------------------------------------
    def _is_mondrian(self) -> bool:  # noqa: D401 - protocol implementation
        return self._mondrian

    def is_multiclass(self) -> bool:  # noqa: D401 - protocol implementation
        return self._multiclass

    def is_fast(self) -> bool:  # noqa: D401 - protocol implementation
        return self._fast

    def _predict(self, x, **kwargs):  # noqa: D401 - protocol implementation
        self.predict_calls.append((np.asarray(x), kwargs))
        size = np.asarray(x).shape[0]
        classes = np.arange(size) if self._multiclass else np.zeros(size, dtype=int)
        return (
            np.full((size,), 0.1),
            np.full((size,), -0.1),
            np.full((size,), 0.2),
            classes,
        )

    def _discretize(self, x):  # noqa: D401 - protocol implementation
        return np.asarray(x) + 1

    def rule_boundaries(self, x, x_perturbed):  # noqa: D401 - protocol implementation
        return {"rule": "boundaries", "shape": np.asarray(x_perturbed).shape}


def test_validate_and_prepare_input_reshapes_vector():
    explainer = _StubExplainer(num_features=3)
    vector = np.array([1.0, 2.0, 3.0])

    prepared = ph.validate_and_prepare_input(explainer, vector)

    assert prepared.shape == (1, 3)
    assert isinstance(prepared, np.ndarray)


def test_validate_and_prepare_input_rejects_wrong_width():
    explainer = _StubExplainer(num_features=4)
    matrix = np.ones((2, 3))

    with pytest.raises(DataShapeError):
        ph.validate_and_prepare_input(explainer, matrix)


def test_initialize_explanation_requires_bins_for_mondrian(monkeypatch):
    explainer = _StubExplainer(mondrian=True)
    x = np.ones((2, explainer.num_features))

    # Avoid constructing the heavy ``CalibratedExplanations`` implementation.
    class _Collection:
        def __init__(self, *_args, **_kwargs) -> None:
            self.low_high_percentiles = None

    # Patch the import inside the initialize_explanation function
    import calibrated_explanations.explanations as exp_module  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(exp_module, "CalibratedExplanations", _Collection)

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
    explainer = _StubExplainer(mondrian=True)
    x = np.ones((2, explainer.num_features))

    class _Collection:
        def __init__(self, *_args, **_kwargs) -> None:
            self.low_high_percentiles = None

    import calibrated_explanations.explanations as exp_module  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(exp_module, "CalibratedExplanations", _Collection)

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
    explainer = _StubExplainer(mode="classification")
    x = np.ones((2, explainer.num_features))

    class _Collection:
        def __init__(self, *_args, **_kwargs) -> None:
            self.low_high_percentiles = None

    import calibrated_explanations.explanations as exp_module  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(exp_module, "CalibratedExplanations", _Collection)

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
    explainer = _StubExplainer(mode="regression")
    x = np.ones((2, explainer.num_features))
    bins = np.ones((2,))
    threshold = [(0.1, 0.2), (0.3, 0.4)]

    recorded_calls: list[tuple] = []

    def _fake_assert(thresh, data):
        recorded_calls.append((thresh, np.asarray(data).shape))
        return thresh

    class _Collection:
        def __init__(self, *_args, **_kwargs) -> None:
            self.low_high_percentiles = None

    import calibrated_explanations.explanations as exp_module  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(exp_module, "CalibratedExplanations", _Collection)
    monkeypatch.setattr(ph, "assert_threshold", _fake_assert)

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
    explainer = _StubExplainer(mode="regression")
    x = np.ones((3, explainer.num_features))

    class _Collection:
        def __init__(self, *_args, **_kwargs) -> None:
            self.low_high_percentiles = None

    import calibrated_explanations.explanations as exp_module  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(exp_module, "CalibratedExplanations", _Collection)

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
    explainer = _StubExplainer(multiclass=False)
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
