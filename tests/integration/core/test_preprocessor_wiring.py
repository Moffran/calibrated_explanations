"""Tests for controlled preprocessing wiring in WrapCalibratedExplainer.

These tests use simple stubs and monkeypatching to validate that when a
user-supplied preprocessor is provided via ExplainerConfig, it's used to:
- fit/transform training data before learner.fit
- fit/transform (or transform) calibration data before CalibratedExplainer
- transform inference data before explain_* calls

When no preprocessor is provided, the default ``auto_encode='auto'`` path must
encode mixed-type input end to end through the public API (issue #202), while
all-numeric input reaches the learner unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations.api.config import ExplainerBuilder, ExplainerConfig
from calibrated_explanations.core import wrap_explainer as we
from calibrated_explanations.utils.exceptions import ValidationError

pytestmark = pytest.mark.integration


class DummyPreprocessor:
    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor
        self.fitted = False

    def fit_transform(self, x):
        self.fitted = True
        return np.asarray(x) * self.factor

    def transform(self, x):
        assert self.fitted
        return np.asarray(x) * self.factor


class SnapshotPreprocessor(DummyPreprocessor):
    def __init__(self, factor: float = 2.0, snapshot: dict[str, float] | None = None) -> None:
        super().__init__(factor)
        self.snapshot_data = snapshot or {"factor": factor}

    def get_mapping_snapshot(self):
        return self.snapshot_data


class StubModel:
    def __init__(self) -> None:
        self.last_fit_X = None
        self.fitted_ = False  # sklearn-style fitted marker

    # provide predict_proba so wrapper picks classification mode
    def predict_proba(self, x):  # pragma: no cover - not used
        return np.zeros((len(x), 2))

    def predict(self, x):  # minimal implementation for validation
        return np.zeros(len(x))

    def fit(self, x, y, **kwargs):
        self.last_fit_X = np.asarray(x)
        self.fitted_ = True


def test_preprocessor_applied_on_calibrate_and_inference(monkeypatch):
    model = StubModel()
    pre = DummyPreprocessor(factor=3.0)
    cfg = ExplainerConfig(model=model, preprocessor=pre)
    w = we.WrapCalibratedExplainer.from_config(cfg)
    w.fitted = True

    captured = {}

    class DummyCE:
        def __init__(self, learner, x_cal, y_cal, **kwargs):  # noqa: D401
            captured["x_cal"] = np.asarray(x_cal)

        def explain_factual(self, x, **kwargs):  # noqa: D401
            captured["x_test"] = np.asarray(x)
            return np.asarray(x)

    monkeypatch.setattr(we, "CalibratedExplainer", DummyCE)

    x_cal = np.array([[1.0, 1.0], [2.0, 2.0]])
    y_cal = np.array([0, 1])
    w.calibrate(x_cal, y_cal)
    np.testing.assert_allclose(captured["x_cal"], x_cal * 3.0)

    # Use the installed DummyCE to check inference transform
    x_test = np.array([[4.0, 5.0]])
    out = w.explain_factual(x_test)
    np.testing.assert_allclose(out, x_test * 3.0)


def test_preprocessor_metadata_exposed_in_telemetry():
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    x, y = make_classification(n_samples=120, n_features=6, random_state=0)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.4, random_state=42, stratify=y
    )
    x_cal, x_test, y_cal, _ = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=24, stratify=y_temp
    )

    pre = SnapshotPreprocessor(factor=1.3)
    cfg = (
        ExplainerBuilder(RandomForestClassifier(n_estimators=10, random_state=0))
        .preprocessor(pre)
        .auto_encode(True)
        .build_config()
    )
    wrapper = we.WrapCalibratedExplainer.from_config(cfg)
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal)

    batch = wrapper.explain_factual(x_test[:3])
    telemetry = getattr(batch, "telemetry", {})
    meta = telemetry.get("preprocessor")
    assert meta is not None
    assert meta.get("auto_encode") == "true"
    expected_transformer = f"{pre.__class__.__module__}:{pre.__class__.__qualname__}"
    assert meta.get("transformer_id") == expected_transformer
    snapshot = meta.get("mapping_snapshot")
    assert snapshot is not None
    assert snapshot.get("custom", {}).get("factor") == pre.factor
    assert wrapper.explainer is not None
    runtime_meta = wrapper.explainer.runtime_telemetry.get("preprocessor")
    assert runtime_meta == meta


def _mixed_frame(n: int = 240) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "color": rng.choice(["red", "green", "blue"], n),
            "size": pd.Categorical(rng.choice(["S", "M", "L"], n)),
            "age": rng.integers(18, 80, n),
        }
    )
    y = ((frame["num"] > 0) ^ (frame["color"] == "green")).astype(int).to_numpy()
    return frame, y


@pytest.mark.parametrize(
    "learner, regression",
    [
        (RandomForestClassifier(n_estimators=10, random_state=0), False),
        (RandomForestRegressor(n_estimators=10, random_state=0), True),
    ],
)
def test_default_wrapper_should_fit_calibrate_and_explain_mixed_dataframe(learner, regression):
    # Arrange
    x, y = _mixed_frame()
    target = x["num"].to_numpy() * 2.0 + y if regression else y
    wrapper = we.WrapCalibratedExplainer(learner)
    assert wrapper.auto_encode == "auto"

    # Act
    wrapper.fit(x.iloc[:120], target[:120])
    wrapper.calibrate(x.iloc[120:200], target[120:200])
    explanation = wrapper.explain_factual(x.iloc[200:205])
    predictions = wrapper.predict(x.iloc[200:205])

    # Assert
    assert len(explanation) == 5
    assert len(predictions) == 5
    snapshot = wrapper.export_preprocessor_mapping()
    assert snapshot is not None
    assert set(snapshot) == {"col_1", "col_2"}
    assert snapshot["col_1"] == ["blue", "green", "red"]


def test_default_wrapper_should_pass_all_numeric_dataframe_through_unchanged():
    # Arrange
    x, y = _mixed_frame()
    numeric = x[["num", "age"]]
    learner = RandomForestClassifier(n_estimators=10, random_state=0)
    wrapper = we.WrapCalibratedExplainer(learner)

    # Act
    wrapper.fit(numeric.iloc[:120], y[:120])
    wrapper.calibrate(numeric.iloc[120:200], y[120:200])

    # Assert
    assert wrapper.export_preprocessor_mapping() is None
    assert list(learner.feature_names_in_) == ["num", "age"]


def test_default_wrapper_should_reject_unseen_category_at_inference():
    # Arrange
    x, y = _mixed_frame()
    wrapper = we.WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=0))
    wrapper.fit(x.iloc[:120], y[:120])
    wrapper.calibrate(x.iloc[120:200], y[120:200])
    unseen = x.iloc[200:201].copy()
    unseen["color"] = "purple"

    # Act / Assert
    with pytest.raises(ValidationError):
        wrapper.predict(unseen)


def test_disabled_auto_encode_should_raise_validation_error_on_mixed_dataframe():
    # Arrange
    x, y = _mixed_frame()
    wrapper = we.WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=0))
    wrapper.auto_encode = False

    # Act / Assert
    with pytest.raises(ValidationError, match="auto_encode='auto'"):
        wrapper.fit(x.iloc[:120], y[:120])
