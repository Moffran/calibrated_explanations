"""Tests for controlled preprocessing wiring in WrapCalibratedExplainer.

These tests use simple stubs and monkeypatching to validate that when a
user-supplied preprocessor is provided via ExplainerConfig, it's used to:
- fit/transform training data before learner.fit
- fit/transform (or transform) calibration data before CalibratedExplainer
- transform inference data before explain_* calls

When no preprocessor is provided, behavior is unchanged (covered by existing tests).
"""

from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.api.config import ExplainerBuilder, ExplainerConfig
from calibrated_explanations.core import wrap_explainer as we


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


def test_preprocessor_applied_on_fit():
    model = StubModel()
    pre = DummyPreprocessor(factor=2.5)
    cfg = ExplainerConfig(model=model, preprocessor=pre)
    w = we.WrapCalibratedExplainer.from_config(cfg)

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    w.fit(x, y)

    assert model.last_fit_X is not None
    np.testing.assert_allclose(model.last_fit_X, x * 2.5)


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


def test_preprocessor_is_persistent_and_deterministic(monkeypatch):
    class RecordingPreprocessor(DummyPreprocessor):
        def __init__(self, factor: float = 2.0) -> None:
            super().__init__(factor)
            self.fit_calls = 0
            self.transform_calls = 0

        def fit_transform(self, x):  # noqa: D401
            self.fit_calls += 1
            return super().fit_transform(x)

        def transform(self, x):  # noqa: D401
            self.transform_calls += 1
            return super().transform(x)

    model = StubModel()
    pre = RecordingPreprocessor(factor=1.5)
    cfg = ExplainerConfig(model=model, preprocessor=pre)
    w = we.WrapCalibratedExplainer.from_config(cfg)

    # Fit path should call fit once
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    w.fit(x, y)
    assert pre.fit_calls == 1

    # Calibrate path should not re-fit if already fitted
    class DummyCE2:
        def __init__(self, learner, x_cal, y_cal, **kwargs):  # noqa: D401
            # ensure transform used and deterministic
            np.testing.assert_allclose(x_cal, np.asarray([[1.0, 1.0], [2.0, 2.0]]) * 1.5)

        def explain_factual(self, x, **kwargs):  # noqa: D401
            return np.asarray(x)

    monkeypatch.setattr(we, "CalibratedExplainer", DummyCE2)
    x_cal = np.array([[1.0, 1.0], [2.0, 2.0]])
    w.calibrate(x_cal, y)
    # Should have used transform only (no new fit)
    assert pre.fit_calls == 1
    assert pre.transform_calls >= 2  # calibrate + inference paths


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
