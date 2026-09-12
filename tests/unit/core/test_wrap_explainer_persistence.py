"""Persistence coverage for WrapCalibratedExplainer ADR-031 safe (schema v3) APIs."""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.preprocessing.builtin_encoder import BuiltinEncoder
from calibrated_explanations.utils.exceptions import (
    IncompatibleStateError,
    ValidationError,
)


def assert_payload_close(left: Any, right: Any) -> None:
    """Recursively compare array-like persistence payloads."""
    if isinstance(left, tuple) and isinstance(right, tuple):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_payload_close(left_item, right_item)
        return
    np.testing.assert_allclose(np.asarray(left), np.asarray(right))


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Persist a JSON manifest payload."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_save_and_load_state_roundtrip_classification(tmp_path: Path) -> None:
    """Round-trip persistence preserves calibrated binary classification predictions."""
    x, y = make_classification(
        n_samples=96,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=7,
    )
    x_train, y_train = x[:48], y[:48]
    x_cal, y_cal = x[48:72], y[48:72]
    x_test = x[72:84]

    learner = RandomForestClassifier(n_estimators=24, random_state=3)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal, seed=13)
    baseline = wrapper.predict_proba(x_test, uq_interval=True)

    state_dir = tmp_path / "classification_state"
    wrapper.save_state(state_dir)
    restored = WrapCalibratedExplainer.load_state(state_dir, learner=learner)
    reloaded = restored.predict_proba(x_test, uq_interval=True)

    assert_payload_close(baseline, reloaded)


def test_save_and_load_state_roundtrip_multiclass(tmp_path: Path) -> None:
    """Round-trip persistence preserves calibrated multiclass predictions."""
    x, y = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=9,
    )
    x_train, y_train = x[:75], y[:75]
    x_cal, y_cal = x[75:115], y[75:115]
    x_test = x[115:130]

    learner = RandomForestClassifier(n_estimators=24, random_state=4)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal, seed=13)
    baseline = wrapper.predict_proba(x_test, uq_interval=True)

    state_dir = tmp_path / "multiclass_state"
    wrapper.save_state(state_dir)
    restored = WrapCalibratedExplainer.load_state(state_dir, learner=learner)
    reloaded = restored.predict_proba(x_test, uq_interval=True)

    assert_payload_close(baseline, reloaded)


def test_save_and_load_state_roundtrip_regression(tmp_path: Path) -> None:
    """Round-trip persistence preserves calibrated probabilistic regression payloads."""
    x, y = make_regression(n_samples=120, n_features=5, noise=0.2, random_state=11)
    x_train, y_train = x[:60], y[:60]
    x_cal, y_cal = x[60:90], y[60:90]
    x_test = x[90:105]
    threshold = float(np.median(y_cal))

    learner = RandomForestRegressor(n_estimators=30, random_state=5)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal, mode="regression", seed=17)
    baseline = wrapper.predict_proba(x_test, threshold=threshold, uq_interval=True)

    state_dir = tmp_path / "regression_state"
    wrapper.save_state(state_dir)
    restored = WrapCalibratedExplainer.load_state(state_dir, learner=learner)
    reloaded = restored.predict_proba(x_test, threshold=threshold, uq_interval=True)

    assert_payload_close(baseline, reloaded)


def test_should_preserve_predictions_when_reloaded_with_same_bins(tmp_path: Path) -> None:
    """Explicit Mondrian/conditional bins should round-trip through persistence unchanged."""
    x, y = make_classification(
        n_samples=96,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=21,
    )
    x_train, y_train = x[:48], y[:48]
    x_cal, y_cal = x[48:72], y[48:72]
    x_test = x[72:84]
    bins_cal = (x_cal[:, 0] >= 0).astype(int)
    bins_test = (x_test[:, 0] >= 0).astype(int)

    learner = RandomForestClassifier(n_estimators=18, random_state=8)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal, bins=bins_cal)
    baseline = wrapper.predict_proba(x_test, bins=bins_test)

    state_dir = tmp_path / "explicit_bins_state"
    wrapper.save_state(state_dir)
    restored = WrapCalibratedExplainer.load_state(state_dir, learner=learner)
    reloaded = restored.predict_proba(x_test, bins=bins_test)

    assert_payload_close(baseline, reloaded)


def test_save_and_load_state_roundtrip_builtin_preprocessing(tmp_path: Path) -> None:
    """Built-in preprocessing mapping round-trips without a supplied preprocessor.

    The built-in ``BuiltinEncoder`` is fully JSON-safe (a deterministic
    per-column category->int mapping), so ``load_state()`` reconstructs it
    automatically from the persisted mapping snapshot rather than requiring
    the caller to supply it, unlike an arbitrary custom preprocessor.
    """
    from calibrated_explanations.preprocessing.builtin_encoder import BuiltinEncoder

    # BuiltinEncoder treats every column as categorical, so use a small,
    # fixed category pool tiled deterministically across rows/columns: every
    # 24-row split then contains the full category set for every column,
    # avoiding spurious "unseen category" failures unrelated to persistence.
    rng = np.random.default_rng(23)
    categories = np.array(["red", "green", "blue"])
    n_samples, n_features = 96, 4
    indices = (np.arange(n_samples)[:, None] + np.arange(n_features)[None, :]) % len(categories)
    x = categories[indices]
    y = (rng.random(n_samples) > 0.5).astype(int)

    x_train, y_train = x[:48], y[:48]
    x_cal, y_cal = x[48:72], y[48:72]
    x_test = x[72:84]

    learner = RandomForestClassifier(n_estimators=16, random_state=2)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.preprocessor = BuiltinEncoder()
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal, seed=5)
    baseline = wrapper.predict_proba(x_test, uq_interval=True)

    state_dir = tmp_path / "builtin_preprocessing_state"
    wrapper.save_state(state_dir)
    restored = WrapCalibratedExplainer.load_state(state_dir, learner=learner)
    reloaded = restored.predict_proba(x_test, uq_interval=True)

    assert_payload_close(baseline, reloaded)


def test_save_and_load_state_roundtrip_builtin_preprocessing_with_mixed_types(
    tmp_path: Path,
) -> None:
    """Built-in preprocessing should round-trip on mixed numeric/categorical inputs."""
    rng = np.random.default_rng(24)
    numeric = np.linspace(0.0, 1.0, 96)
    categories = np.array(["red", "green", "blue"])
    x = np.column_stack(
        [
            numeric,
            categories[np.arange(96) % len(categories)],
            numeric * 2.0,
            categories[(np.arange(96) + 1) % len(categories)],
        ]
    ).astype(object)
    y = (rng.random(96) > 0.5).astype(int)

    x_train, y_train = x[:48], y[:48]
    x_cal, y_cal = x[48:72], y[48:72]
    x_test = x[72:84]

    learner = RandomForestClassifier(n_estimators=16, random_state=3)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.preprocessor = BuiltinEncoder()
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal, seed=6)
    baseline = wrapper.predict_proba(x_test, uq_interval=True)

    state_dir = tmp_path / "builtin_preprocessing_mixed_state"
    wrapper.save_state(state_dir)
    restored = WrapCalibratedExplainer.load_state(state_dir, learner=learner)
    reloaded = restored.predict_proba(x_test, uq_interval=True)

    assert_payload_close(baseline, reloaded)


class _Task45RoundTripPreprocessor:
    """Deterministic numeric preprocessor used to prove fail-fast preprocessing
    behavior survives a save_state/load_state round trip (bug-list/pre-v4 S4-B1)."""

    def fit_transform(self, x: Any) -> Any:
        return np.asarray(x, dtype=float) + 1.0

    def transform(self, x: Any) -> Any:
        return np.asarray(x, dtype=float) + 1.0


class _Task45AlwaysFailsTransform:
    def transform(self, x: Any) -> Any:
        raise RuntimeError("boom-transform")


def test_reloaded_state_still_fails_fast_when_preprocessor_transform_fails(
    tmp_path: Path,
) -> None:
    """A rejected preprocessing call must fail fast even after a state round trip,
    and must not disturb the restored wrapper's calibrated lifecycle state."""
    x, y = make_classification(
        n_samples=96,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=7,
    )
    x_train, y_train = x[:48], y[:48]
    x_cal, y_cal = x[48:72], y[48:72]
    x_test = x[72:84]

    learner = RandomForestClassifier(n_estimators=24, random_state=3)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.preprocessor = _Task45RoundTripPreprocessor()
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal, seed=13)

    state_dir = tmp_path / "preprocessing_failure_state"
    wrapper.save_state(state_dir)
    restored = WrapCalibratedExplainer.load_state(
        state_dir, learner=learner, preprocessor=_Task45RoundTripPreprocessor()
    )
    assert restored.calibrated is True

    restored.preprocessor = _Task45AlwaysFailsTransform()
    with pytest.raises(ValidationError, match="Preprocessor transform failed during inference"):
        restored.predict_proba(x_test)

    # The rejected call must not silently fall back to raw features or
    # disturb the restored wrapper's calibrated lifecycle state.
    assert restored.calibrated is True


def test_load_state_requires_custom_preprocessor_to_be_supplied(tmp_path: Path) -> None:
    """A custom (non-built-in) preprocessor cannot be reconstructed from persisted data."""
    x, y = make_classification(
        n_samples=64,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=7,
    )
    learner = RandomForestClassifier(n_estimators=12, random_state=3)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.preprocessor = _Task45RoundTripPreprocessor()
    wrapper.fit(x[:32], y[:32])
    wrapper.calibrate(x[32:48], y[32:48], seed=13)

    state_dir = tmp_path / "custom_preprocessor_missing"
    wrapper.save_state(state_dir)

    with pytest.raises(ValidationError, match="custom preprocessor"):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)


def test_should_pickle_silently_when_conditional_mc_is_absent(tmp_path: Path) -> None:
    """Wrappers without mc should serialize without fallback warnings."""
    x, y = make_classification(
        n_samples=64,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=17,
    )
    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=12, random_state=6))
    wrapper.fit(x[:32], y[:32])
    wrapper.calibrate(x[32:48], y[32:48], bins=(x[32:48, 0] >= 0).astype(int))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pickle_payload = pickle.dumps(wrapper)
        wrapper.save_state(tmp_path / "silent_state")

    assert pickle_payload
    assert caught == []


def test_save_state_writes_schema_version_3_manifest(tmp_path: Path) -> None:
    """save_state writes the current safe (ADR-031 v3) schema version."""
    x, y = make_classification(
        n_samples=64,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=29,
    )
    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=12, random_state=4))
    wrapper.fit(x[:32], y[:32])
    wrapper.calibrate(x[32:48], y[32:48], seed=11)

    state_dir = tmp_path / "schema_v3_state"
    wrapper.save_state(state_dir)
    manifest = json.loads((state_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 3
    assert set(manifest["files"]) <= {
        "explainer_state.json",
        "calibrator_primitive.json",
        "preprocessing_mapping.json",
    }
    assert "wrapper.pkl" not in manifest["files"]
    assert not (state_dir / "wrapper.pkl").exists()


@pytest.mark.parametrize("legacy_schema_version", [1, 2])
def test_load_state_rejects_legacy_schema_versions(
    tmp_path: Path, legacy_schema_version: int
) -> None:
    """load_state() refuses legacy (pickle-based) schema v1/v2 artifacts unconditionally."""
    x, y = make_classification(
        n_samples=64,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=31,
    )
    learner = RandomForestClassifier(n_estimators=12, random_state=5)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.fit(x[:32], y[:32])
    wrapper.calibrate(x[32:48], y[32:48], seed=13)

    state_dir = tmp_path / f"schema_v{legacy_schema_version}_state"
    wrapper.save_state(state_dir)
    manifest_path = state_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = legacy_schema_version
    write_manifest(manifest_path, manifest)

    with pytest.raises(IncompatibleStateError, match="legacy"):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)


def test_load_state_rejects_checksum_mismatch(tmp_path: Path) -> None:
    """Tampering with persisted files fails checksum verification."""
    x, y = make_classification(
        n_samples=64,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=19,
    )
    learner = RandomForestClassifier(n_estimators=16, random_state=1)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.fit(x[:32], y[:32])
    wrapper.calibrate(x[32:48], y[32:48], seed=5)

    state_dir = tmp_path / "checksum_state"
    wrapper.save_state(state_dir)
    with (state_dir / "explainer_state.json").open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(IncompatibleStateError, match="checksum"):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)


def test_load_state_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    """Unsupported manifest schema versions fail fast with actionable errors."""
    x, y = make_classification(
        n_samples=64,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=23,
    )
    learner = RandomForestClassifier(n_estimators=12, random_state=2)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.fit(x[:32], y[:32])
    wrapper.calibrate(x[32:48], y[32:48], seed=7)

    state_dir = tmp_path / "schema_state"
    wrapper.save_state(state_dir)
    manifest_path = state_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    write_manifest(manifest_path, manifest)

    with pytest.raises(IncompatibleStateError, match="schema_version"):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)


def test_load_state_requires_a_learner(tmp_path: Path) -> None:
    """load_state() fails clearly when no runtime learner is supplied."""
    x, y = make_classification(
        n_samples=64,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=41,
    )
    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=12, random_state=1))
    wrapper.fit(x[:32], y[:32])
    wrapper.calibrate(x[32:48], y[32:48], seed=3)

    state_dir = tmp_path / "missing_learner_state"
    wrapper.save_state(state_dir)

    with pytest.raises(ValidationError, match="learner"):
        WrapCalibratedExplainer.load_state(state_dir)


def test_load_state_rejects_learner_with_incompatible_task(tmp_path: Path) -> None:
    """A supplied learner exposing the wrong task (classifier vs regressor) fails clearly."""
    x, y = make_classification(
        n_samples=64,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=43,
    )
    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=12, random_state=1))
    wrapper.fit(x[:32], y[:32])
    wrapper.calibrate(x[32:48], y[32:48], seed=3)

    state_dir = tmp_path / "task_mismatch_state"
    wrapper.save_state(state_dir)

    wrong_task_learner = RandomForestRegressor(n_estimators=12, random_state=1)
    x_reg, y_reg = make_regression(n_samples=32, n_features=4, random_state=1)
    wrong_task_learner.fit(x_reg, y_reg)

    with pytest.raises(ValidationError, match="task"):
        WrapCalibratedExplainer.load_state(state_dir, learner=wrong_task_learner)


def test_load_state_rejects_learner_with_incompatible_feature_count(tmp_path: Path) -> None:
    """A supplied learner trained on a different feature count fails clearly."""
    x, y = make_classification(
        n_samples=64,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=47,
    )
    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=12, random_state=1))
    wrapper.fit(x[:32], y[:32])
    wrapper.calibrate(x[32:48], y[32:48], seed=3)

    state_dir = tmp_path / "feature_mismatch_state"
    wrapper.save_state(state_dir)

    mismatched_learner = RandomForestClassifier(n_estimators=12, random_state=1)
    x_wide, y_wide = make_classification(n_samples=32, n_features=8, random_state=1)
    mismatched_learner.fit(x_wide, y_wide)

    with pytest.raises(ValidationError, match="feature"):
        WrapCalibratedExplainer.load_state(state_dir, learner=mismatched_learner)
