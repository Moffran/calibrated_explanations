"""Persistence coverage for WrapCalibratedExplainer ADR-031 APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.utils.exceptions import IncompatibleStateError


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
    """Round-trip persistence preserves calibrated classification predictions."""
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

    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=24, random_state=3))
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal, seed=13)
    baseline = wrapper.predict_proba(x_test, uq_interval=True)

    state_dir = tmp_path / "classification_state"
    wrapper.save_state(state_dir)
    restored = WrapCalibratedExplainer.load_state(state_dir)
    reloaded = restored.predict_proba(x_test, uq_interval=True)

    assert_payload_close(baseline, reloaded)


def test_save_and_load_state_roundtrip_regression(tmp_path: Path) -> None:
    """Round-trip persistence preserves calibrated probabilistic regression payloads."""
    x, y = make_regression(n_samples=120, n_features=5, noise=0.2, random_state=11)
    x_train, y_train = x[:60], y[:60]
    x_cal, y_cal = x[60:90], y[60:90]
    x_test = x[90:105]
    threshold = float(np.median(y_cal))

    wrapper = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=30, random_state=5))
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal, mode="regression", seed=17)
    baseline = wrapper.predict_proba(x_test, threshold=threshold, uq_interval=True)

    state_dir = tmp_path / "regression_state"
    wrapper.save_state(state_dir)
    restored = WrapCalibratedExplainer.load_state(state_dir)
    reloaded = restored.predict_proba(x_test, threshold=threshold, uq_interval=True)

    assert_payload_close(baseline, reloaded)


def test_load_state_rejects_checksum_mismatch(tmp_path: Path) -> None:
    """Tampering with persisted files fails checksum verification."""
    x, y = make_classification(
        n_samples=64,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=19,
    )
    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=16, random_state=1))
    wrapper.fit(x[:32], y[:32])
    wrapper.calibrate(x[32:48], y[32:48], seed=5)

    state_dir = tmp_path / "checksum_state"
    wrapper.save_state(state_dir)
    with (state_dir / "wrapper.pkl").open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(IncompatibleStateError, match="checksum"):
        WrapCalibratedExplainer.load_state(state_dir)


def test_load_state_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    """Unsupported manifest schema versions fail fast with actionable errors."""
    x, y = make_classification(
        n_samples=64,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=23,
    )
    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=12, random_state=2))
    wrapper.fit(x[:32], y[:32])
    wrapper.calibrate(x[32:48], y[32:48], seed=7)

    state_dir = tmp_path / "schema_state"
    wrapper.save_state(state_dir)
    manifest_path = state_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    write_manifest(manifest_path, manifest)

    with pytest.raises(IncompatibleStateError, match="schema_version"):
        WrapCalibratedExplainer.load_state(state_dir)
