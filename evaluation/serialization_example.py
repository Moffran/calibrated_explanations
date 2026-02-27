"""Comprehensive serialization round-trip example for WrapCalibratedExplainer.

Covers:
1. ADR-031 state persistence (`save_state` / `load_state`)
2. Wrapper pickle round-trip
3. Wrapper joblib round-trip
4. Explanation object pickle round-trip
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Force sequential runtime to avoid parallel fallback warnings in restricted environments.
os.environ["CE_PARALLEL"] = "sequential"

from calibrated_explanations import WrapCalibratedExplainer


def _assert_payload_close(left, right) -> None:
    if isinstance(left, tuple) and isinstance(right, tuple):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            _assert_payload_close(lhs, rhs)
        return
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if np.issubdtype(left_arr.dtype, np.number) and np.issubdtype(right_arr.dtype, np.number):
        np.testing.assert_allclose(left_arr, right_arr, atol=1e-9)
        return
    assert np.array_equal(left_arr, right_arr)


def _assert_explanations_equivalent(left, right) -> None:
    assert len(left) == len(right)
    for lhs, rhs in zip(left, right, strict=True):
        assert repr(lhs) == repr(rhs)


def main() -> None:
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size=0.4, random_state=0)
    x_cal, x_test, y_cal, _ = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=0)

    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=20, random_state=0))
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal)

    baseline_proba = wrapper.predict_proba(x_test[:8], uq_interval=True)
    baseline_pred = wrapper.predict(x_test[:8])
    baseline_explanations = wrapper.explain_factual(x_test[:2])

    work_dir = Path(tempfile.mkdtemp(prefix="ce-serialization-demo-"))

    # -------------------------------------------------------------------------
    # 1) ADR-031 state persistence
    # -------------------------------------------------------------------------
    state_dir = work_dir / "explainer_state"
    wrapper.save_state(state_dir)

    manifest_path = state_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest.get("schema_version"), int)
    assert isinstance(manifest.get("created_at_utc"), str)
    assert isinstance(manifest.get("files"), dict)
    assert "wrapper.pkl" in manifest["files"]
    assert "calibrator_primitive.json" in manifest["files"]

    restored_state = WrapCalibratedExplainer.load_state(state_dir)
    restored_state_proba = restored_state.predict_proba(x_test[:8], uq_interval=True)
    restored_state_pred = restored_state.predict(x_test[:8])
    restored_state_explanations = restored_state.explain_factual(x_test[:2])

    _assert_payload_close(baseline_proba, restored_state_proba)
    _assert_payload_close(baseline_pred, restored_state_pred)
    _assert_explanations_equivalent(baseline_explanations, restored_state_explanations)
    print("ADR-031 round-trip: OK")

    # -------------------------------------------------------------------------
    # 2) Wrapper pickle round-trip
    # -------------------------------------------------------------------------
    wrapper_pickle_path = work_dir / "explainer.pkl"
    with wrapper_pickle_path.open("wb") as handle:
        pickle.dump(wrapper, handle)
    with wrapper_pickle_path.open("rb") as handle:
        restored_pickle = pickle.load(handle)

    restored_pickle_proba = restored_pickle.predict_proba(x_test[:8], uq_interval=True)
    restored_pickle_pred = restored_pickle.predict(x_test[:8])
    restored_pickle_explanations = restored_pickle.explain_factual(x_test[:2])
    _assert_payload_close(baseline_proba, restored_pickle_proba)
    _assert_payload_close(baseline_pred, restored_pickle_pred)
    _assert_explanations_equivalent(baseline_explanations, restored_pickle_explanations)
    print("Wrapper pickle round-trip: OK")

    # -------------------------------------------------------------------------
    # 3) Wrapper joblib round-trip
    # -------------------------------------------------------------------------
    wrapper_joblib_path = work_dir / "explainer.joblib"
    joblib.dump(wrapper, wrapper_joblib_path)
    restored_joblib = joblib.load(wrapper_joblib_path)

    restored_joblib_proba = restored_joblib.predict_proba(x_test[:8], uq_interval=True)
    restored_joblib_pred = restored_joblib.predict(x_test[:8])
    restored_joblib_explanations = restored_joblib.explain_factual(x_test[:2])
    _assert_payload_close(baseline_proba, restored_joblib_proba)
    _assert_payload_close(baseline_pred, restored_joblib_pred)
    _assert_explanations_equivalent(baseline_explanations, restored_joblib_explanations)
    print("Wrapper joblib round-trip: OK")

    # -------------------------------------------------------------------------
    # 4) Explanation object pickle round-trip
    # -------------------------------------------------------------------------
    explanation_pickle_path = work_dir / "explanations.pkl"
    with explanation_pickle_path.open("wb") as handle:
        pickle.dump(baseline_explanations, handle)
    with explanation_pickle_path.open("rb") as handle:
        restored_explanations = pickle.load(handle)

    _assert_explanations_equivalent(baseline_explanations, restored_explanations)
    print("Explanation object pickle round-trip: OK")

    print(f"All serialization paths succeeded. Artifacts in: {work_dir}")


if __name__ == "__main__":
    main()
