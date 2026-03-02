"""Comprehensive serialization round-trip examples for WrapCalibratedExplainer.

Covers:
1. Classification round-trip:
   - ADR-031 state persistence (`save_state` / `load_state`)
   - Wrapper pickle round-trip
   - Wrapper joblib round-trip
   - Explanation JSON round-trip
   - Explanation object pickle round-trip
2. Regression round-trip:
   - ADR-031 state persistence (`save_state` / `load_state`)
   - Wrapper pickle round-trip
   - Wrapper joblib round-trip
   - Explanation JSON round-trip
   - Explanation object pickle round-trip
3. Multiclass round-trip:
   - ADR-031 state persistence (`save_state` / `load_state`)
   - Wrapper pickle round-trip
   - Wrapper joblib round-trip
   - Explanation JSON round-trip for:
     * `explain_factual(..., multi_labels_enabled=False)`
     * `explain_factual(..., multi_labels_enabled=True)`
     * `explore_alternatives(..., multi_labels_enabled=False)`
     * `explore_alternatives(..., multi_labels_enabled=True)`
   - Explanation object pickle round-trip for the same outputs
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# Force sequential runtime to avoid parallel fallback warnings in restricted environments.
os.environ["CE_PARALLEL"] = "sequential"

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.serialization import to_json as explanation_to_json


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


def _assert_json_payload_equivalent(left, right) -> None:
    left_json = json.dumps(left, sort_keys=True)
    right_json = json.dumps(right, sort_keys=True)
    assert left_json == right_json


def _assert_exported_explanations_match_payload(explanation_payload, exported_collection) -> None:
    exported_payload = [explanation_to_json(item) for item in exported_collection.explanations]
    _assert_json_payload_equivalent(explanation_payload, exported_payload)


def main() -> None:
    # -------------------------------------------------------------------------
    # Classification example
    # -------------------------------------------------------------------------
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size=0.4, random_state=0)
    x_cal, x_test, y_cal, _ = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=0)

    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=20, random_state=0))
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal)

    baseline_proba = wrapper.predict_proba(x_test[:8], uq_interval=True)
    baseline_pred = wrapper.predict(x_test[:8])
    baseline_explanations = wrapper.explain_factual(x_test[:2])

    work_dir = Path(tempfile.mkdtemp(prefix="ce-serialization-demo-classification-"))

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
    # 4) Explanation JSON round-trip
    # -------------------------------------------------------------------------
    explanation_json_payload = baseline_explanations.to_json()
    restored_json_explanations = baseline_explanations.__class__.from_json(explanation_json_payload)
    _assert_exported_explanations_match_payload(
        explanation_json_payload["explanations"],
        restored_json_explanations,
    )
    print("Explanation JSON round-trip: OK")

    # -------------------------------------------------------------------------
    # 5) Explanation object pickle round-trip
    # -------------------------------------------------------------------------
    explanation_pickle_path = work_dir / "explanations.pkl"
    with explanation_pickle_path.open("wb") as handle:
        pickle.dump(baseline_explanations, handle)
    with explanation_pickle_path.open("rb") as handle:
        restored_explanations = pickle.load(handle)

    _assert_explanations_equivalent(baseline_explanations, restored_explanations)
    print("Explanation object pickle round-trip: OK")

    print(f"All classification serialization paths succeeded. Artifacts in: {work_dir}")

    # -------------------------------------------------------------------------
    # Regression example
    # -------------------------------------------------------------------------
    x_reg, y_reg = load_diabetes(return_X_y=True)
    x_reg_train, x_reg_tmp, y_reg_train, y_reg_tmp = train_test_split(
        x_reg, y_reg, test_size=0.4, random_state=0
    )
    x_reg_cal, x_reg_test, y_reg_cal, _ = train_test_split(
        x_reg_tmp, y_reg_tmp, test_size=0.5, random_state=0
    )

    reg_wrapper = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=50, random_state=0))
    reg_wrapper.fit(x_reg_train, y_reg_train)
    reg_wrapper.calibrate(x_reg_cal, y_reg_cal)

    reg_baseline_pred = reg_wrapper.predict(x_reg_test[:8], uq_interval=True)
    reg_baseline_conformal = reg_wrapper.explain_factual(
        x_reg_test[:2], low_high_percentiles=(5, 95)
    )
    reg_baseline_probabilistic_single = reg_wrapper.explain_factual(x_reg_test[:2], threshold=150)
    reg_baseline_probabilistic_interval = reg_wrapper.explain_factual(
        x_reg_test[:2], threshold=(100, 200)
    )

    reg_work_dir = Path(tempfile.mkdtemp(prefix="ce-serialization-demo-regression-"))

    # -------------------------------------------------------------------------
    # 1) ADR-031 state persistence
    # -------------------------------------------------------------------------
    reg_state_dir = reg_work_dir / "explainer_state"
    reg_wrapper.save_state(reg_state_dir)

    reg_manifest_path = reg_state_dir / "manifest.json"
    reg_manifest = json.loads(reg_manifest_path.read_text(encoding="utf-8"))
    assert isinstance(reg_manifest.get("schema_version"), int)
    assert isinstance(reg_manifest.get("created_at_utc"), str)
    assert isinstance(reg_manifest.get("files"), dict)
    assert "wrapper.pkl" in reg_manifest["files"]
    assert "calibrator_primitive.json" in reg_manifest["files"]

    reg_restored_state = WrapCalibratedExplainer.load_state(reg_state_dir)
    reg_restored_state_pred = reg_restored_state.predict(x_reg_test[:8], uq_interval=True)
    reg_restored_state_conformal = reg_restored_state.explain_factual(
        x_reg_test[:2], low_high_percentiles=(5, 95)
    )
    reg_restored_state_probabilistic_single = reg_restored_state.explain_factual(
        x_reg_test[:2], threshold=150
    )
    reg_restored_state_probabilistic_interval = reg_restored_state.explain_factual(
        x_reg_test[:2], threshold=(100, 200)
    )

    _assert_payload_close(reg_baseline_pred, reg_restored_state_pred)
    _assert_explanations_equivalent(reg_baseline_conformal, reg_restored_state_conformal)
    _assert_explanations_equivalent(
        reg_baseline_probabilistic_single, reg_restored_state_probabilistic_single
    )
    _assert_explanations_equivalent(
        reg_baseline_probabilistic_interval, reg_restored_state_probabilistic_interval
    )
    print("Regression ADR-031 round-trip: OK")

    # -------------------------------------------------------------------------
    # 2) Wrapper pickle round-trip
    # -------------------------------------------------------------------------
    reg_wrapper_pickle_path = reg_work_dir / "explainer.pkl"
    with reg_wrapper_pickle_path.open("wb") as handle:
        pickle.dump(reg_wrapper, handle)
    with reg_wrapper_pickle_path.open("rb") as handle:
        reg_restored_pickle = pickle.load(handle)

    reg_restored_pickle_pred = reg_restored_pickle.predict(x_reg_test[:8], uq_interval=True)
    reg_restored_pickle_conformal = reg_restored_pickle.explain_factual(
        x_reg_test[:2], low_high_percentiles=(5, 95)
    )
    reg_restored_pickle_probabilistic_single = reg_restored_pickle.explain_factual(
        x_reg_test[:2], threshold=150
    )
    reg_restored_pickle_probabilistic_interval = reg_restored_pickle.explain_factual(
        x_reg_test[:2], threshold=(100, 200)
    )
    _assert_payload_close(reg_baseline_pred, reg_restored_pickle_pred)
    _assert_explanations_equivalent(reg_baseline_conformal, reg_restored_pickle_conformal)
    _assert_explanations_equivalent(
        reg_baseline_probabilistic_single, reg_restored_pickle_probabilistic_single
    )
    _assert_explanations_equivalent(
        reg_baseline_probabilistic_interval, reg_restored_pickle_probabilistic_interval
    )
    print("Regression wrapper pickle round-trip: OK")

    # -------------------------------------------------------------------------
    # 3) Wrapper joblib round-trip
    # -------------------------------------------------------------------------
    reg_wrapper_joblib_path = reg_work_dir / "explainer.joblib"
    joblib.dump(reg_wrapper, reg_wrapper_joblib_path)
    reg_restored_joblib = joblib.load(reg_wrapper_joblib_path)

    reg_restored_joblib_pred = reg_restored_joblib.predict(x_reg_test[:8], uq_interval=True)
    reg_restored_joblib_conformal = reg_restored_joblib.explain_factual(
        x_reg_test[:2], low_high_percentiles=(5, 95)
    )
    reg_restored_joblib_probabilistic_single = reg_restored_joblib.explain_factual(
        x_reg_test[:2], threshold=150
    )
    reg_restored_joblib_probabilistic_interval = reg_restored_joblib.explain_factual(
        x_reg_test[:2], threshold=(100, 200)
    )
    _assert_payload_close(reg_baseline_pred, reg_restored_joblib_pred)
    _assert_explanations_equivalent(reg_baseline_conformal, reg_restored_joblib_conformal)
    _assert_explanations_equivalent(
        reg_baseline_probabilistic_single, reg_restored_joblib_probabilistic_single
    )
    _assert_explanations_equivalent(
        reg_baseline_probabilistic_interval, reg_restored_joblib_probabilistic_interval
    )
    print("Regression wrapper joblib round-trip: OK")

    # -------------------------------------------------------------------------
    # 4) Explanation JSON round-trip
    # -------------------------------------------------------------------------
    reg_explanation_json_payload = reg_baseline_conformal.to_json()
    reg_restored_json_explanations = reg_baseline_conformal.__class__.from_json(
        reg_explanation_json_payload
    )
    _assert_exported_explanations_match_payload(
        reg_explanation_json_payload["explanations"],
        reg_restored_json_explanations,
    )
    reg_prob_single_json_payload = reg_baseline_probabilistic_single.to_json()
    reg_prob_single_restored_json = reg_baseline_probabilistic_single.__class__.from_json(
        reg_prob_single_json_payload
    )
    _assert_exported_explanations_match_payload(
        reg_prob_single_json_payload["explanations"],
        reg_prob_single_restored_json,
    )
    reg_prob_interval_json_payload = reg_baseline_probabilistic_interval.to_json()
    reg_prob_interval_restored_json = reg_baseline_probabilistic_interval.__class__.from_json(
        reg_prob_interval_json_payload
    )
    _assert_exported_explanations_match_payload(
        reg_prob_interval_json_payload["explanations"],
        reg_prob_interval_restored_json,
    )
    print("Regression explanation JSON round-trip: OK")

    # -------------------------------------------------------------------------
    # 5) Explanation object pickle round-trip
    # -------------------------------------------------------------------------
    reg_explanation_pickle_path = reg_work_dir / "regression_explanations.pkl"
    with reg_explanation_pickle_path.open("wb") as handle:
        pickle.dump(
            {
                "conformal": reg_baseline_conformal,
                "probabilistic_single": reg_baseline_probabilistic_single,
                "probabilistic_interval": reg_baseline_probabilistic_interval,
            },
            handle,
        )
    with reg_explanation_pickle_path.open("rb") as handle:
        reg_restored_explanations = pickle.load(handle)

    _assert_explanations_equivalent(reg_baseline_conformal, reg_restored_explanations["conformal"])
    _assert_explanations_equivalent(
        reg_baseline_probabilistic_single,
        reg_restored_explanations["probabilistic_single"],
    )
    _assert_explanations_equivalent(
        reg_baseline_probabilistic_interval,
        reg_restored_explanations["probabilistic_interval"],
    )
    print("Regression explanation object pickle round-trip: OK")

    print(f"All regression serialization paths succeeded. Artifacts in: {reg_work_dir}")

    # -------------------------------------------------------------------------
    # Multiclass example
    # -------------------------------------------------------------------------
    x_multi, y_multi = load_iris(return_X_y=True)
    x_multi_train, x_multi_tmp, y_multi_train, y_multi_tmp = train_test_split(
        x_multi,
        y_multi,
        test_size=0.4,
        random_state=0,
        stratify=y_multi,
    )
    x_multi_cal, x_multi_test, y_multi_cal, _ = train_test_split(
        x_multi_tmp,
        y_multi_tmp,
        test_size=0.5,
        random_state=0,
        stratify=y_multi_tmp,
    )

    multi_wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=50, random_state=0))
    multi_wrapper.fit(x_multi_train, y_multi_train)
    multi_wrapper.calibrate(x_multi_cal, y_multi_cal)

    multi_baseline_proba = multi_wrapper.predict_proba(x_multi_test[:8], uq_interval=True)
    multi_baseline_pred = multi_wrapper.predict(x_multi_test[:8])

    factual_single = multi_wrapper.explain_factual(x_multi_test[:2], multi_labels_enabled=False)
    factual_multi = multi_wrapper.explain_factual(x_multi_test[:2], multi_labels_enabled=True)
    alternative_single = multi_wrapper.explore_alternatives(
        x_multi_test[:2], multi_labels_enabled=False
    )
    alternative_multi = multi_wrapper.explore_alternatives(x_multi_test[:2], multi_labels_enabled=True)

    multi_work_dir = Path(tempfile.mkdtemp(prefix="ce-serialization-demo-multiclass-"))

    # -------------------------------------------------------------------------
    # 1) ADR-031 state persistence
    # -------------------------------------------------------------------------
    multi_state_dir = multi_work_dir / "explainer_state"
    multi_wrapper.save_state(multi_state_dir)

    multi_manifest_path = multi_state_dir / "manifest.json"
    multi_manifest = json.loads(multi_manifest_path.read_text(encoding="utf-8"))
    assert isinstance(multi_manifest.get("schema_version"), int)
    assert isinstance(multi_manifest.get("created_at_utc"), str)
    assert isinstance(multi_manifest.get("files"), dict)
    assert "wrapper.pkl" in multi_manifest["files"]
    assert "calibrator_primitive.json" in multi_manifest["files"]

    multi_restored_state = WrapCalibratedExplainer.load_state(multi_state_dir)
    multi_restored_state_proba = multi_restored_state.predict_proba(x_multi_test[:8], uq_interval=True)
    multi_restored_state_pred = multi_restored_state.predict(x_multi_test[:8])
    factual_single_state = multi_restored_state.explain_factual(
        x_multi_test[:2], multi_labels_enabled=False
    )
    factual_multi_state = multi_restored_state.explain_factual(
        x_multi_test[:2], multi_labels_enabled=True
    )
    alternative_single_state = multi_restored_state.explore_alternatives(
        x_multi_test[:2], multi_labels_enabled=False
    )
    alternative_multi_state = multi_restored_state.explore_alternatives(
        x_multi_test[:2], multi_labels_enabled=True
    )

    _assert_payload_close(multi_baseline_proba, multi_restored_state_proba)
    _assert_payload_close(multi_baseline_pred, multi_restored_state_pred)
    _assert_explanations_equivalent(factual_single, factual_single_state)
    _assert_explanations_equivalent(factual_multi, factual_multi_state)
    _assert_explanations_equivalent(alternative_single, alternative_single_state)
    _assert_explanations_equivalent(alternative_multi, alternative_multi_state)
    print("Multiclass ADR-031 round-trip: OK")

    # -------------------------------------------------------------------------
    # 2) Wrapper pickle round-trip
    # -------------------------------------------------------------------------
    multi_wrapper_pickle_path = multi_work_dir / "explainer.pkl"
    with multi_wrapper_pickle_path.open("wb") as handle:
        pickle.dump(multi_wrapper, handle)
    with multi_wrapper_pickle_path.open("rb") as handle:
        multi_restored_pickle = pickle.load(handle)

    multi_restored_pickle_proba = multi_restored_pickle.predict_proba(x_multi_test[:8], uq_interval=True)
    multi_restored_pickle_pred = multi_restored_pickle.predict(x_multi_test[:8])
    factual_single_pickle = multi_restored_pickle.explain_factual(
        x_multi_test[:2], multi_labels_enabled=False
    )
    factual_multi_pickle = multi_restored_pickle.explain_factual(
        x_multi_test[:2], multi_labels_enabled=True
    )
    alternative_single_pickle = multi_restored_pickle.explore_alternatives(
        x_multi_test[:2], multi_labels_enabled=False
    )
    alternative_multi_pickle = multi_restored_pickle.explore_alternatives(
        x_multi_test[:2], multi_labels_enabled=True
    )

    _assert_payload_close(multi_baseline_proba, multi_restored_pickle_proba)
    _assert_payload_close(multi_baseline_pred, multi_restored_pickle_pred)
    _assert_explanations_equivalent(factual_single, factual_single_pickle)
    _assert_explanations_equivalent(factual_multi, factual_multi_pickle)
    _assert_explanations_equivalent(alternative_single, alternative_single_pickle)
    _assert_explanations_equivalent(alternative_multi, alternative_multi_pickle)
    print("Multiclass wrapper pickle round-trip: OK")

    # -------------------------------------------------------------------------
    # 3) Wrapper joblib round-trip
    # -------------------------------------------------------------------------
    multi_wrapper_joblib_path = multi_work_dir / "explainer.joblib"
    joblib.dump(multi_wrapper, multi_wrapper_joblib_path)
    multi_restored_joblib = joblib.load(multi_wrapper_joblib_path)

    multi_restored_joblib_proba = multi_restored_joblib.predict_proba(x_multi_test[:8], uq_interval=True)
    multi_restored_joblib_pred = multi_restored_joblib.predict(x_multi_test[:8])
    factual_single_joblib = multi_restored_joblib.explain_factual(
        x_multi_test[:2], multi_labels_enabled=False
    )
    factual_multi_joblib = multi_restored_joblib.explain_factual(
        x_multi_test[:2], multi_labels_enabled=True
    )
    alternative_single_joblib = multi_restored_joblib.explore_alternatives(
        x_multi_test[:2], multi_labels_enabled=False
    )
    alternative_multi_joblib = multi_restored_joblib.explore_alternatives(
        x_multi_test[:2], multi_labels_enabled=True
    )

    _assert_payload_close(multi_baseline_proba, multi_restored_joblib_proba)
    _assert_payload_close(multi_baseline_pred, multi_restored_joblib_pred)
    _assert_explanations_equivalent(factual_single, factual_single_joblib)
    _assert_explanations_equivalent(factual_multi, factual_multi_joblib)
    _assert_explanations_equivalent(alternative_single, alternative_single_joblib)
    _assert_explanations_equivalent(alternative_multi, alternative_multi_joblib)
    print("Multiclass wrapper joblib round-trip: OK")

    # -------------------------------------------------------------------------
    # 4) Explanation JSON round-trip (single-label and multi-label modes)
    # -------------------------------------------------------------------------
    factual_single_json_payload = factual_single.to_json()
    factual_multi_json_payload = factual_multi.to_json()
    alternative_single_json_payload = alternative_single.to_json()
    alternative_multi_json_payload = alternative_multi.to_json()

    factual_single_from_json = factual_single.__class__.from_json(factual_single_json_payload)
    factual_multi_from_json = factual_multi.__class__.from_json(factual_multi_json_payload)
    alternative_single_from_json = alternative_single.__class__.from_json(alternative_single_json_payload)
    alternative_multi_from_json = alternative_multi.__class__.from_json(alternative_multi_json_payload)

    _assert_exported_explanations_match_payload(
        factual_single_json_payload["explanations"],
        factual_single_from_json,
    )
    _assert_exported_explanations_match_payload(
        factual_multi_json_payload["explanations"],
        factual_multi_from_json,
    )
    _assert_exported_explanations_match_payload(
        alternative_single_json_payload["explanations"],
        alternative_single_from_json,
    )
    _assert_exported_explanations_match_payload(
        alternative_multi_json_payload["explanations"],
        alternative_multi_from_json,
    )
    print("Multiclass explanation JSON round-trip (single-label + multi-label): OK")

    # -------------------------------------------------------------------------
    # 5) Explanation object pickle round-trip (single-label and multi-label modes)
    # -------------------------------------------------------------------------
    multi_explanations_pickle_path = multi_work_dir / "multiclass_explanations.pkl"
    with multi_explanations_pickle_path.open("wb") as handle:
        pickle.dump(
            {
                "factual_single": factual_single,
                "factual_multi": factual_multi,
                "alternative_single": alternative_single,
                "alternative_multi": alternative_multi,
            },
            handle,
        )
    with multi_explanations_pickle_path.open("rb") as handle:
        multi_restored_explanations = pickle.load(handle)

    _assert_explanations_equivalent(factual_single, multi_restored_explanations["factual_single"])
    _assert_explanations_equivalent(factual_multi, multi_restored_explanations["factual_multi"])
    _assert_explanations_equivalent(
        alternative_single,
        multi_restored_explanations["alternative_single"],
    )
    _assert_explanations_equivalent(
        alternative_multi,
        multi_restored_explanations["alternative_multi"],
    )
    print("Multiclass explanation object pickle round-trip (single-label + multi-label): OK")

    print(f"All multiclass serialization paths succeeded. Artifacts in: {multi_work_dir}")


if __name__ == "__main__":
    main()
