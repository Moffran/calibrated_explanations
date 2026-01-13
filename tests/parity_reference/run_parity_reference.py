"""Parity reference harness for canonical fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.explanations import legacy_to_domain
from calibrated_explanations.explanations.explanation import (
    AlternativeExplanation,
    FastExplanation,
    FactualExplanation,
)
from calibrated_explanations.plugins import ensure_builtin_plugins
from calibrated_explanations.serialization import to_json
from calibrated_explanations.testing import parity_compare

ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_serializable(payload), indent=2, sort_keys=True))


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_serializable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def build_payload(exp: Any) -> dict[str, Any]:
    return {
        "task": getattr(exp, "get_mode", lambda: "unknown")(),
        "rules": getattr(exp, "rules", {"rule": [], "feature": []}),
        "feature_weights": getattr(exp, "feature_weights", {}),
        "feature_predict": getattr(exp, "feature_predict", {}),
        "prediction": getattr(exp, "prediction", {}),
        "explanation_type": _explanation_type_from_instance(exp),
    }


def _explanation_type_from_instance(exp: Any) -> str:
    if isinstance(exp, AlternativeExplanation):
        return "alternative"
    if isinstance(exp, FastExplanation):
        return "fast"
    if isinstance(exp, FactualExplanation):
        return "factual"
    return "factual"


def build_explainer(dataset: dict[str, Any], *, fast: bool = False) -> CalibratedExplainer:
    mode = dataset.get("mode", "classification")
    x_train = np.array(dataset.get("x_train", []), dtype=float)
    y_train = np.array(dataset.get("y_train", []))

    if mode == "classification":
        model = DecisionTreeClassifier(random_state=42)
    else:
        from sklearn.tree import DecisionTreeRegressor

        model = DecisionTreeRegressor(random_state=42)
        y_train = y_train.astype(float)

    if len(x_train) > 0:
        model.fit(x_train, y_train)

    y_cal_arr = np.array(dataset["y_cal"], dtype=float) if mode == "regression" else np.array(dataset["y_cal"])
    return CalibratedExplainer(
        model,
        np.array(dataset["x_cal"], dtype=float),
        y_cal_arr,
        mode=mode,
        feature_names=dataset.get("feature_names"),
        categorical_features=dataset.get("categorical_features", []),
        class_labels=dataset.get("class_labels"),
        seed=42,
        fast=fast,
    )


def compute_outputs(dataset: dict[str, Any]) -> dict[str, Any]:
    x_test = np.array(dataset["x_test"], dtype=float)
    explainer = build_explainer(dataset)

    predictions = explainer.predict(x_test).tolist()
    factual = explainer.explain_factual(x_test)
    alternatives = explainer.explore_alternatives(x_test)

    ensure_builtin_plugins()
    fast_explainer = build_explainer(dataset, fast=True)
    fast = fast_explainer.explain_fast(x_test)

    def to_payload(exps: Any) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for exp in exps:
            core_payload = legacy_to_domain(exp.index, build_payload(exp))
            try:
                # Try the validated JSON path, but fall back to a primitive payload
                # built from `build_payload` if schema validation fails for modes
                # that include fast outputs.
                payloads.append(to_json(core_payload))
            except Exception:
                payloads.append(to_serializable(build_payload(exp)))
        return payloads

    return {
        "predictions": to_serializable(predictions),
        "factual": to_serializable(to_payload(factual)),
        "alternatives": to_serializable(to_payload(alternatives)),
        "fast": to_serializable(to_payload(fast)),
    }


def compare_payload(name: str, expected: Any, actual: Any) -> list[dict[str, Any]]:
    diffs = parity_compare(expected, actual, rtol=1e-6, atol=1e-8)
    if diffs:
        print(f"::group::{name} parity diffs")
        for diff in diffs:
            print(json.dumps(diff, indent=2, sort_keys=True))
        print("::endgroup::")
    return diffs


def run(dataset_name: str = "classification", update: bool = False) -> int:
    dataset_path = ROOT / f"canonical_dataset_{dataset_name}.json" if dataset_name != "classification" else ROOT / "canonical_dataset.json"
    if not dataset_path.exists():
        print(f"Dataset fixture not found: {dataset_path}")
        return 2

    dataset = load_json(dataset_path)
    outputs = compute_outputs(dataset)

    fixtures = {
        "predictions": ROOT / f"predictions_{dataset_name}.json" if dataset_name != "classification" else ROOT / "predictions.json",
        "factual": ROOT / f"factual_{dataset_name}.json" if dataset_name != "classification" else ROOT / "factual.json",
        "alternatives": ROOT / f"alternatives_{dataset_name}.json" if dataset_name != "classification" else ROOT / "alternatives.json",
        "fast": ROOT / f"fast_{dataset_name}.json" if dataset_name != "classification" else ROOT / "fast.json",
    }

    if update:
        for key, path in fixtures.items():
            dump_json(path, outputs[key])
        print("Parity fixtures updated.")
        return 0

    diffs: list[dict[str, Any]] = []
    for key, path in fixtures.items():
        expected = load_json(path)
        diffs.extend(compare_payload(key, expected, outputs[key]))

    if diffs:
        print(f"Detected {len(diffs)} parity diff(s).")
        return 1
    print("Parity reference fixtures match.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parity reference harness.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update parity fixtures with current outputs.",
    )
    parser.add_argument(
        "--dataset",
        choices=["classification", "regression", "multiclass", "probabilistic_regression"],
        default="classification",
        help="Select canonical dataset to run.",
    )
    args = parser.parse_args()
    raise SystemExit(run(dataset_name=args.dataset, update=args.update))


if __name__ == "__main__":
    main()
