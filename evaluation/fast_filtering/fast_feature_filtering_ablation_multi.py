"""Multi-dataset ablation for fast feature filtering."""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder

from evaluation.fast_filtering.dataset_utils import load_dataset, resolve_dataset_specs


def _build_model(task: str, model_name: str):
    if task == "classification":
        if model_name == "RF":
            return RandomForestClassifier(n_estimators=200, random_state=42)
        if model_name == "HGB":
            return HistGradientBoostingClassifier(random_state=42)
    if task == "regression":
        if model_name == "RF":
            return RandomForestRegressor(n_estimators=200, random_state=42)
        if model_name == "HGB":
            return HistGradientBoostingRegressor(random_state=42)
    raise ValueError(f"Unsupported model '{model_name}' for task '{task}'.")


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    test_size: float,
    calibration_size: float,
    random_state: int,
):
    stratify = y if task == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    stratify_train = y_train if task == "classification" else None
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_train,
        y_train,
        test_size=calibration_size,
        random_state=random_state,
        stratify=stratify_train,
    )
    return x_train, x_cal, y_train, y_cal, x_test, y_test


def _build_explainer(learner: Any, *, task: str, enable_filtering: bool, top_k: int):
    builder = ExplainerBuilder(learner)
    builder = builder.task(task).perf_parallel(False)
    if enable_filtering:
        builder = builder.perf_feature_filter(True, per_instance_top_k=top_k)
    else:
        builder = builder.perf_feature_filter(False)
    config = builder.build_config()
    return WrapCalibratedExplainer.from_config(config)


def _timed_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn()
    duration = time.perf_counter() - start
    return result, duration


def _run_lime(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    model: Any,
    task: str,
    feature_names: list[str],
):
    from lime.lime_tabular import LimeTabularExplainer

    mode = "classification" if task == "classification" else "regression"
    explainer = LimeTabularExplainer(x_train, feature_names=feature_names, mode=mode)

    if task == "classification":
        predictor = model.predict_proba
    else:
        predictor = model.predict

    for x in x_test:
        explainer.explain_instance(x, predictor, num_features=x_test.shape[1])


def _run_shap(*, x_train: np.ndarray, x_test: np.ndarray, model: Any):
    import shap

    explainer = shap.Explainer(model, x_train)
    _ = explainer(x_test)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast filtering ablation across datasets")
    parser.add_argument("--tasks", nargs="+", default=["classification", "multiclass", "regression"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=["RF", "HGB"])
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--calibration-size", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-explain", type=int, default=250)
    parser.add_argument("--fast-filter-top-k", type=int, default=8)
    parser.add_argument("--include-fast", action="store_true")
    parser.add_argument("--include-alternatives", action="store_true")
    parser.add_argument("--allow-regression-alternatives", action="store_true")
    parser.add_argument("--include-lime", action="store_true")
    parser.add_argument("--include-shap", action="store_true")
    parser.add_argument(
        "--results-file",
        type=str,
        default="evaluation/fast_filtering/fast_filtering_ablation_multi_results.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    specs = resolve_dataset_specs(args.tasks, limit=args.limit, dataset_names=args.datasets)

    results: dict[str, Any] = {
        "meta": {
            "tasks": args.tasks,
            "models": args.models,
            "test_size": args.test_size,
            "calibration_size": args.calibration_size,
            "max_samples": args.max_samples,
            "max_explain": args.max_explain,
            "fast_filter_top_k": args.fast_filter_top_k,
            "include_fast": args.include_fast,
            "include_alternatives": args.include_alternatives,
            "include_lime": args.include_lime,
            "include_shap": args.include_shap,
            "timestamp": time.time(),
        },
        "datasets": {},
    }

    for spec in specs:
        print(f"Dataset: {spec.name} ({spec.task})")
        X, y, feature_names = load_dataset(spec, max_samples=args.max_samples)
        task = "classification" if spec.task == "classification" else "regression"
        x_train, x_cal, y_train, y_cal, x_test, _ = _split_data(
            X,
            y,
            task=task,
            test_size=args.test_size,
            calibration_size=args.calibration_size,
            random_state=42,
        )
        if args.max_explain is not None and len(x_test) > args.max_explain:
            x_test = x_test[: args.max_explain]

        results["datasets"].setdefault(spec.name, {"task": task, "models": {}})

        for model_name in args.models:
            model = _build_model(task, model_name)
            model.fit(x_train, y_train)
            model_results: dict[str, Any] = {"runs": {}}

            for filtering in (False, True):
                label = "with_filtering" if filtering else "without_filtering"
                print(f"  {model_name}: {label}")
                try:
                    explainer = _build_explainer(
                        model,
                        task=task,
                        enable_filtering=filtering,
                        top_k=args.fast_filter_top_k,
                    )
                    _, init_duration = _timed_call(lambda: explainer.calibrate(x_cal, y_cal))

                    explain_results: dict[str, Any] = {"init": init_duration}

                    _, factual_time = _timed_call(lambda: explainer.explain_factual(x_test))
                    explain_results["explain_factual"] = factual_time

                    if args.include_alternatives and (
                        task == "classification" or args.allow_regression_alternatives
                    ):
                        _, alt_time = _timed_call(lambda: explainer.explore_alternatives(x_test))
                        explain_results["explore_alternatives"] = alt_time

                    if args.include_fast:
                        _, fast_time = _timed_call(lambda: explainer.explain_fast(x_test))
                        explain_results["explain_fast"] = fast_time

                    model_results["runs"][label] = explain_results
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    model_results["runs"][label] = {"error": str(exc)}

            comparators: dict[str, Any] = {}
            if args.include_lime:
                try:
                    _, lime_time = _timed_call(
                        lambda: _run_lime(
                            x_train=x_train,
                            x_test=x_test,
                            model=model,
                            task=task,
                            feature_names=feature_names,
                        )
                    )
                    comparators["lime"] = {"explain_factual": lime_time}
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    comparators["lime"] = {"error": str(exc)}

            if args.include_shap:
                try:
                    _, shap_time = _timed_call(
                        lambda: _run_shap(x_train=x_train, x_test=x_test, model=model)
                    )
                    comparators["shap"] = {"explain_factual": shap_time}
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    comparators["shap"] = {"error": str(exc)}

            if comparators:
                model_results["comparators"] = comparators

            results["datasets"][spec.name]["models"][model_name] = model_results

    print(f"Saving results to {args.results_file}...")
    with open(args.results_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
