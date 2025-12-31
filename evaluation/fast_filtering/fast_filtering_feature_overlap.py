"""Measure feature overlap between full and fast-filtered explanations."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder

from evaluation.fast_filtering.dataset_utils import load_dataset, resolve_dataset_specs
from evaluation.fast_filtering.metrics_utils import compute_overlap_metrics, summarize_metrics


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


def _build_explainer(learner, *, task: str, enable_filtering: bool, top_k: int):
    builder = ExplainerBuilder(learner)
    builder = builder.task(task).perf_parallel(False)
    if enable_filtering:
        builder = builder.perf_feature_filter(True, per_instance_top_k=top_k)
    else:
        builder = builder.perf_feature_filter(False)
    config = builder.build_config()
    return WrapCalibratedExplainer.from_config(config)


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


def _collect_metrics(
    *,
    baseline_explanations,
    filtered_explanations,
    top_k: int,
):
    metrics = []
    for base_exp, filtered_exp in zip(
        baseline_explanations.explanations, filtered_explanations.explanations
    ):
        base_weights = np.asarray(base_exp.feature_weights["predict"]).reshape(-1)
        filtered_weights = np.asarray(filtered_exp.feature_weights["predict"]).reshape(-1)
        num_features = base_weights.shape[0]
        baseline_topk = base_exp.rank_features(base_weights, num_to_show=top_k)
        ignored = filtered_exp.ignored_features_for_instance()
        keep = set(range(num_features)) - set(int(i) for i in ignored)
        metrics.append(
            compute_overlap_metrics(
                baseline_weights=base_weights,
                filtered_weights=filtered_weights,
                baseline_topk=baseline_topk,
                filtered_keep=keep,
            )
        )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature overlap for fast filtering")
    parser.add_argument("--tasks", nargs="+", default=["classification", "multiclass", "regression"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=["RF", "HGB"])
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--calibration-size", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-explain", type=int, default=250)
    parser.add_argument("--fast-filter-top-k", type=int, default=8)
    parser.add_argument("--include-alternatives", action="store_true")
    parser.add_argument("--allow-regression-alternatives", action="store_true")
    parser.add_argument("--save-instance-metrics", action="store_true")
    parser.add_argument(
        "--results-file",
        type=str,
        default="evaluation/fast_filtering/fast_filtering_feature_overlap_results.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    specs = resolve_dataset_specs(args.tasks, limit=args.limit, dataset_names=args.datasets)

    results: dict[str, object] = {
        "meta": {
            "tasks": args.tasks,
            "models": args.models,
            "test_size": args.test_size,
            "calibration_size": args.calibration_size,
            "max_samples": args.max_samples,
            "max_explain": args.max_explain,
            "fast_filter_top_k": args.fast_filter_top_k,
            "include_alternatives": args.include_alternatives,
            "save_instance_metrics": args.save_instance_metrics,
            "timestamp": time.time(),
        },
        "datasets": {},
    }

    for spec in specs:
        print(f"Dataset: {spec.name} ({spec.task})")
        X, y, _ = load_dataset(spec, max_samples=args.max_samples)
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

            baseline_explainer = _build_explainer(
                model, task=task, enable_filtering=False, top_k=args.fast_filter_top_k
            )
            filtered_explainer = _build_explainer(
                model, task=task, enable_filtering=True, top_k=args.fast_filter_top_k
            )
            baseline_explainer.calibrate(x_cal, y_cal)
            filtered_explainer.calibrate(x_cal, y_cal)

            model_results: dict[str, object] = {}

            baseline_factual = baseline_explainer.explain_factual(x_test)
            filtered_factual = filtered_explainer.explain_factual(x_test)
            factual_metrics = _collect_metrics(
                baseline_explanations=baseline_factual,
                filtered_explanations=filtered_factual,
                top_k=args.fast_filter_top_k,
            )
            model_results["explain_factual"] = {
                "summary": summarize_metrics(factual_metrics),
            }
            if args.save_instance_metrics:
                model_results["explain_factual"]["instances"] = [m.__dict__ for m in factual_metrics]

            if args.include_alternatives and (
                task == "classification" or args.allow_regression_alternatives
            ):
                baseline_alt = baseline_explainer.explore_alternatives(x_test)
                filtered_alt = filtered_explainer.explore_alternatives(x_test)
                alt_metrics = _collect_metrics(
                    baseline_explanations=baseline_alt,
                    filtered_explanations=filtered_alt,
                    top_k=args.fast_filter_top_k,
                )
                model_results["explore_alternatives"] = {
                    "summary": summarize_metrics(alt_metrics),
                }
                if args.save_instance_metrics:
                    model_results["explore_alternatives"]["instances"] = [
                        m.__dict__ for m in alt_metrics
                    ]

            results["datasets"][spec.name]["models"][model_name] = model_results

    print(f"Saving results to {args.results_file}...")
    with open(args.results_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
