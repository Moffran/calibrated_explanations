"""Extended setup ablation across multiple datasets and models."""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder

from evaluation.fast_filtering.dataset_utils import load_dataset, resolve_dataset_specs

SETUPS = ["legacy", "sequential", "instance_parallel", "caching", "fast_filtering"]


def _configure_parallel_env(
    *,
    enabled: bool,
    strategy: str | None = None,
    workers: int | None = None,
    instance_chunk_size: int | None = None,
) -> None:
    if not enabled:
        os.environ["CE_PARALLEL"] = "off"
        return

    tokens: list[str] = ["enable"]
    if strategy is not None:
        tokens.append(strategy)
    if workers is not None:
        tokens.append(f"workers={workers}")
    if instance_chunk_size is not None:
        tokens.append(f"instance_chunk_size={instance_chunk_size}")
    os.environ["CE_PARALLEL"] = ",".join(tokens)


def _configure_caching_env(*, enabled: bool) -> None:
    os.environ["CE_CACHE"] = "on" if enabled else "off"


def _cleanup_env() -> None:
    for key in ("CE_PARALLEL", "CE_CACHE"):
        os.environ.pop(key, None)


def _expected_execution_plugin_identifier() -> str:
    ce_parallel = os.environ.get("CE_PARALLEL", "off")

    if ce_parallel.lower() in ("off", "0", "false"):
        return "core.explanation.factual.sequential"

    tokens = [t.strip() for t in ce_parallel.split(",") if t.strip()]
    strategy = None
    for token in tokens:
        lower = token.lower()
        if lower in ("threads", "processes", "joblib", "sequential"):
            strategy = lower
            break

    if strategy == "sequential":
        return "core.explanation.factual.sequential"
    if strategy in ("threads", "processes", "joblib"):
        return "core.explanation.factual.instance_parallel"

    print(
        f"    Warning: Unable to determine parallel strategy from CE_PARALLEL='{ce_parallel}'. Defaulting to sequential."
    )
    return "core.explanation.factual.sequential"


def _build_fast_filter_explainer(learner: Any, *, task: str, top_k: int) -> WrapCalibratedExplainer:
    builder = ExplainerBuilder(learner)
    config = (
        builder.task(task)
        .perf_parallel(False)
        .perf_feature_filter(True, per_instance_top_k=top_k)
        .build_config()
    )
    return WrapCalibratedExplainer.from_config(config)


def _instantiate_explainer(
    learner: Any, *, setup: str, task: str, top_k: int
) -> WrapCalibratedExplainer:
    if setup == "fast_filtering":
        return _build_fast_filter_explainer(learner, task=task, top_k=top_k)
    return WrapCalibratedExplainer(learner)


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


def _run_setup(
    *,
    learner: Any,
    setup: str,
    task: str,
    workers: int,
    instance_chunk_size: int,
    x_cal: Any,
    y_cal: Any,
    x_test: Any,
    top_k: int,
) -> dict[str, float]:
    if setup == "instance_parallel":
        _configure_parallel_env(
            enabled=True,
            strategy="threads",
            workers=workers,
            instance_chunk_size=instance_chunk_size,
        )
    else:
        _configure_parallel_env(enabled=False)
    _configure_caching_env(enabled=(setup == "caching"))

    plugin_id = _expected_execution_plugin_identifier()

    try:
        init_start = time.perf_counter()
        explainer = _instantiate_explainer(learner, setup=setup, task=task, top_k=top_k)
        explainer.calibrate(x_cal, y_cal, factual_plugin=plugin_id)
        init_duration = time.perf_counter() - init_start

        explain_kwargs: dict[str, Any] = {}
        if setup == "legacy":
            explain_kwargs["_use_plugin"] = False

        explain_start = time.perf_counter()
        _ = explainer.explain_factual(x_test, **explain_kwargs)
        explain_duration = time.perf_counter() - explain_start

        total = init_duration + explain_duration
        print(
            f"    {setup}: init={init_duration:.2f}s explain={explain_duration:.2f}s total={total:.2f}s"
        )

        return {"init": init_duration, "explain": explain_duration, "total": total}
    finally:
        _cleanup_env()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup ablation across multiple datasets")
    parser.add_argument("--tasks", nargs="+", default=["classification", "multiclass", "regression"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=["RF", "HGB"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--instance-chunk-size", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--calibration-size", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fast-filter-top-k", type=int, default=8)
    parser.add_argument(
        "--results-file",
        type=str,
        default="evaluation/fast_filtering/setup_ablation_multi_results.json",
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
            "fast_filter_top_k": args.fast_filter_top_k,
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

        results["datasets"].setdefault(spec.name, {"task": task, "models": {}})

        for model_name in args.models:
            model = _build_model(task, model_name)
            model.fit(x_train, y_train)

            model_results: dict[str, Any] = {}
            for setup in SETUPS:
                print(f"  Setup: {setup} ({model_name})")
                model_results[setup] = _run_setup(
                    learner=model,
                    setup=setup,
                    task=task,
                    workers=args.workers,
                    instance_chunk_size=args.instance_chunk_size,
                    x_cal=x_cal,
                    y_cal=y_cal,
                    x_test=x_test,
                    top_k=args.fast_filter_top_k,
                )

            results["datasets"][spec.name]["models"][model_name] = model_results

    print(f"Saving results to {args.results_file}...")
    with open(args.results_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
