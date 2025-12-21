"""Micro-benchmark harness comparing legacy, sequential, parallel, caching, and feature-filtering setups."""
import argparse
import json
import os
import time
from typing import Any, Dict, List

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder

RESULTS_FILE = "evaluation/setup_ablation_results.json"
WORKERS = 6
INSTANCE_CHUNK_SIZE = 50
FAST_FILTER_TOP_K = 5
CALIBRATION_SIZE = 200
SMALL_INSTANCE = 2000
LARGE_INSTANCE = 6000
SMALL_FEATURE = 20
LARGE_FEATURE = 100
SMALL_TEST = 1000
LARGE_TEST = 5000
SETUPS = ["legacy", "sequential", "instance_parallel", "caching", "fast_filtering"]


def _configure_parallel_env(
    *,
    enabled: bool,
    strategy: str | None = None,
    workers: int | None = None,
    instance_chunk_size: int | None = None,
) -> None:
    """Set the CE_PARALLEL environment variable for the desired parallel strategy."""
    if not enabled:
        os.environ["CE_PARALLEL"] = "off"
        return

    tokens: List[str] = ["enable"]
    if strategy is not None:
        tokens.append(strategy)
    if workers is not None:
        tokens.append(f"workers={workers}")
    if instance_chunk_size is not None:
        tokens.append(f"instance_chunk_size={instance_chunk_size}")
    os.environ["CE_PARALLEL"] = ",".join(tokens)


def _configure_caching_env(*, enabled: bool) -> None:
    """Toggle CE_CACHE for caching ablations."""
    os.environ["CE_CACHE"] = "on" if enabled else "off"


def _cleanup_env() -> None:
    for key in ("CE_PARALLEL", "CE_CACHE"):
        os.environ.pop(key, None)


def _expected_execution_plugin_identifier() -> str:
    """Map the current CE_PARALLEL configuration to the factual plugin identifier."""
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


def _build_fast_filter_explainer(learner: Any, *, mode: str) -> WrapCalibratedExplainer:
    task = "classification" if mode == "classification" else "regression"
    builder = ExplainerBuilder(learner)
    config = (
        builder.task(task)
        .perf_parallel(False)
        .perf_feature_filter(True, per_instance_top_k=FAST_FILTER_TOP_K)
        .build_config()
    )
    return WrapCalibratedExplainer._from_config(config)


def _instantiate_explainer(learner: Any, *, setup: str, mode: str) -> WrapCalibratedExplainer:
    if setup == "fast_filtering":
        return _build_fast_filter_explainer(learner, mode=mode)
    return WrapCalibratedExplainer(learner)


def _run_setup(
    *,
    learner: Any,
    setup: str,
    mode: str,
    workers: int,
    instance_chunk_size: int,
    x_cal: Any,
    y_cal: Any,
    x_test: Any,
) -> Dict[str, float]:
    """Execute one of the requested setups and return timing statistics."""
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
        init_start = time.time()
        explainer = _instantiate_explainer(learner, setup=setup, mode=mode)
        explainer.calibrate(x_cal, y_cal, factual_plugin=plugin_id)
        init_duration = time.time() - init_start

        explain_kwargs: Dict[str, Any] = {}
        if setup == "legacy":
            explain_kwargs["_use_plugin"] = False

        explain_start = time.time()
        _ = explainer.explain_factual(x_test, **explain_kwargs)
        explain_duration = time.time() - explain_start

        total = init_duration + explain_duration
        print(
            f"    {setup}: init={init_duration:.2f}s explain={explain_duration:.2f}s total={total:.2f}s"
        )

        return {"init": init_duration, "explain": explain_duration, "total": total}
    finally:
        _cleanup_env()


def _prepare_data(mode: str, *, n_samples: int, n_features: int, n_test: int):
    if mode == "classification":
        X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
        learner = RandomForestClassifier(n_estimators=10, random_state=42)
    else:
        X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
        learner = RandomForestRegressor(n_estimators=10, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=42)
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_train, y_train, test_size=CALIBRATION_SIZE, random_state=42
    )

    learner.fit(x_train, y_train)
    return learner, x_cal, y_cal, x_test


def run_benchmark(
    mode: str,
    *,
    n_samples: int,
    n_features: int,
    n_test: int,
    workers: int,
    instance_chunk_size: int,
) -> Dict[str, Any]:
    print(f"Running {mode} benchmark (n_samples={n_samples}, n_features={n_features}, n_test={n_test})...")
    learner, x_cal, y_cal, x_test = _prepare_data(mode, n_samples=n_samples, n_features=n_features, n_test=n_test)

    results: Dict[str, Dict[str, float]] = {}
    for setup in SETUPS:
        print(f"  Setup: {setup}")
        results[setup] = _run_setup(
            learner=learner,
            setup=setup,
            mode=mode,
            workers=workers,
            instance_chunk_size=instance_chunk_size,
            x_cal=x_cal,
            y_cal=y_cal,
            x_test=x_test,
        )

    return results


def print_summary(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"{'BENCHMARK SUMMARY':^60}")
    print("=" * 60)

    modes = [
        "classification_small",
        "classification_feature",
        "classification_instance",
        "regression_small",
        "regression_feature",
        "regression_instance",
    ]
    for mode in modes:
        if mode not in results:
            continue
        print(f"\nMode: {mode.upper()}")
        print(
            f"{'Setup':<25} | {'Total duration (s)':<15} | {'Init (s)':<10} | {'Explain (s)':<10} | {'Speedup':<10}"
        )
        print("-" * 85)

        mode_results = results[mode]
        baseline = mode_results.get("sequential", {}).get("total", 0.0)

        sorted_setups = sorted(
            ((key, val.get("total", 0.0)) for key, val in mode_results.items()),
            key=lambda item: item[1] if item[1] is not None else float("inf"),
        )

        for key, total in sorted_setups:
            entry = mode_results.get(key, {})
            init = entry.get("init", 0.0) if isinstance(entry, dict) else 0.0
            explain = entry.get("explain", 0.0) if isinstance(entry, dict) else 0.0
            speedup = baseline / total if baseline and total else 0.0
            speedup_str = f"{speedup:.2f}x" if baseline and total else "-"
            print(f"{key:<25} | {total:<15.4f} | {init:<10.2f} | {explain:<10.2f} | {speedup_str:<10}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup ablation benchmark")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--chunk-size", type=int, default=INSTANCE_CHUNK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    results: Dict[str, Any] = {
        "meta": {
            "timestamp": time.time(),
            "platform": os.name,
            "cpu_count": os.cpu_count(),
            "workers": args.workers,
            "instance_chunk_size": args.chunk_size,
        },
        "classification_small": run_benchmark(
            "classification",
            n_samples=SMALL_INSTANCE,
            n_features=SMALL_FEATURE,
            n_test=SMALL_TEST,
            workers=args.workers,
            instance_chunk_size=args.chunk_size,
        ),
        "classification_feature": run_benchmark(
            "classification",
            n_samples=SMALL_INSTANCE,
            n_features=LARGE_FEATURE,
            n_test=SMALL_TEST,
            workers=args.workers,
            instance_chunk_size=args.chunk_size,
        ),
        "classification_instance": run_benchmark(
            "classification",
            n_samples=LARGE_INSTANCE,
            n_features=SMALL_FEATURE,
            n_test=LARGE_TEST,
            workers=args.workers,
            instance_chunk_size=args.chunk_size,
        ),
        "regression_small": run_benchmark(
            "regression",
            n_samples=SMALL_INSTANCE,
            n_features=SMALL_FEATURE,
            n_test=SMALL_TEST,
            workers=args.workers,
            instance_chunk_size=args.chunk_size,
        ),
        "regression_feature": run_benchmark(
            "regression",
            n_samples=SMALL_INSTANCE,
            n_features=LARGE_FEATURE,
            n_test=SMALL_TEST,
            workers=args.workers,
            instance_chunk_size=args.chunk_size,
        ),
        "regression_instance": run_benchmark(
            "regression",
            n_samples=LARGE_INSTANCE,
            n_features=SMALL_FEATURE,
            n_test=LARGE_TEST,
            workers=args.workers,
            instance_chunk_size=args.chunk_size,
        ),
    }

    print_summary(results)

    print(f"\nSaving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
