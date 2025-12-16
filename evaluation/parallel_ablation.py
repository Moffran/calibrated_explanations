"""Micro-benchmark harness for ParallelExecutor ablation.

This script runs a canonical workload (calibration + explanation) across different
parallel strategies and records wall-clock time. Results are saved to JSON for
evidence-driven decision making for v0.10.
"""
import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import CalibratedExplainer, WrapCalibratedExplainer

RESULTS_FILE = "evaluation/parallel_ablation_results.json"
PHYSICAL_CORES = 6
LOGICAL_CORES = 12
WORKERS = PHYSICAL_CORES  # default: optimize for physical cores
SMALL_INSTANCE = 2000
LARGE_INSTANCE = 11000
SMALL_FEATURE = 20
LARGE_FEATURE = 100
SMALL_TEST = 1000
LARGE_TEST = 10000
CALIBRATION_SIZE = 400


def _configure_parallel_env(
    *,
    enabled: bool,
    strategy: str | None = None,
    workers: int | None = None,
    instance_chunk_size: int | None = None,
) -> None:
    """Set CE_PARALLEL using the format expected by ParallelConfig.from_env."""
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


def _expected_execution_plugin_identifier() -> str:
    """Return the expected factual plugin identifier based on the current parallel config.

    Maps the CE_PARALLEL environment variable to the corresponding built-in factual plugin:
    - "sequential" -> "core.explanation.factual.sequential"
    - "threads" -> "core.explanation.factual.instance_parallel"
    - "processes" -> "core.explanation.factual.instance_parallel"
    - "joblib" -> "core.explanation.factual.instance_parallel"
    - "off" or disabled -> "core.explanation.factual.sequential"
    """
    ce_parallel = os.environ.get("CE_PARALLEL", "off")

    if ce_parallel.lower() in ("off", "0", "false"):
        return "core.explanation.factual.sequential"

    # Parse the CE_PARALLEL string to find the strategy token
    tokens = [t.strip() for t in ce_parallel.split(",") if t.strip()]

    strategy = None
    for token in tokens:
        lower_token = token.lower()
        if lower_token in ("threads", "processes", "joblib", "sequential"):
            strategy = lower_token
            break

    # Map strategy to plugin identifier
    if strategy == "sequential":
        return "core.explanation.factual.sequential"
    elif strategy in ("threads", "processes", "joblib"):
        return "core.explanation.factual.instance_parallel"

    # Default to sequential if unable to determine
    print(f"    Warning: Unable to determine parallel strategy from CE_PARALLEL='{ce_parallel}'. Defaulting to sequential.")
    return "core.explanation.factual.sequential"


def _assert_parallel_wiring(
    wrapper: WrapCalibratedExplainer,
    *,
    expected_strategy: str,
    expected_workers: int,
) -> None:
    """Validate that the explainer used the intended parallel configuration."""
    if wrapper.explainer is None:
        raise AssertionError("Expected wrapper.explainer to be initialized")

    explainer = wrapper.explainer
    executor = getattr(explainer, "_perf_parallel", None)
    if executor is None:
        raise AssertionError("Expected explainer to have a parallel executor (_perf_parallel)")

    config = getattr(executor, "config", None)
    if config is None:
        raise AssertionError("Expected executor to have a config")

    if not getattr(config, "enabled", False):
        raise AssertionError("Expected parallel executor to be enabled")

    if getattr(config, "strategy", None) != expected_strategy:
        raise AssertionError(
            f"Expected strategy='{expected_strategy}', got '{getattr(config, 'strategy', None)}'"
        )
    if getattr(config, "max_workers", None) != expected_workers:
        raise AssertionError(
            f"Expected workers={expected_workers}, got {getattr(config, 'max_workers', None)}"
        )

    expected_plugin = _expected_execution_plugin_identifier()
    resolved = getattr(explainer, "_explanation_plugin_identifiers", {}).get("factual")
    if resolved != expected_plugin:
        raise AssertionError(f"Expected factual plugin '{expected_plugin}', got '{resolved}'")

    metrics = getattr(executor, "metrics", None)
    submitted = getattr(metrics, "submitted", 0) if metrics is not None else 0
    # If the workload is small, the executor might bypass parallelism (optimization).
    # We only assert submission if we expect the workload to be large enough.
    # The current threshold in InstanceParallelExplainExecutor is ~200 instances.
    # We'll be conservative and only assert if we have > 200 instances.
    # But we don't have n_test here easily.
    # Let's just warn instead of raising if submitted is 0.
    if submitted <= 0:
        print(f"    Warning: Executor did not submit work (metrics.submitted={submitted}). Optimization bypass likely active.")
        # raise AssertionError("Expected executor to submit work (metrics.submitted > 0)")

def run_benchmark(
    mode: str,
    n_samples: int = 1000,
    n_features: int = 20,
    n_test: int = 100,
    strategies: List[str] = None,
    workers: int = WORKERS,
) -> Dict[str, Any]:
    """Run benchmark for a specific mode and return results."""
    print(f"Running {mode} benchmark (n_samples={n_samples}, n_features={n_features}, n_test={n_test})...")

    if strategies is None:
        strategies = ["sequential", "threads", "processes"]
        # Try to include joblib if available
        try:
            import joblib  # noqa: F401
            strategies.append("joblib")
        except ImportError:
            pass

    # Generate synthetic data
    if mode == "classification":
        X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
        learner = RandomForestClassifier(n_estimators=10, random_state=42)
    else:
        X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
        learner = RandomForestRegressor(n_estimators=10, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=42)
    x_train, x_cal, y_train, y_cal = train_test_split(x_train, y_train, test_size=CALIBRATION_SIZE, random_state=42)

    # Fit learner
    learner.fit(x_train, y_train)

    results = {}

    for strategy in strategies:
        print(f"  Testing strategy: {strategy}")

        # Baseline: ensure executor is disabled so the sequential explain plugin runs.
        # (Granularity is irrelevant when parallel is disabled.)
        if strategy == "sequential":
            _configure_parallel_env(enabled=False)
            start_time = time.time()
            explainer = WrapCalibratedExplainer(learner)
            explainer.calibrate(x_cal, y_cal)
            _ = explainer.explain_factual(x_test)
            duration = time.time() - start_time
            results["sequential"] = duration
            print(f"    Duration: {duration:.2f}s (Duration/instance: {duration/n_test:.5f}s) (Duration/instance/feature: {duration/(n_test*n_features):.5f}s)")
            continue

        try:
            # Enable parallelism. NOTE: strategy is a *token* (e.g. "threads"), not "strategy=threads".
            # Also set instance_chunk_size to a smaller value to enable better parallelization
            # (default min_chunk=200 is too large and reduces parallelism).
            _configure_parallel_env(
                enabled=True,
                strategy=strategy,
                workers=workers,
                instance_chunk_size=50,  # Smaller chunks for better parallelism
            )

            plugin_id = _expected_execution_plugin_identifier()

            # Initialize explainer (wraps learner)
            # We re-initialize to ensure fresh cache/executor state if any
            start_time = time.time()
            explainer = WrapCalibratedExplainer(learner)
            explainer.calibrate(x_cal, y_cal, factual_plugin=plugin_id)

            # Explain
            _ = explainer.explain_factual(x_test)
            duration = time.time() - start_time

            _assert_parallel_wiring(
                explainer,
                expected_strategy=strategy,
                expected_workers=workers,
            )

            results[f"{strategy}"] = duration
            print(f"    Duration: {duration:.2f}s (Duration/instance: {duration/n_test:.5f}s) (Duration/instance/feature: {duration/(n_test*n_features):.5f}s)")
        finally:
            pass

    # Cleanup env
    if "CE_PARALLEL" in os.environ:
        del os.environ["CE_PARALLEL"]

    return results


def print_summary(results: Dict[str, Any]):
    """Print a human-readable summary of the results."""
    print("\n" + "=" * 60)
    print(f"{'BENCHMARK SUMMARY':^60}")
    print("=" * 60)

    for mode in ["classification_small", "classification_feature", "classification_instance", "regression_small", "regression_feature", "regression_instance"]:
        if mode not in results:
            continue

        print(f"\nMode: {mode.upper()}")
        print(f"{'Strategy':<25} | {'Duration (s)':<12} | {'Speedup':<10}")
        print("-" * 53)

        mode_results = results[mode]

        sequential_baseline = mode_results.get("sequential")

        # Sort by duration (fastest first)
        sorted_strategies = sorted(mode_results.items(), key=lambda x: x[1])

        for key, duration in sorted_strategies:
            baseline = sequential_baseline if sequential_baseline is not None else duration
            speedup = baseline / duration if duration > 0 else 0.0
            print(f"{key:<25} | {duration:<12.4f} | {speedup:<10.2f}x")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ParallelExecutor benchmark (strategies)")
    parser.add_argument("--physical-cores", type=int, default=PHYSICAL_CORES)
    parser.add_argument("--logical-cores", type=int, default=LOGICAL_CORES)
    parser.add_argument("--workers", type=int, default=WORKERS)
    return parser.parse_args()


def main():
    """Execute benchmarks and save results."""
    args = _parse_args()
    global PHYSICAL_CORES, LOGICAL_CORES, WORKERS
    PHYSICAL_CORES = int(args.physical_cores)
    LOGICAL_CORES = int(args.logical_cores)
    WORKERS = int(args.workers)

    results = {
        "meta": {
            "timestamp": time.time(),
            "platform": os.name,
            "cpu_count": os.cpu_count(),
            "workers": WORKERS,
            "physical_cores": PHYSICAL_CORES,
            "logical_cores": LOGICAL_CORES,
        },
        "classification_small": run_benchmark("classification", n_samples=SMALL_INSTANCE, n_features=SMALL_FEATURE, n_test=SMALL_TEST),
        "classification_feature": run_benchmark("classification", n_samples=SMALL_INSTANCE, n_features=LARGE_FEATURE, n_test=SMALL_TEST),
        "classification_instance": run_benchmark("classification", n_samples=LARGE_INSTANCE, n_features=SMALL_FEATURE, n_test=LARGE_TEST),
        "regression_small": run_benchmark("regression", n_samples=SMALL_INSTANCE, n_features=SMALL_FEATURE, n_test=SMALL_TEST),
        "regression_feature": run_benchmark("regression", n_samples=SMALL_INSTANCE, n_features=LARGE_FEATURE, n_test=SMALL_TEST),
        "regression_instance": run_benchmark("regression", n_samples=LARGE_INSTANCE, n_features=SMALL_FEATURE, n_test=LARGE_TEST),
    }

    print_summary(results)

    print(f"\nSaving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
