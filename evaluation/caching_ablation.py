"""Micro-benchmark harness for caching ablation.

This script runs a canonical workload (calibration + explanation) with and without
caching enabled and records wall-clock time. Results are saved to JSON for
evidence-driven decision making.
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

RESULTS_FILE = "evaluation/caching_ablation_results.json"
SMALL_INSTANCE = 6500
LARGE_INSTANCE = 11000
SMALL_FEATURE = 20
LARGE_FEATURE = 100
SMALL_TEST = 500
LARGE_TEST = 5000
CALIBRATION_SIZE = 5000


def _configure_caching_env(*, enabled: bool) -> None:
    """Set CE_CACHE using the format expected by CacheConfig.from_env."""
    if not enabled:
        os.environ["CE_CACHE"] = "off"
    else:
        os.environ["CE_CACHE"] = "on"


def run_benchmark(
    mode: str,
    n_samples: int = 1000,
    n_features: int = 20,
    n_test: int = 100,
) -> Dict[str, Any]:
    """Run benchmark for a specific mode and return results."""
    print(f"Running {mode} benchmark (n_samples={n_samples}, n_features={n_features}, n_test={n_test})...")

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

    # Baseline: caching disabled
    print("  Testing: caching disabled")
    _configure_caching_env(enabled=False)
    start_time = time.time()
    explainer = WrapCalibratedExplainer(learner)
    explainer.calibrate(x_cal, y_cal)
    _ = explainer.explain_factual(x_test)
    duration_no_cache = time.time() - start_time
    results["no_cache"] = duration_no_cache
    print(f"    Duration: {duration_no_cache:.2f}s (Duration/instance: {duration_no_cache/n_test:.5f}s) (Duration/instance/feature: {duration_no_cache/(n_test*n_features):.5f}s)")

    # Test: caching enabled
    print("  Testing: caching enabled")
    _configure_caching_env(enabled=True)
    start_time = time.time()
    explainer = WrapCalibratedExplainer(learner)
    explainer.calibrate(x_cal, y_cal)
    _ = explainer.explain_factual(x_test)
    duration_with_cache = time.time() - start_time
    results["with_cache"] = duration_with_cache
    print(f"    Duration: {duration_with_cache:.2f}s (Duration/instance: {duration_with_cache/n_test:.5f}s) (Duration/instance/feature: {duration_with_cache/(n_test*n_features):.5f}s)")

    # Cleanup env
    if "CE_CACHE" in os.environ:
        del os.environ["CE_CACHE"]

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
        print(f"{'Caching':<25} | {'Duration (s)':<12} | {'Speedup':<10}")
        print("-" * 53)

        mode_results = results[mode]

        no_cache_duration = mode_results.get("no_cache")
        with_cache_duration = mode_results.get("with_cache")

        print(f"{'disabled':<25} | {no_cache_duration:<12.4f} | {'1.00x':<10}")
        if with_cache_duration is not None and no_cache_duration is not None and no_cache_duration > 0:
            speedup = no_cache_duration / with_cache_duration
            print(f"{'enabled':<25} | {with_cache_duration:<12.4f} | {speedup:<10.2f}x")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caching ablation benchmark")
    return parser.parse_args()


def main():
    """Execute benchmarks and save results."""
    args = _parse_args()

    results = {
        "meta": {
            "timestamp": time.time(),
            "platform": os.name,
            "cpu_count": os.cpu_count(),
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
