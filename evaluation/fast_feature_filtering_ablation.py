"""Micro-benchmark harness for fast feature filtering ablation.

This script runs a canonical workload (calibration + explanation) with and without
fast feature filtering enabled and records wall-clock time. Results are saved to JSON for
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
from calibrated_explanations.api.config import ExplainerBuilder

RESULTS_FILE = "evaluation/fast_feature_filtering_ablation_results.json"
SMALL_INSTANCE = 2000
SMALL_FEATURE = 10
MEDIUM_FEATURE = 100
LARGE_FEATURE = 1000
SMALL_TEST = 100
CALIBRATION_SIZE = 100
TOP_K = 5


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
    builder = ExplainerBuilder(learner)
    config = builder.task("classification" if mode == "classification" else "regression").perf_parallel(False).perf_feature_filter(False).build_config()
    start_time = time.time()
    explainer = WrapCalibratedExplainer._from_config(config)
    explainer.calibrate(x_cal, y_cal)
    _ = explainer.explain_factual(x_test[:3])  # Warm-up run to avoid cold-start effects
    
    results = {}

    # Baseline: fast feature filtering disabled
    print("  Testing: fast feature filtering disabled")
    builder = ExplainerBuilder(learner)
    config = builder.task("classification" if mode == "classification" else "regression").perf_parallel(False).perf_feature_filter(False).build_config()
    start_time = time.time()
    explainer = WrapCalibratedExplainer._from_config(config)
    explainer.calibrate(x_cal, y_cal)
    _ = explainer.explain_factual(x_test)
    duration_without_filtering = time.time() - start_time
    results["without_filtering"] = duration_without_filtering
    print(f"    Duration: {duration_without_filtering:.2f}s (Duration/instance: {duration_without_filtering/n_test:.5f}s) (Duration/instance/feature: {duration_without_filtering/(n_test*n_features):.5f}s)")

    # Test: fast feature filtering enabled
    print("  Testing: fast feature filtering enabled")
    builder = ExplainerBuilder(learner)
    config = builder.task("classification" if mode == "classification" else "regression").perf_parallel(False).perf_feature_filter(True, per_instance_top_k=TOP_K).build_config()
    start_time = time.time()
    explainer = WrapCalibratedExplainer._from_config(config)
    explainer.calibrate(x_cal, y_cal)
    _ = explainer.explain_factual(x_test)
    duration_with_filtering = time.time() - start_time
    results["with_filtering"] = duration_with_filtering
    print(f"    Duration: {duration_with_filtering:.2f}s (Duration/instance: {duration_with_filtering/n_test:.5f}s) (Duration/instance/feature: {duration_with_filtering/(n_test*n_features):.5f}s)")

    return results


def print_summary(results: Dict[str, Any]):
    """Print a human-readable summary of the results."""
    print("\n" + "=" * 60)
    print(f"{'BENCHMARK SUMMARY':^60}")
    print("=" * 60)

    for mode in ["classification_small", "classification_medium", "classification_large", "regression_small", "regression_medium", "regression_large"]:
        if mode not in results:
            continue

        print(f"\nMode: {mode.upper()}")
        print(f"{'Fast Feature Filtering':<25} | {'Duration (s)':<12} | {'Speedup':<10}")
        print("-" * 53)

        mode_results = results[mode]

        without_filtering_duration = mode_results.get("without_filtering")
        with_filtering_duration = mode_results.get("with_filtering")

        print(f"{'disabled':<25} | {without_filtering_duration:<12.4f} | {'1.00x':<10}")
        if with_filtering_duration is not None and without_filtering_duration is not None and without_filtering_duration > 0:
            speedup = without_filtering_duration / with_filtering_duration
            print(f"{'enabled':<25} | {with_filtering_duration:<12.4f} | {speedup:<10.2f}x")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast feature filtering ablation benchmark")
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
        "classification_medium": run_benchmark("classification", n_samples=SMALL_INSTANCE, n_features=MEDIUM_FEATURE, n_test=SMALL_TEST),
        "classification_large": run_benchmark("classification", n_samples=SMALL_INSTANCE, n_features=LARGE_FEATURE, n_test=SMALL_TEST),
        "regression_small": run_benchmark("regression", n_samples=SMALL_INSTANCE, n_features=SMALL_FEATURE, n_test=SMALL_TEST),
        "regression_medium": run_benchmark("regression", n_samples=SMALL_INSTANCE, n_features=MEDIUM_FEATURE, n_test=SMALL_TEST),
        "regression_large": run_benchmark("regression", n_samples=SMALL_INSTANCE, n_features=LARGE_FEATURE, n_test=SMALL_TEST),
    }

    print_summary(results)

    print(f"\nSaving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
