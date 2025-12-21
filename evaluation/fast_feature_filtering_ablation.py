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
    # Measure init (creation + calibration)
    init_start = time.time()
    explainer = WrapCalibratedExplainer._from_config(config)
    explainer.calibrate(x_cal, y_cal)
    init_duration = time.time() - init_start
    # Measure explanation separately
    explain_start = time.time()
    _ = explainer.explain_factual(x_test)
    explain_duration = time.time() - explain_start
    results["without_filtering"] = {
        "init": init_duration,
        "explain": explain_duration,
        "total": init_duration + explain_duration,
    }
    print(f"    Init: {init_duration:.2f}s, Explain: {explain_duration:.2f}s, Total: {init_duration+explain_duration:.2f}s (Per-instance explain: {explain_duration/n_test:.5f}s) (Explain/instance/feature: {explain_duration/(n_test*n_features):.5f}s)")

    # Test: fast feature filtering enabled
    print("  Testing: fast feature filtering enabled")
    builder = ExplainerBuilder(learner)
    config = builder.task("classification" if mode == "classification" else "regression").perf_parallel(False).perf_feature_filter(True, per_instance_top_k=TOP_K).build_config()
    # Measure init (creation + calibration)
    init_start = time.time()
    explainer = WrapCalibratedExplainer._from_config(config)
    explainer.calibrate(x_cal, y_cal)
    init_duration = time.time() - init_start
    # Measure explanation separately
    explain_start = time.time()
    _ = explainer.explain_factual(x_test)
    explain_duration = time.time() - explain_start
    results["with_filtering"] = {
        "init": init_duration,
        "explain": explain_duration,
        "total": init_duration + explain_duration,
    }
    print(f"    Init: {init_duration:.2f}s, Explain: {explain_duration:.2f}s, Total: {init_duration+explain_duration:.2f}s (Per-instance explain: {explain_duration/n_test:.5f}s) (Explain/instance/feature: {explain_duration/(n_test*n_features):.5f}s)")

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
        print(f"{'Fast Feature Filtering':<25} | {'Total duration (s)':<15} | {'Init (s)':<10} | {'Explain (s)':<10} | {'Speedup':<10}")
        print("-" * 85)

        mode_results = results[mode]

        without_filtering = mode_results.get("without_filtering")
        with_filtering = mode_results.get("with_filtering")

        def _total(v):
            if v is None:
                return None
            return v.get("total") if isinstance(v, dict) else v

        base_total = _total(without_filtering) or 0.0
        base_init = without_filtering.get("init") if isinstance(without_filtering, dict) else 0.0
        base_explain = without_filtering.get("explain") if isinstance(without_filtering, dict) else 0.0

        other_total = _total(with_filtering) if with_filtering is not None else None
        other_init = with_filtering.get("init") if isinstance(with_filtering, dict) else 0.0
        other_explain = with_filtering.get("explain") if isinstance(with_filtering, dict) else 0.0

        # Disabled row (baseline)
        print(f"{'disabled':<25} | {base_total:<15.4f} | {base_init:<10.2f} | {base_explain:<10.2f} | {'1.00x':<10}")

        # Enabled row (compare if available)
        if other_total is not None and base_total > 0:
            speedup = base_total / other_total if other_total else 0.0
            print(f"{'enabled':<25} | {other_total:<15.4f} | {other_init:<10.2f} | {other_explain:<10.2f} | {speedup:<10.2f}x")
        elif other_total is not None:
            print(f"{'enabled':<25} | {other_total:<15.4f} | {other_init:<10.2f} | {other_explain:<10.2f} | {'-':<10}")


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
