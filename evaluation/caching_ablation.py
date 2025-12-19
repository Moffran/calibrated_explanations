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
    # Measure init (creation + calibration)
    init_start = time.time()
    explainer = WrapCalibratedExplainer(learner)
    explainer.calibrate(x_cal, y_cal)
    init_duration = time.time() - init_start
    # Measure explanation separately
    explain_start = time.time()
    _ = explainer.explain_factual(x_test)
    explain_duration = time.time() - explain_start
    duration_no_cache = init_duration + explain_duration
    results["no_cache"] = {
        "init": init_duration,
        "explain": explain_duration,
        "total": duration_no_cache,
    }
    print(f"    Init: {init_duration:.2f}s, Explain: {explain_duration:.2f}s, Total: {duration_no_cache:.2f}s (Per-instance explain: {explain_duration/n_test:.5f}s) (Explain/instance/feature: {explain_duration/(n_test*n_features):.5f}s)")

    # Test: caching enabled
    print("  Testing: caching enabled")
    _configure_caching_env(enabled=True)
    # Measure init (creation + calibration)
    init_start = time.time()
    explainer = WrapCalibratedExplainer(learner)
    explainer.calibrate(x_cal, y_cal)
    init_duration = time.time() - init_start
    # Measure explanation separately
    explain_start = time.time()
    _ = explainer.explain_factual(x_test)
    explain_duration = time.time() - explain_start
    duration_with_cache = init_duration + explain_duration
    results["with_cache"] = {
        "init": init_duration,
        "explain": explain_duration,
        "total": duration_with_cache,
    }
    print(f"    Init: {init_duration:.2f}s, Explain: {explain_duration:.2f}s, Total: {duration_with_cache:.2f}s (Per-instance explain: {explain_duration/n_test:.5f}s) (Explain/instance/feature: {explain_duration/(n_test*n_features):.5f}s)")

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
        print(f"{'Caching':<25} | {'Total duration (s)':<15} | {'Init (s)':<10} | {'Explain (s)':<10} | {'Speedup':<10}")
        print("-" * 85)

        mode_results = results[mode]

        no_cache = mode_results.get("no_cache")
        with_cache = mode_results.get("with_cache")

        def _total(v):
            if v is None:
                return None
            return v.get("total") if isinstance(v, dict) else v

        base_total = _total(no_cache) or 0.0
        base_init = no_cache.get("init") if isinstance(no_cache, dict) else 0.0
        base_explain = no_cache.get("explain") if isinstance(no_cache, dict) else 0.0

        other_total = _total(with_cache) if with_cache is not None else None
        other_init = with_cache.get("init") if isinstance(with_cache, dict) else 0.0
        other_explain = with_cache.get("explain") if isinstance(with_cache, dict) else 0.0

        print(f"{'disabled':<25} | {base_total:<15.4f} | {base_init:<10.2f} | {base_explain:<10.2f} | {'1.00x':<10}")
        if other_total is not None and base_total > 0:
            speedup = base_total / other_total if other_total else 0.0
            print(f"{'enabled':<25} | {other_total:<15.4f} | {other_init:<10.2f} | {other_explain:<10.2f} | {speedup:<10.2f}x")
        elif other_total is not None:
            print(f"{'enabled':<25} | {other_total:<15.4f} | {other_init:<10.2f} | {other_explain:<10.2f} | {'-':<10}")


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
