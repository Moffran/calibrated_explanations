"""Micro-benchmark harness for ParallelExecutor ablation.

This script runs a canonical workload (calibration + explanation) across different
parallel strategies and records wall-clock time. Results are saved to JSON for
evidence-driven decision making for v0.10.
"""
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


def run_benchmark(
    mode: str,
    n_samples: int = 1000,
    n_features: int = 20,
    n_test: int = 100,
    strategies: List[str] = None,
) -> Dict[str, Any]:
    """Run benchmark for a specific mode and return results."""
    print(f"Running {mode} benchmark (n_samples={n_samples}, n_features={n_features})...")
    
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

    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test = X_cal[:n_test]

    # Fit learner
    learner.fit(X_train, y_train)

    results = {}

    for strategy in strategies:
        print(f"  Testing strategy: {strategy}")
        # Set env var to force strategy
        os.environ["CE_PARALLEL"] = f"enable,strategy={strategy}"
        
        # Initialize explainer (wraps learner)
        # We re-initialize to ensure fresh cache/executor state if any
        start_time = time.time()
        explainer = WrapCalibratedExplainer(learner)
        explainer.calibrate(X_cal, y_cal)
        
        # Explain
        _ = explainer.explain_factual(X_test)
        
        duration = time.time() - start_time
        results[strategy] = duration
        print(f"    Duration: {duration:.4f}s")

    # Cleanup env
    if "CE_PARALLEL" in os.environ:
        del os.environ["CE_PARALLEL"]

    return results


def main():
    """Execute benchmarks and save results."""
    results = {
        "meta": {
            "timestamp": time.time(),
            "platform": os.name,
            "cpu_count": os.cpu_count(),
        },
        "classification": run_benchmark("classification"),
        "regression": run_benchmark("regression"),
    }

    print(f"Saving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
