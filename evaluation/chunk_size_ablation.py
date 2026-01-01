"""Ablation for instance chunk sizes when using parallel execution.

Runs one classification and one regression experiment using 10000 test instances
and measures wall-clock time across several `instance_chunk_size` values.

Results are written to `evaluation/chunk_size_ablation_results.json`.
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

from calibrated_explanations import WrapCalibratedExplainer

RESULTS_FILE = "evaluation/chunk_size_ablation_results.json"
WORKERS = 6
CALIBRATION_SIZE = 200
CHUNK_SIZES = [10, 50, 200, 500, 1000, 2000]


def _configure_parallel_env(*, enabled: bool, strategy: str | None = None, workers: int | None = None, instance_chunk_size: int | None = None) -> None:
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


def _prepare_data_for_mode(mode: str, *, n_test: int = 10000):
    """Generate synthetic dataset and return learner, x_cal, y_cal, x_test.

    Keeps random_state deterministic for reproducibility.
    """
    n_samples = n_test + CALIBRATION_SIZE + 1000

    if mode == "classification":
        X, y = make_classification(n_samples=n_samples, n_features=20, random_state=42)
        learner = RandomForestClassifier(n_estimators=10, random_state=42)
    else:
        X, y = make_regression(n_samples=n_samples, n_features=20, random_state=42)
        learner = RandomForestRegressor(n_estimators=10, random_state=42)

    # Split: hold out test set of size n_test
    x_rest, x_test, y_rest, y_test = train_test_split(X, y, test_size=n_test, random_state=42)
    # From rest, carve out calibration set
    x_train, x_cal, y_train, y_cal = train_test_split(x_rest, y_rest, test_size=CALIBRATION_SIZE, random_state=42)

    learner.fit(x_train, y_train)
    return learner, x_cal, y_cal, x_test


def _run_for_strategy(learner, x_cal, y_cal, x_test, strategy: str, chunk_sizes: List[int], *, workers: int) -> Dict[int, float]:
    """Run experiments for a single strategy (expects prepared data) and return mapping chunk_size -> duration."""
    print(f"Running strategy={strategy} workers={workers} chunk_sizes={chunk_sizes}...")

    results: Dict[int, float] = {}

    for chunk in chunk_sizes:
        print(f"  chunk_size={chunk} -> ", end="", flush=True)

        # Enable/disable parallelism appropriately
        if strategy == "sequential":
            _configure_parallel_env(enabled=False)
        else:
            _configure_parallel_env(enabled=True, strategy=strategy, workers=workers, instance_chunk_size=chunk)

        try:
            start = time.time()
            explainer = WrapCalibratedExplainer(learner)
            explainer.calibrate(x_cal, y_cal)
            _ = explainer.explain_factual(x_test)
            duration = time.time() - start
            results[chunk] = duration
            print(f"{duration:.2f}s")
        finally:
            if "CE_PARALLEL" in os.environ:
                del os.environ["CE_PARALLEL"]

    return results


def main():
    parser = argparse.ArgumentParser(description="Chunk size ablation for parallel explain executor")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    strategies = ["threads", "processes"]
    # include joblib if available
    try:
        import joblib  # noqa: F401
        strategies.append("joblib")
    except Exception:
        pass

    results: Dict[str, Any] = {
        "meta": {
            "timestamp": time.time(),
            "platform": os.name,
            "cpu_count": os.cpu_count(),
            "workers": args.workers,
            "chunk_sizes": CHUNK_SIZES,
        }
    }

    # Run classification + regression (single experiment each)
    for mode in ("classification", "regression"):
        print(f"\n=== Mode: {mode.upper()} ===")
        learner, x_cal, y_cal, x_test = _prepare_data_for_mode(mode)

        results[mode] = {}

        # Sequential baseline (single run)
        print("Running sequential baseline...")
        _configure_parallel_env(enabled=False)
        start = time.time()
        explainer = WrapCalibratedExplainer(learner)
        explainer.calibrate(x_cal, y_cal)
        _ = explainer.explain_factual(x_test)
        duration = time.time() - start
        results[mode]["sequential"] = duration
        print(f"  sequential: {duration:.2f}s")

        # Parallel strategies (vary chunk sizes)
        for strat in strategies:
            results[mode][strat] = _run_for_strategy(learner, x_cal, y_cal, x_test, strat, CHUNK_SIZES, workers=args.workers)

    print(f"\nSaving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
