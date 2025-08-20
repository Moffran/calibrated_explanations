# pylint: disable=line-too-long, missing-function-docstring, too-many-locals, import-outside-toplevel, invalid-name, no-member, unused-import
"""Run micro benchmarks with optional repetition for trend tracking.

This script is intentionally light-weight to be CI friendly.
Use pytest-benchmark for deeper analysis later.
"""
from __future__ import annotations
import argparse
import statistics
import time
from typing import Dict, Any, Callable, List
import json

from sklearn.datasets import load_breast_cancer, load_diabetes  # noqa
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa

from calibrated_explanations import WrapCalibratedExplainer  # noqa


def time_fn(fn: Callable[[], Any], repeat: int) -> Dict[str, float]:
    samples: List[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "stdev": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "samples": samples,
    }


def bench_classification() -> Dict[str, Any]:
    data = load_breast_cancer()
    X, y = data.data, data.target
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    expl = WrapCalibratedExplainer(clf)

    results: Dict[str, Any] = {}

    results["fit"] = time_fn(lambda: expl.fit(X[:300], y[:300]), repeat=1)
    results["calibrate"] = time_fn(lambda: expl.calibrate(X[300:400], y[300:400]), repeat=1)
    results["predict"] = time_fn(lambda: expl.predict(X[400:450]), repeat=5)
    return results


def bench_regression() -> Dict[str, Any]:
    data = load_diabetes()
    X, y = data.data, data.target
    reg = RandomForestRegressor(n_estimators=50, random_state=42)
    reg.fit(X, y)
    expl = WrapCalibratedExplainer(reg)

    results: Dict[str, Any] = {}

    results["fit"] = time_fn(lambda: expl.fit(X[:250], y[:250]), repeat=1)
    results["calibrate"] = time_fn(lambda: expl.calibrate(X[250:350], y[250:350]), repeat=1)
    results["predict"] = time_fn(lambda: expl.predict(X[350:400]), repeat=5)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run micro benchmarks for calibrated_explanations")
    parser.add_argument("--output", "-o", type=str, help="Optional JSON output file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    results = {
        "classification": bench_classification(),
        "regression": bench_regression(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2 if args.pretty else None, sort_keys=True)
        print(f"Benchmarks written to {args.output}")
    else:
        print(json.dumps(results, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
