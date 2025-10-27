"""Micro benchmarks for perf scaffolding (manual smoke test).

Measures:
- Import time for calibrated_explanations
- Sequential comprehension vs ParallelExecutor.map on a trivial CPU-bound function
- Factual and alternative explanation timings for simple classifiers/regressors

Usage (optional): run as a script to print JSON metrics. CI integration can
parse this later and compare to thresholds in benchmarks/perf_thresholds.json.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Literal

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ExplainBackend = Literal["current", "legacy"]


def measure_import_time() -> float:
    t0 = time.perf_counter()
    import calibrated_explanations  # noqa: F401

    return time.perf_counter() - t0


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def measure_map_throughput() -> dict[str, float]:
    from calibrated_explanations.perf import ParallelConfig, ParallelExecutor

    items = [30] * 2000  # small, deterministic CPU work

    t0 = time.perf_counter()
    [fib(item) for item in items]
    seq = time.perf_counter() - t0

    executor = ParallelExecutor(ParallelConfig(enabled=True, strategy="threads"))
    t0 = time.perf_counter()
    executor.map(fib, items)
    par = time.perf_counter() - t0

    return {"sequential_s": seq, "parallel_s": par}


def _prepare_classification() -> tuple[Any, np.ndarray]:
    from calibrated_explanations import WrapCalibratedExplainer

    data = load_breast_cancer()
    X, y = data.data, data.target
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)

    wrapper = WrapCalibratedExplainer(clf)
    wrapper.fit(X[:300], y[:300])
    wrapper.calibrate(X[300:400], y[300:400])
    return wrapper, np.asarray(X[400:410])


def _prepare_regression() -> tuple[Any, np.ndarray]:
    from calibrated_explanations import WrapCalibratedExplainer

    data = load_diabetes()
    X, y = data.data, data.target
    reg = RandomForestRegressor(n_estimators=50, random_state=42)
    reg.fit(X, y)

    wrapper = WrapCalibratedExplainer(reg)
    wrapper.fit(X[:250], y[:250])
    wrapper.calibrate(X[250:350], y[250:350])
    return wrapper, np.asarray(X[350:360])


def _time_explain(
    backend: ExplainBackend,
    variant: Literal["factual", "alternatives"],
    mode: Literal["classification", "regression"],
) -> float:
    if mode == "classification":
        wrapper, sample = _prepare_classification()
    else:
        wrapper, sample = _prepare_regression()

    if backend == "legacy":
        from calibrated_explanations.core._legacy_explain import (
            explain as legacy_explain,
        )

        assert wrapper.explainer is not None
        explainer = wrapper.explainer
        if variant == "factual":
            discretizer = "binaryRegressor" if "regression" in explainer.mode else "binaryEntropy"
        else:
            discretizer = "regressor" if "regression" in explainer.mode else "entropy"
        explainer.set_discretizer(discretizer)
        start = time.perf_counter()
        legacy_explain(explainer, sample)
        return time.perf_counter() - start

    start = time.perf_counter()
    if variant == "factual":
        wrapper.explain_factual(sample)
    else:
        wrapper.explore_alternatives(sample)
    return time.perf_counter() - start


def collect_metrics(explain_backend: ExplainBackend = "current") -> Dict[str, Any]:
    """Collect micro benchmark metrics."""
    return {
        "import_time_seconds": measure_import_time(),
        "map": measure_map_throughput(),
        "classification": {
            "explain_factual_time_s": _time_explain(explain_backend, "factual", "classification"),
            "explore_alternatives_time_s": _time_explain(
                explain_backend, "alternatives", "classification"
            ),
        },
        "regression": {
            "explain_factual_time_s": _time_explain(explain_backend, "factual", "regression"),
            "explore_alternatives_time_s": _time_explain(
                explain_backend, "alternatives", "regression"
            ),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect micro benchmark metrics.")
    parser.add_argument(
        "--explain-backend",
        choices=("current", "legacy"),
        default="current",
        help="Explanation path to benchmark (current=optimized, legacy=historical).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()

    metrics = collect_metrics(args.explain_backend)
    print(json.dumps(metrics, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
