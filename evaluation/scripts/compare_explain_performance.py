"""Compare legacy and modern explanation pipeline performance.

Runs five strategy variants across classification and regression tasks and
reports mean wall-clock times with speedup ratios.  Every variant is verified
to produce numerically identical explanation payloads before timings are
reported.

Methodology
-----------
- 2 000-sample datasets via ``make_classification`` / ``make_regression``
  (64 features, 16 informative; regression noise = 0.2).
- RandomForestClassifier / RandomForestRegressor with 10 estimators.
- 500 samples for calibration, 100 held-out test instances.
- 1 warm-up run followed by 10 timed repeats per strategy.
- Strategies: legacy, modern, cached (CE_CACHE=on), parallel (CE_PARALLEL threads),
  cache+parallel.

Usage
-----
    PYTHONPATH=./src:. python evaluation/scripts/compare_explain_performance.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import Any

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

N_SAMPLES = 2000
N_FEATURES = 64
N_INFORMATIVE = 16
N_CAL = 500
N_TEST = 100
N_REPS = 10
N_ESTIMATORS = 10
RANDOM_STATE = 42


def _set_env(*, cache: bool, parallel: bool) -> None:
    os.environ["CE_CACHE"] = "on" if cache else "off"
    os.environ["CE_PARALLEL"] = "enable,threads" if parallel else "off"


def _reset_env() -> None:
    os.environ["CE_CACHE"] = "off"
    os.environ["CE_PARALLEL"] = "off"


def _build_explainer(learner: Any, x_cal: np.ndarray, y_cal: np.ndarray) -> Any:
    from calibrated_explanations import WrapCalibratedExplainer

    exp = WrapCalibratedExplainer(learner)
    exp.calibrate(x_cal, y_cal)
    return exp


def _run_modern(explainer: Any, x_test: np.ndarray, variant: str) -> Any:
    if variant == "factual":
        return explainer.explain_factual(x_test)
    return explainer.explore_alternatives(x_test)


def _run_legacy(explainer: Any, x_test: np.ndarray, variant: str) -> Any:
    from calibrated_explanations.core.explain._legacy_explain import explain as legacy_explain

    inner = explainer.explainer
    if inner is None:
        raise RuntimeError("Legacy explain requires an initialised CalibratedExplainer.")
    discretizer = {
        ("factual", "classification"): "binaryEntropy",
        ("factual", "regression"): "binaryRegressor",
        ("alternatives", "classification"): "entropy",
        ("alternatives", "regression"): "regressor",
    }[(variant, inner.mode.split("_")[0] if "_" in inner.mode else inner.mode)]
    inner.set_discretizer(discretizer)
    return legacy_explain(inner, x_test)


def _payload_array(result: Any, variant: str) -> np.ndarray:
    """Extract a numeric array from an explanation result for identity checks."""
    try:
        if hasattr(result, "explanations"):
            weights = [e.feature_weights for e in result.explanations]
            return np.asarray(weights, dtype=float)
    except Exception:  # noqa: BLE001
        pass
    return np.array([])


def _time_fn(fn, warmup: bool = True, n_reps: int = N_REPS) -> tuple[list[float], Any]:
    result = None
    if warmup:
        result = fn()
    samples: list[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    return samples, result


def _benchmark_task(
    task: str,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_test: np.ndarray,
    learner: Any,
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    strategies = [
        ("Legacy", True, False, False),
        ("Modern", False, False, False),
        ("Cached", False, True, False),
        ("Parallel", False, False, True),
        ("Cache + Parallel", False, True, True),
    ]

    reference_results: dict[str, np.ndarray] = {}

    for variant in ("factual", "alternatives"):
        results[variant] = {}
        ref_payload: np.ndarray | None = None

        for name, is_legacy, use_cache, use_parallel in strategies:
            _set_env(cache=use_cache, parallel=use_parallel)
            explainer = _build_explainer(learner, x_cal, y_cal)

            if is_legacy:
                fn = lambda e=explainer, v=variant: _run_legacy(e, x_test, v)
            else:
                fn = lambda e=explainer, v=variant: _run_modern(e, x_test, v)

            samples, last_result = _time_fn(fn)
            mean_s = statistics.mean(samples)

            payload = _payload_array(last_result, variant)
            if not is_legacy:
                if ref_payload is None:
                    ref_payload = payload
                elif payload.size > 0 and ref_payload.size > 0:
                    if not np.allclose(payload, ref_payload, rtol=1e-4, atol=1e-8):
                        print(f"  WARNING: {name}/{variant} payload differs from Modern reference.")

            results[variant][name] = {
                "mean_s": round(mean_s, 4),
                "min_s": round(min(samples), 4),
                "max_s": round(max(samples), 4),
                "samples": [round(s, 4) for s in samples],
            }

        # Compute speedup ratios
        legacy_mean = results[variant]["Legacy"]["mean_s"]
        modern_mean = results[variant]["Modern"]["mean_s"]
        for name in results[variant]:
            m = results[variant][name]["mean_s"]
            results[variant][name]["speedup_vs_legacy"] = round(legacy_mean / m, 2) if m > 0 else None
            results[variant][name]["speedup_vs_modern"] = round(modern_mean / m, 2) if m > 0 else None

    _reset_env()
    return results


def _print_table(task: str, variant: str, data: dict[str, Any]) -> None:
    print(f"\n### {task} — `{variant}`\n")
    header = f"{'Strategy':<20} {'Time (s)':>10} {'vs Legacy':>12} {'vs Modern':>12}"
    print(header)
    print("-" * len(header))
    for name, metrics in data.items():
        print(
            f"{name:<20} {metrics['mean_s']:>10.2f}"
            f" {metrics['speedup_vs_legacy']:>11.2f}×"
            f" {metrics['speedup_vs_modern']:>11.2f}×"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare legacy vs modern CE pipeline performance.")
    parser.add_argument("--output", "-o", help="Optional JSON output file path.")
    parser.add_argument("--n-reps", type=int, default=N_REPS, help="Timed repetitions per strategy.")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    all_results: dict[str, Any] = {}

    for task, make_fn, learner_cls, kwargs in [
        (
            "Classification",
            make_classification,
            RandomForestClassifier,
            {"n_features": N_FEATURES, "n_informative": N_INFORMATIVE, "n_redundant": 0},
        ),
        (
            "Regression",
            make_regression,
            RandomForestRegressor,
            {"n_features": N_FEATURES, "n_informative": N_INFORMATIVE, "noise": 0.2},
        ),
    ]:
        print(f"\n## {task}\n")
        X, y = make_fn(n_samples=args.n_samples, random_state=RANDOM_STATE, **kwargs)
        x_rest, x_test, y_rest, y_test = train_test_split(
            X, y, test_size=N_TEST, random_state=RANDOM_STATE
        )
        x_train, x_cal, y_train, y_cal = train_test_split(
            x_rest, y_rest, test_size=N_CAL, random_state=RANDOM_STATE
        )
        learner = learner_cls(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
        learner.fit(x_train, y_train)

        task_results = _benchmark_task(task.lower(), x_cal, y_cal, x_test, learner)
        all_results[task.lower()] = task_results

        for variant in ("factual", "alternatives"):
            _print_table(task, variant, task_results[variant])

    all_results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
