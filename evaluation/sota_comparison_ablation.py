"""Micro-benchmark harness comparing sequential, fast_filtering, LIME, and TreeSHAP."""
import argparse
import json
import os
import time
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder

# Try to import SOTA methods
try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    lime = None

try:
    import shap
except ImportError:
    shap = None

RESULTS_FILE = "evaluation/sota_comparison_results.json"
FAST_FILTER_TOP_K = 10
CALIBRATION_SIZE = 200
SMALL_INSTANCE = 2000
LARGE_INSTANCE = 6000
SMALL_FEATURE = 20
LARGE_FEATURE = 100
SMALL_TEST = 100
LARGE_TEST = 500

SETUPS = ["sequential", "fast_filtering", "lime", "shap", "lime_calibrated", "shap_calibrated"]


def _configure_env() -> None:
    os.environ["CE_PARALLEL"] = "off"
    os.environ["CE_CACHE"] = "off"


def _cleanup_env() -> None:
    for key in ("CE_PARALLEL", "CE_CACHE"):
        os.environ.pop(key, None)


def _build_fast_filter_explainer(learner: Any, *, mode: str) -> WrapCalibratedExplainer:
    task = "classification" if mode == "classification" else "regression"
    builder = ExplainerBuilder(learner)
    config = (
        builder.task(task)
        .perf_parallel(False)
        .perf_feature_filter(True, per_instance_top_k=FAST_FILTER_TOP_K)
        .build_config()
    )
    return WrapCalibratedExplainer.from_config(config)


def _run_ce_setup(
    *,
    learner: Any,
    setup: str,
    mode: str,
    x_cal: Any,
    y_cal: Any,
    x_test: Any,
) -> Dict[str, float]:
    _configure_env()
    try:
        init_start = time.time()
        if setup == "fast_filtering":
            explainer = _build_fast_filter_explainer(learner, mode=mode)
        else:
            explainer = WrapCalibratedExplainer(learner)

        explainer.calibrate(x_cal, y_cal)
        init_duration = time.time() - init_start

        explain_start = time.time()
        _ = explainer.explain_factual(x_test)
        explain_duration = time.time() - explain_start

        total = init_duration + explain_duration
        return {"init": init_duration, "explain": explain_duration, "total": total}
    finally:
        _cleanup_env()


def _run_lime_setup(
    *,
    learner: Any,
    setup: str,
    mode: str,
    x_train: Any,
    x_test: Any,
    x_cal: Any = None,
    y_cal: Any = None,
) -> Dict[str, float]:
    if lime is None:
        return {"init": 0.0, "explain": 0.0, "total": 0.0, "error": "LIME not installed"}

    try:
        init_start = time.time()
        lime_mode = "classification" if mode == "classification" else "regression"
        explainer = LimeTabularExplainer(
            x_train,
            mode=lime_mode,
            discretize_continuous=True,
        )

        predict_fn = learner.predict_proba if mode == "classification" else learner.predict

        if setup == "lime_calibrated":
            ce_explainer = WrapCalibratedExplainer(learner)
            ce_explainer.calibrate(x_cal, y_cal)
            predict_fn = ce_explainer.predict_proba if mode == "classification" else ce_explainer.predict

        init_duration = time.time() - init_start

        explain_start = time.time()
        num_samples = 500

        for i in range(len(x_test)):
            _ = explainer.explain_instance(x_test[i], predict_fn, num_samples=num_samples)

        explain_duration = time.time() - explain_start
        total = init_duration + explain_duration
        return {"init": init_duration, "explain": explain_duration, "total": total}
    except Exception as e:
        return {"init": 0.0, "explain": 0.0, "total": 0.0, "error": str(e)}


def _run_shap_setup(
    *,
    learner: Any,
    setup: str,
    mode: str,
    x_train: Any,
    x_test: Any,
    x_cal: Any = None,
    y_cal: Any = None,
) -> Dict[str, float]:
    if shap is None:
        return {"init": 0.0, "explain": 0.0, "total": 0.0, "error": "SHAP not installed"}

    try:
        init_start = time.time()
        if setup == "shap":
            # Use TreeExplainer for TreeSHAP on uncalibrated model
            explainer = shap.TreeExplainer(learner)
        else:
            # setup == "shap_calibrated"
            ce_explainer = WrapCalibratedExplainer(learner)
            ce_explainer.calibrate(x_cal, y_cal)

            if mode == "classification":
                predict_fn = lambda x: ce_explainer.predict_proba(x)[:, 1]
            else:
                predict_fn = ce_explainer.predict

            # Use KernelExplainer for calibrated model (black-box)
            background = shap.sample(x_train, 50)
            explainer = shap.KernelExplainer(predict_fn, background)

        init_duration = time.time() - init_start

        explain_start = time.time()
        if setup == "shap":
            _ = explainer.shap_values(x_test)
        else:
            _ = explainer.shap_values(x_test, nsamples="auto")
        explain_duration = time.time() - explain_start

        total = init_duration + explain_duration
        return {"init": init_duration, "explain": explain_duration, "total": total}
    except Exception as e:
        return {"init": 0.0, "explain": 0.0, "total": 0.0, "error": str(e)}


def _prepare_data(mode: str, *, n_samples: int, n_features: int, n_test: int):
    if mode == "classification":
        X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
        learner = RandomForestClassifier(n_estimators=10, random_state=42)
    else:
        X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
        learner = RandomForestRegressor(n_estimators=10, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=42)

    # Ensure calibration size is not larger than training set
    cal_size = min(CALIBRATION_SIZE, int(len(x_train) * 0.33))

    x_train_prop, x_cal, y_train_prop, y_cal = train_test_split(
        x_train, y_train, test_size=cal_size, random_state=42
    )

    learner.fit(x_train_prop, y_train_prop)
    return learner, x_train_prop, x_cal, y_cal, x_test


def run_benchmark(
    mode: str,
    *,
    n_samples: int,
    n_features: int,
    n_test: int,
) -> Dict[str, Any]:
    print(f"Running {mode} benchmark (n_samples={n_samples}, n_features={n_features}, n_test={n_test})...")
    learner, x_train, x_cal, y_cal, x_test = _prepare_data(mode, n_samples=n_samples, n_features=n_features, n_test=n_test)

    results: Dict[str, Dict[str, float]] = {}
    for setup in SETUPS:
        print(f"  Setup: {setup}")
        if setup in ["sequential", "fast_filtering"]:
            results[setup] = _run_ce_setup(
                learner=learner,
                setup=setup,
                mode=mode,
                x_cal=x_cal,
                y_cal=y_cal,
                x_test=x_test,
            )
        elif setup in ["lime", "lime_calibrated"]:
            results[setup] = _run_lime_setup(
                learner=learner,
                setup=setup,
                mode=mode,
                x_train=x_train,
                x_test=x_test,
                x_cal=x_cal,
                y_cal=y_cal,
            )
        elif setup in ["shap", "shap_calibrated"]:
            results[setup] = _run_shap_setup(
                learner=learner,
                setup=setup,
                mode=mode,
                x_train=x_train,
                x_test=x_test,
                x_cal=x_cal,
                y_cal=y_cal,
            )

        res = results[setup]
        if "error" in res:
            print(f"    {setup}: ERROR: {res['error']}")
        else:
            print(f"    {setup}: init={res['init']:.2f}s explain={res['explain']:.2f}s total={res['total']:.2f}s")

    return results


def print_summary(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(f"{'SOTA COMPARISON BENCHMARK SUMMARY':^100}")
    print("=" * 100)

    modes = [
        "classification_small",
        "classification_feature",
        "classification_instance",
        "regression_small",
        "regression_feature",
        "regression_instance",
    ]
    for mode in modes:
        if mode not in results:
            continue
        print(f"\nMode: {mode.upper()}")
        print(
            f"{'Setup':<20} | {'Total (s)':<12} | {'Init (s)':<10} | {'Explain (s)':<12} | {'Speedup':<10} | {'Status':<10}"
        )
        print("-" * 100)

        mode_results = results[mode]
        baseline = mode_results.get("sequential", {}).get("total", 0.0)

        for setup in SETUPS:
            entry = mode_results.get(setup, {})
            total = entry.get("total", 0.0)
            init = entry.get("init", 0.0)
            explain = entry.get("explain", 0.0)
            status = "OK" if "error" not in entry else "FAILED"

            speedup = baseline / total if baseline and total else 0.0
            speedup_str = f"{speedup:.2f}x" if baseline and total else "-"

            print(f"{setup:<20} | {total:<12.4f} | {init:<10.2f} | {explain:<12.2f} | {speedup_str:<10} | {status:<10}")


def main() -> None:
    results: Dict[str, Any] = {
        "meta": {
            "timestamp": time.time(),
            "platform": os.name,
            "cpu_count": os.cpu_count(),
        },
        "classification_small": run_benchmark(
            "classification",
            n_samples=SMALL_INSTANCE,
            n_features=SMALL_FEATURE,
            n_test=SMALL_TEST,
        ),
        "classification_feature": run_benchmark(
            "classification",
            n_samples=SMALL_INSTANCE,
            n_features=LARGE_FEATURE,
            n_test=SMALL_TEST,
        ),
        "classification_instance": run_benchmark(
            "classification",
            n_samples=LARGE_INSTANCE,
            n_features=SMALL_FEATURE,
            n_test=LARGE_TEST,
        ),
        "regression_small": run_benchmark(
            "regression",
            n_samples=SMALL_INSTANCE,
            n_features=SMALL_FEATURE,
            n_test=SMALL_TEST,
        ),
        "regression_feature": run_benchmark(
            "regression",
            n_samples=SMALL_INSTANCE,
            n_features=LARGE_FEATURE,
            n_test=SMALL_TEST,
        ),
        "regression_instance": run_benchmark(
            "regression",
            n_samples=LARGE_INSTANCE,
            n_features=SMALL_FEATURE,
            n_test=LARGE_TEST,
        ),
    }

    print_summary(results)

    print(f"\nSaving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
