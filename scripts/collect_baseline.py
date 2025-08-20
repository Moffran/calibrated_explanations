# pylint: disable=line-too-long, missing-function-docstring, too-many-locals, import-outside-toplevel, invalid-name, no-member, unused-import
"""Collect baseline performance, memory, and API surface metrics.

Run: python scripts/collect_baseline.py --output benchmarks/baseline_$(date +%Y%m%d).json
Windows PowerShell example:
  python scripts/collect_baseline.py --output benchmarks/baseline_$(Get-Date -Format yyyyMMdd).json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import importlib
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

import psutil

try:
    import numpy as np  # noqa
except ImportError:
    print("numpy required for baseline script", file=sys.stderr)
    sys.exit(1)

PACKAGE = "calibrated_explanations"


def _measure_import_time(module_name: str) -> float:
    start = time.perf_counter()
    importlib.import_module(module_name)
    end = time.perf_counter()
    return end - start


def _list_public_api(module_name: str) -> List[str]:
    mod = sys.modules.get(module_name) or importlib.import_module(module_name)
    if hasattr(mod, "__all__") and mod.__all__:
        symbols = sorted(set(mod.__all__))
    else:
        symbols = [n for n, v in vars(mod).items() if not n.startswith("_") and not callable(v)]
        symbols += [n for n, v in vars(mod).items() if not n.startswith("_") and callable(v)]
        symbols = sorted(set(symbols))
    return symbols


def _memory_usage_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def _safe_len(obj: Any) -> int:
    try:
        return len(obj)
    except TypeError:
        return -1


def collect_runtime_benchmarks() -> Dict[str, Any]:
    """Run lightweight runtime benchmarks using small synthetic data.
    Avoid heavy dataset loads to keep CI fast.
    """
    from sklearn.datasets import load_diabetes, load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from calibrated_explanations import WrapCalibratedExplainer

    results: Dict[str, Any] = {}

    # Classification benchmark
    clf_data = load_breast_cancer()
    Xc, yc = clf_data.data, clf_data.target
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(Xc, yc)

    expl_clf = WrapCalibratedExplainer(clf)
    t0 = time.perf_counter()
    expl_clf.fit(Xc[:300], yc[:300])
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    expl_clf.calibrate(Xc[300:400], yc[300:400])
    calibrate_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = expl_clf.predict(Xc[400:450])
    infer_time = time.perf_counter() - t0

    results["classification"] = {
        "fit_time_s": fit_time,
        "calibrate_time_s": calibrate_time,
        "predict_batch_time_s": infer_time,
        "batch_size": 50,
    }

    # Regression benchmark
    reg_data = load_diabetes()
    Xr, yr = reg_data.data, reg_data.target
    reg = RandomForestRegressor(n_estimators=50, random_state=42)
    reg.fit(Xr, yr)

    expl_reg = WrapCalibratedExplainer(reg)
    t0 = time.perf_counter()
    expl_reg.fit(Xr[:250], yr[:250])
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    expl_reg.calibrate(Xr[250:350], yr[250:350])
    calibrate_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = expl_reg.predict(Xr[350:400])
    infer_time = time.perf_counter() - t0

    results["regression"] = {
        "fit_time_s": fit_time,
        "calibrate_time_s": calibrate_time,
        "predict_batch_time_s": infer_time,
        "batch_size": 50,
    }

    return results


def collect_baseline() -> Dict[str, Any]:
    data: Dict[str, Any] = {"package": PACKAGE}

    # Import timing & memory snapshot
    tracemalloc.start()
    before_mem = _memory_usage_mb()
    import_time = _measure_import_time(PACKAGE)
    current_mem = _memory_usage_mb()
    peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    symbols = _list_public_api(PACKAGE)

    data["import_time_seconds"] = import_time
    data["memory_rss_mb_before"] = before_mem
    data["memory_rss_mb_after"] = current_mem
    data["memory_tracemalloc_peak_mb"] = peak
    data["public_api_symbols"] = symbols
    data["public_api_symbol_count"] = len(symbols)

    # Runtime micro-benchmarks
    data["runtime"] = collect_runtime_benchmarks()

    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return data


def main():
    parser = argparse.ArgumentParser(description="Collect baseline metrics for calibrated_explanations")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    baseline = collect_baseline()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2 if args.pretty else None, sort_keys=True)

    print(f"Baseline metrics written to {out_path}")
    print(json.dumps({k: baseline[k] for k in ["import_time_seconds", "public_api_symbol_count"]}, indent=2))


if __name__ == "__main__":
    main()
