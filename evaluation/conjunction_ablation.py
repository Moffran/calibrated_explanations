"""Benchmark legacy vs current conjunction builders for factual/alternative explanations.

Usage:
    python evaluation/conjunction_ablation.py --runs 5 --instances 8

Writes aggregated results to evaluation/conjunction_ablation_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibrated_explanations.explanations.legacy_conjunctions import (
    add_conjunctions_alternative_legacy,
    add_conjunctions_factual_legacy,
)
from tests.helpers.dataset_utils import make_binary_dataset, make_regression_dataset
from tests.helpers.explainer_utils import make_explainer_from_dataset

RESULTS_FILE = Path("evaluation/conjunction_ablation_results.json")
RULE_SIZES = (2, 3, 4, 5)
N_TOP_FEATURES = 10


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _snapshot_conjunction_payload(explanation) -> Dict[str, Any]:
    if not getattr(explanation, "_has_conjunctive_rules", False):
        raise RuntimeError("Explanation has no conjunctive rules to snapshot.")
    return _normalize(explanation.conjunctive_rules)


def _build_collection(explainer, x_batch: np.ndarray, *, mode: str):
    if mode == "factual":
        return explainer.explain_factual(x_batch)
    if mode == "alternative":
        return explainer.explore_alternatives(x_batch)
    raise ValueError(f"Unsupported mode {mode}")


def _run_once(
    explainer,
    x_batch: np.ndarray,
    *,
    rule_size: int,
    mode: str,
    n_top_features: int,
) -> Tuple[float | None, float, float | None, bool]:
    """Return (new_duration, batched_duration, legacy_duration, parity_flag)."""
    # Build fresh explanation collections for each measurement run.
    new_collection = _build_collection(explainer, x_batch, mode=mode)
    batched_collection = _build_collection(explainer, x_batch, mode=mode)
    legacy_collection = _build_collection(explainer, x_batch, mode=mode)

    new_exp = new_collection[0]
    batched_exp = batched_collection[0]
    legacy_exp = legacy_collection[0]

    # 1. Sequential
    sequential_duration = None
    if rule_size < 4:
        start = time.perf_counter()
        new_exp.add_conjunctions(
            n_top_features=n_top_features, max_rule_size=rule_size, _use_batched=False
        )
        sequential_duration = time.perf_counter() - start

    # 2. Batched (Newest)
    start = time.perf_counter()
    batched_exp.add_conjunctions(
        n_top_features=n_top_features, max_rule_size=rule_size, _use_batched=True
    )
    batched_duration = time.perf_counter() - start

    # 3. Legacy
    legacy_duration = None
    if rule_size < 4:
        start = time.perf_counter()
        if mode == "factual":
            add_conjunctions_factual_legacy(
                legacy_exp, n_top_features=n_top_features, max_rule_size=rule_size
            )
        else:
            add_conjunctions_alternative_legacy(
                legacy_exp, n_top_features=n_top_features, max_rule_size=rule_size
            )
        legacy_duration = time.perf_counter() - start

    # Parity check (only if legacy ran)
    parity = True
    if legacy_duration is not None:
        # Check Sequential vs Legacy
        if sequential_duration is not None:
            parity = parity and _payloads_equal(
                _snapshot_conjunction_payload(new_exp),
                _snapshot_conjunction_payload(legacy_exp)
            )
        # Check Batched vs Legacy
        parity = parity and _payloads_equal(
            _snapshot_conjunction_payload(batched_exp),
            _snapshot_conjunction_payload(legacy_exp)
        )

    return sequential_duration, batched_duration, legacy_duration, parity


def _payloads_equal(p1: Dict[str, Any], p2: Dict[str, Any]) -> bool:
    """Compare two payloads with tolerance for floating point differences."""
    if p1.keys() != p2.keys():
        print(f"Keys mismatch: {p1.keys()} != {p2.keys()}")
        return False
    for k in p1:
        v1 = p1[k]
        v2 = p2[k]
        if isinstance(v1, (list, tuple, np.ndarray)):
            if len(v1) != len(v2):
                print(f"Length mismatch for key {k}: {len(v1)} != {len(v2)}")
                return False
            for i in range(len(v1)):
                val1 = v1[i]
                val2 = v2[i]
                if isinstance(val1, (float, np.floating)) or isinstance(val2, (float, np.floating)):
                    if not np.isclose(val1, val2, rtol=1e-5, atol=1e-8):
                        print(f"Value mismatch for key {k} at index {i}: {val1} != {val2}")
                        return False
                elif isinstance(val1, (list, tuple, np.ndarray)):
                     # Recursive check for nested lists (like features)
                     if not _nested_equal(val1, val2):
                         print(f"Array mismatch for key {k} at index {i}: {val1} != {val2}")
                         return False
                elif val1 != val2:
                    print(f"Value mismatch for key {k} at index {i}: {val1} != {val2}")
                    return False
        elif isinstance(v1, (float, np.floating)) or isinstance(v2, (float, np.floating)):
            if not np.isclose(v1, v2, rtol=1e-5, atol=1e-8):
                print(f"Value mismatch for key {k}: {v1} != {v2}")
                return False
        elif v1 != v2:
            print(f"Value mismatch for key {k}: {v1} != {v2}")
            return False
    return True

def _nested_equal(v1, v2):
    if isinstance(v1, (list, tuple, np.ndarray)) and isinstance(v2, (list, tuple, np.ndarray)):
        if len(v1) != len(v2):
            return False
        for i in range(len(v1)):
            if not _nested_equal(v1[i], v2[i]):
                return False
        return True
    if isinstance(v1, (float, np.floating)) or isinstance(v2, (float, np.floating)):
        return np.isclose(v1, v2, rtol=1e-5, atol=1e-8)
    return v1 == v2


def _aggregate(durations: Iterable[float]) -> Dict[str, Any]:
    runs = list(durations)
    return {
        "runs": runs,
        "mean": mean(runs) if runs else None,
        "min": min(runs) if runs else None,
        "max": max(runs) if runs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy vs current conjunction builders.")
    parser.add_argument("--runs", type=int, default=5, help="Number of repeated measurements.")
    parser.add_argument(
        "--instances",
        type=int,
        default=8,
        help="Number of test instances to include in each measurement batch.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_FILE),
        help="Where to write the aggregated results JSON.",
    )
    args = parser.parse_args()

    tasks = [
        ("classification", make_binary_dataset),
        ("regression", make_regression_dataset),
    ]

    results: Dict[str, Any] = {
        "meta": {
            "timestamp": time.time(),
            "runs": args.runs,
            "instances": args.instances,
            "rule_sizes": list(RULE_SIZES),
            "n_top_features": N_TOP_FEATURES,
        }
    }

    for task_name, dataset_factory in tasks:
        print(f"Running ablation for {task_name}...")
        dataset = dataset_factory()
        explainer, x_test = make_explainer_from_dataset(dataset, mode=task_name)
        batch = x_test[: args.instances]

        task_results = {}

        for mode in ("factual", "alternative"):
            mode_results: Dict[str, Any] = {}

            for rule_size in RULE_SIZES:
                sequential_times = []
                batched_times = []
                legacy_times = []
                parity = True

                for _ in range(args.runs):
                    seq_dur, batched_dur, legacy_dur, match = _run_once(
                        explainer,
                        batch,
                        rule_size=rule_size,
                        mode=mode,
                        n_top_features=N_TOP_FEATURES,
                    )
                    if seq_dur is not None:
                        sequential_times.append(seq_dur)
                    if batched_dur is not None:
                        batched_times.append(batched_dur)
                    if legacy_dur is not None:
                        legacy_times.append(legacy_dur)
                    parity = parity and match

                summary = {
                    "sequential": _aggregate(sequential_times) if sequential_times else None,
                    "batched": _aggregate(batched_times) if batched_times else None,
                    "legacy": _aggregate(legacy_times) if legacy_times else None,
                    "parity": parity,
                }
                if summary["legacy"] and summary["legacy"]["mean"]:
                    if summary["sequential"] and summary["sequential"]["mean"]:
                        summary["speedup_legacy_vs_sequential"] = (
                            summary["legacy"]["mean"] / summary["sequential"]["mean"]
                        )
                    if summary["batched"] and summary["batched"]["mean"]:
                        summary["speedup_legacy_vs_batched"] = (
                            summary["legacy"]["mean"] / summary["batched"]["mean"]
                        )
                if (
                    summary["sequential"]
                    and summary["sequential"]["mean"]
                    and summary["batched"]
                    and summary["batched"]["mean"]
                ):
                    summary["speedup_sequential_vs_batched"] = (
                        summary["sequential"]["mean"] / summary["batched"]["mean"]
                    )

                mode_results[f"rule_size_{rule_size}"] = summary

            task_results[mode] = mode_results
        results[task_name] = task_results

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print_summary(results)
    print(f"Saved results to {output_path}")


def print_summary(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of the results."""
    print("\n" + "=" * 80)
    print(f"{'CONJUNCTION ABLATION SUMMARY':^80}")
    print("=" * 80)

    for task_name in ["classification", "regression"]:
        if task_name not in results:
            continue

        print(f"\nTask: {task_name.upper()}")

        for mode in ["factual", "alternative"]:
            if mode not in results[task_name]:
                continue

            print(f"\n  Mode: {mode.upper()}")
            print(f"  {'Rule Size':<10} | {'Legacy (s)':<12} | {'Sequential (s)':<15} | {'Batched (s)':<12} | {'Speedup (L/B)':<15} | {'Speedup (S/B)':<15}")
            print("  " + "-" * 95)

            mode_results = results[task_name][mode]
            # Sort by rule size (extract integer from key "rule_size_X")
            sorted_keys = sorted(mode_results.keys(), key=lambda k: int(k.split("_")[-1]))

            for key in sorted_keys:
                rule_size = key.split("_")[-1]
                data = mode_results[key]

                legacy_mean = data.get("legacy", {}).get("mean") if data.get("legacy") else None
                sequential_mean = data.get("sequential", {}).get("mean") if data.get("sequential") else None
                batched_mean = data.get("batched", {}).get("mean") if data.get("batched") else None

                legacy_str = f"{legacy_mean:.4f}" if legacy_mean is not None else "N/A"
                sequential_str = f"{sequential_mean:.4f}" if sequential_mean is not None else "N/A"
                batched_str = f"{batched_mean:.4f}" if batched_mean is not None else "N/A"

                speedup_lb = data.get("speedup_legacy_vs_batched")
                speedup_lb_str = f"{speedup_lb:.2f}x" if speedup_lb is not None else "-"

                speedup_sb = data.get("speedup_sequential_vs_batched")
                speedup_sb_str = f"{speedup_sb:.2f}x" if speedup_sb is not None else "-"

                print(f"  {rule_size:<10} | {legacy_str:<12} | {sequential_str:<15} | {batched_str:<12} | {speedup_lb_str:<15} | {speedup_sb_str:<15}")




if __name__ == "__main__":
    main()
