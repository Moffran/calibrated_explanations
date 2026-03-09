"""Runner for the evaluation/reject suite.

Supports `--quick` for fast smoke runs, executes all reject scenarios, and
regenerates the consolidated outcome summary at the end.
"""
from __future__ import annotations

import argparse
import importlib
from time import perf_counter

from .common_reject import RunConfig
from .summarize_results import summarize

# Validation scenarios (integration / API validation)
VALIDATION_SCENARIOS = [
    "scenario_a_policy_matrix",
    "scenario_b_ncf_sweep",
    "scenario_c_confidence_monotonicity",
    "scenario_d_regression_threshold",
]

# Research scenarios (paper evaluation, RQ-mapped)
RESEARCH_SCENARIOS = [
    "scenario_e_binary_coverage_sweep",
    "scenario_f_multiclass_coverage",
    "scenario_g_regression_coverage",
    "scenario_h_ncf_grid",
    "scenario_i_explanation_quality",
    "scenario_j_stress_tests",
    "scenario_k_mondrian_regression",
]

SCENARIOS = VALIDATION_SCENARIOS + RESEARCH_SCENARIOS


def run_all(quick: bool = True) -> None:
    """Run every reject scenario and regenerate the top-level summary."""
    cfg = RunConfig(seed=42, quick=quick)
    for s in SCENARIOS:
        mod = importlib.import_module(f"evaluation.reject.{s}")
        print(f"Running {s} (quick={quick})...")
        started = perf_counter()
        mod.run(cfg)
        elapsed = perf_counter() - started
        print(f"  -> completed in {elapsed:.1f}s")
    summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run the explicit quick dataset subsets")
    parser.add_argument("--full", action="store_true", help="Run the full dataset registry")
    parser.add_argument(
        "--research-only",
        action="store_true",
        help="Run only the research-tier scenarios (E--K)",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Run only the validation-tier scenarios (A--D)",
    )
    arguments = parser.parse_args()
    quick_mode = not bool(arguments.full)
    if arguments.research_only:
        to_run = RESEARCH_SCENARIOS
    elif arguments.validation_only:
        to_run = VALIDATION_SCENARIOS
    else:
        to_run = SCENARIOS

    cfg = RunConfig(seed=42, quick=quick_mode)
    for s in to_run:
        mod = importlib.import_module(f"evaluation.reject.{s}")
        print(f"Running {s} (quick={quick_mode})...")
        started = perf_counter()
        mod.run(cfg)
        elapsed = perf_counter() - started
        print(f"  -> completed in {elapsed:.1f}s")
    summarize()
