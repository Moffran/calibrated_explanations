"""Runner for the evaluation/reject suite.

Supports `--quick` for fast smoke runs, executes all reject scenarios, and
regenerates the consolidated outcome summary at the end.

All core scenarios (1–6) map directly to paper contributions (C1–C4) or research questions
(RQ1–RQ6).  Scenario 7 is supplementary and requires the RT-2 sigma-normalisation fix.
Run with ``--supplementary`` to include it.
"""
from __future__ import annotations

import argparse
import importlib
from time import perf_counter

from .common_reject import RunConfig
from .summarize_results import summarize

# Core research scenarios — directly map to paper RQs and contributions
CORE_SCENARIOS = [
    "scenario_1_binary_coverage",               # C1 / RQ1 — formal target
    "scenario_2_multiclass_correctness",        # C2 / RQ2 — empirical
    "scenario_3_regression_threshold_baseline", # RQ3 — empirical baseline
    "scenario_4_ncf_weight_grid",               # C2 / RQ4 — empirical
    "scenario_5_explanation_quality",           # C4 / RQ5 — empirical
    "scenario_6_finite_sample_stress",          # RQ6 — empirical
]

# Supplementary scenarios — depend on RT-2 K1 fix being complete
SUPPLEMENTARY_SCENARIOS = [
    "scenario_7_ncf_coverage_validity",         # Empirical companion to Proposition 1
]

SCENARIOS = CORE_SCENARIOS + SUPPLEMENTARY_SCENARIOS


def run_all(quick: bool = True, supplementary: bool = False) -> None:
    """Run every reject scenario and regenerate the top-level summary."""
    to_run = CORE_SCENARIOS + (SUPPLEMENTARY_SCENARIOS if supplementary else [])
    cfg = RunConfig(seed=42, quick=quick)
    for s in to_run:
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
        "--supplementary",
        action="store_true",
        help="Also run supplementary scenarios (requires RT-2 K1 fix)",
    )
    arguments = parser.parse_args()
    quick_mode = not bool(arguments.full)
    to_run = CORE_SCENARIOS + (SUPPLEMENTARY_SCENARIOS if arguments.supplementary else [])

    cfg = RunConfig(seed=42, quick=quick_mode)
    for s in to_run:
        mod = importlib.import_module(f"evaluation.reject.{s}")
        print(f"Running {s} (quick={quick_mode})...")
        started = perf_counter()
        mod.run(cfg)
        elapsed = perf_counter() - started
        print(f"  -> completed in {elapsed:.1f}s")
    summarize()
