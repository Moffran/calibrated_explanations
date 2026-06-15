"""Runner for the evaluation/reject suite.

Supports `--quick` for fast smoke runs, executes all reject scenarios, and
regenerates the consolidated outcome summary at the end.

Core scenarios (1-6, 8-11) map to paper contributions (C1-C4) or research questions (RQ1-RQ6).
Supplementary scenarios require ``--supplementary``.
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
    "scenario_3_regression_threshold_baseline", # RQ3 - binary-event validity
    "scenario_4_ncf_weight_grid",               # C2 / RQ4 — empirical
    "scenario_5_explanation_quality",           # C4 / RQ5 — empirical
    "scenario_6_finite_sample_stress",          # RQ6 — empirical
    "scenario_8_difficulty_reject_ablation",    # Empirical ablation of the existing difficulty path
    "scenario_9_difficulty_normalized_ncf",     # Empirical ablation of direct difficulty-normalized reject scoring
    "scenario_10_ambiguity_novelty_reject",     # Empirical ablation of novelty-aware experimental reject scoring
    "scenario_11_operating_point_selection",    # Matched reject-rate operating-point selection
]

# Supplementary scenarios.
SUPPLEMENTARY_SCENARIOS = [
    "scenario_7_ncf_coverage_validity",                                # Empirical companion to Proposition 1
    "scenario_12_coverage_validity_difficulty_normalized",             # RT-3: arm A vs arm C coverage validity
    "scenario_13_ncal_coverage_sweep",                                 # RT-3 follow-up: n_cal sweep, variance-inflation hypothesis
    "scenario_14_routing_policy_contract",                             # Routing contract: FLAG / ONLY_ACCEPTED / ONLY_REJECTED
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
        help="Also run supplementary scenarios",
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
