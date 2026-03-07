"""Master runner for the guarded explanation evaluation suite.

Calls each scenario script as a subprocess, collects exit codes, and writes
a top-level summary report.

Usage:
    python run_all_guarded.py                  # all scenarios, full grid
    python run_all_guarded.py --quick          # fast smoke-test for all
    python run_all_guarded.py --scenarios b,e  # run only B and E
    python run_all_guarded.py --scenarios all --quick
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


SCENARIO_SCRIPTS: Dict[str, str] = {
    "a": "scenario_a_guarded_vs_standard.py",
    "b": "scenario_b_ood_detection_quality.py",
    "c": "scenario_c_regression.py",
    "d": "scenario_d_real_datasets.py",
    "e": "scenario_e_edge_cases.py",
}

SCENARIO_LABELS: Dict[str, str] = {
    "a": "Scenario A — Domain plausibility (synthetic constraint)",
    "b": "Scenario B — OOD detection quality",
    "c": "Scenario C — Regression invariants",
    "d": "Scenario D — Real dataset correctness",
    "e": "Scenario E — Edge case behavior",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated list of scenario letters (a,b,c,d,e) or 'all'.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Pass --quick to each scenario script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "guarded",
        help="Root output directory for the summary report.",
    )
    return parser.parse_args()


def _run_scenario(
    key: str,
    script: str,
    quick: bool,
    script_dir: Path,
) -> Dict:
    """Run one scenario script as a subprocess. Returns a result dict."""
    cmd = [sys.executable, str(script_dir / script)]
    if quick:
        cmd.append("--quick")

    label = SCENARIO_LABELS.get(key, key)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=str(script_dir), capture_output=False, check=False)
        exit_code = result.returncode
    except Exception as exc:  # noqa: BLE001
        exit_code = -1
        print(f"  [ERROR] Failed to launch: {exc}")

    elapsed = time.perf_counter() - t0
    status = "PASS" if exit_code == 0 else "FAIL"
    print(f"  → {status} (exit={exit_code}, {elapsed:.1f}s)")

    return {
        "scenario": key,
        "label": label,
        "status": status,
        "exit_code": exit_code,
        "runtime_s": round(elapsed, 1),
    }


def _write_summary_report(results: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "summary_report.md"

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    total_time = sum(r["runtime_s"] for r in results)

    lines = [
        "# Guarded Explanation Evaluation — Summary Report",
        "",
        f"**{n_pass} PASS / {n_fail} FAIL** across {len(results)} scenario(s).",
        f"Total runtime: {total_time:.1f}s",
        "",
        "## Results by Scenario",
        "",
        "| Scenario | Status | Runtime (s) |",
        "|---|---|---|",
    ]
    for r in results:
        icon = "✓" if r["status"] == "PASS" else "✗"
        lines.append(f"| {r['label']} | {icon} {r['status']} | {r['runtime_s']} |")

    lines += [
        "",
        "## Per-Scenario Reports",
        "",
        "Each scenario writes its own `report.md` under `artifacts/guarded/scenario_*/`.",
        "See those files for metric details and interpretation.",
        "",
        "## Metrics Quick Reference",
        "",
        "| Metric | Scenario | Claim | Healthy | Red flag |",
        "|---|---|---|---|---|",
        "| `violation_rate` (guarded < standard) | A | Detection | Guarded lower | Guarded ≥ standard |",
        "| `auroc` | B | Detection | > 0.80 for moderate+ shift | < 0.60 for extreme shift |",
        "| `fpr_at_significance` | B | Calibration | ≈ significance | >> significance |",
        "| `n_invariant_violations` | C | Correctness | 0 always | Any > 0 = bug |",
        "| `audit_field_completeness` | D | Correctness | True always | Any False = bug |",
        "| `fraction_instances_fully_filtered` | D | Usability | < 0.05 at α=0.10 | > 0.10 |",
        "| Edge case PASS/FAIL | E | Correctness | All PASS | Any unexpected FAIL |",
    ]

    if n_fail > 0:
        lines += [
            "",
            "## Failed Scenarios",
            "",
        ]
        for r in results:
            if r["status"] == "FAIL":
                lines.append(f"- **{r['label']}** (exit code {r['exit_code']})")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote summary: {report_path}")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).parent

    if args.scenarios.strip().lower() == "all":
        selected = list(SCENARIO_SCRIPTS.keys())
    else:
        selected = [s.strip().lower() for s in args.scenarios.split(",") if s.strip()]
        unknown = [s for s in selected if s not in SCENARIO_SCRIPTS]
        if unknown:
            print(f"Unknown scenarios: {unknown}. Valid: {list(SCENARIO_SCRIPTS.keys())}")
            sys.exit(1)

    results: List[Dict] = []
    for key in selected:
        script = SCENARIO_SCRIPTS[key]
        result = _run_scenario(key, script, args.quick, script_dir)
        results.append(result)

    _write_summary_report(results, args.output_dir)

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
