"""Conservative pruning helper for generated tests.

This script does NOT alter tests by default. It reads `reports/over_testing/cov_fill_adr30_scan.csv`
and/or `reports/over_testing/gaps.csv` and produces `reports/over_testing/prune_plan.json`
with recommended deletions. It requires manual review before deletions are applied.

Usage:
  python scripts/over_testing/prune_generated_tests.py --plan
  python scripts/over_testing/prune_generated_tests.py --apply  # after manual review
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "reports" / "over_testing" / "cov_fill_adr30_scan.csv"
PLAN = ROOT / "reports" / "over_testing" / "prune_plan.json"


def load_scan() -> List[dict]:
    import csv
    if not SCAN.exists():
        return []
    out = []
    with SCAN.open(encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for row in r:
            out.append(row)
    return out


def make_plan(rows: List[dict]) -> dict:
    # Conservative heuristics: propose removal iff no assertions and not marked
    to_remove = []
    questionable = []
    for r in rows:
        if r["has_assertion"].lower() in {"false", "0", ""} and r["has_marker"].lower() in {"false", "0", ""}:
            to_remove.append(r["file"])
        else:
            questionable.append(r["file"])
    plan = {"proposed_removals": to_remove, "questionable": questionable}
    return plan


def apply_plan(plan: dict) -> None:
    # Backup then delete files listed in plan['proposed_removals']
    import shutil
    bdir = ROOT / "reports" / "over_testing" / "backup_removed_tests"
    bdir.mkdir(parents=True, exist_ok=True)
    for f in plan.get("proposed_removals", []):
        p = ROOT / f
        if p.exists():
            shutil.move(str(p), str(bdir / p.name))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="Produce prune_plan.json")
    parser.add_argument("--apply", action="store_true", help="Apply plan (move files)")
    parser.add_argument("--aggressive", action="store_true", help="Use aggressive heuristic to drop many generated tests")
    parser.add_argument("--keep-every", type=int, default=6, help="When aggressive, keep every Nth generated test (default=6)")
    args = parser.parse_args(argv)

    rows = load_scan()
    plan = make_plan(rows)

    # Aggressive mode: propose removals from tests/generated pattern while keeping a sample
    if args.aggressive:
        gen_dir = ROOT / "tests" / "generated"
        candidates = sorted([p for p in gen_dir.glob("test_cov_fill_*.py")])
        proposed = []
        keep_every = max(1, args.keep_every)
        for i, p in enumerate(candidates):
            # keep every Nth, propose removal for others
            if (i % keep_every) != 0:
                proposed.append(str(p.relative_to(ROOT)))
        plan = {"proposed_removals": proposed, "questionable": [r["file"] for r in rows]}

    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Wrote prune plan to {PLAN}")
    if args.apply:
        if not plan.get("proposed_removals"):
            print("No proposed removals in plan. Nothing to apply.")
            return 0
        apply_plan(plan)
        print("Applied removals (moved to reports/over_testing/backup_removed_tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
