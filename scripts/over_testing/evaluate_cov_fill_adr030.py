"""Scan generated cov-fill tests for ADR-030 compliance.

Produces CSV output with: filename, has_assertion, uses_private_member, has_slow_or_viz_marker

Usage: python scripts/over_testing/evaluate_cov_fill_adr030.py
"""
from __future__ import annotations
import ast
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GLOB = ROOT / "tests" / "generated" / "test_cov_fill_*.py"
OUT = ROOT / "reports" / "over_testing" / "cov_fill_adr30_scan.csv"


def analyze_file(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    has_assert = False
    uses_private = False
    has_marker = False

    # simple check: look for assert statements / pytest.raises / pytest.warns
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            has_assert = True
        if isinstance(node, ast.Call):
            func = getattr(node.func, "id", None) or getattr(getattr(node.func, "attr", None), "__str__", None)
            if func in {"raises", "warns"}:
                has_assert = True
        if isinstance(node, ast.Attribute):
            if getattr(node, "attr", "").startswith("_"):
                uses_private = True
        if isinstance(node, ast.Expr):
            # look for pytest.mark.something usage in module-level decorators
            if hasattr(node, "value") and isinstance(node.value, ast.Call):
                if getattr(node.value.func, "attr", "").lower() in {"mark", "marks"}:
                    has_marker = True

    return {
        "file": str(path.relative_to(ROOT)),
        "has_assertion": has_assert,
        "uses_private_member": uses_private,
        "has_marker": has_marker,
    }


def main() -> int:
    files = sorted((Path("tests/generated")).glob("test_cov_fill_*.py"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "has_assertion", "uses_private_member", "has_marker"])
        writer.writeheader()
        for f in files:
            r = analyze_file(f.resolve())
            writer.writerow(r)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
