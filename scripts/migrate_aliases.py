"""Migration helper: scan Python files and suggest replacing deprecated alias
parameters with their canonical equivalents from calibrated_explanations.api.params.

Usage (local):
    python scripts/migrate_aliases.py path/to/code

This script is intentionally conservative: it only prints suggestions and does not
modify files. It looks for simple token occurrences of the alias names and
prints a diff-style suggestion.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict

from calibrated_explanations.api.params import ALIAS_MAP


def find_alias_usages(path: Path) -> Dict[Path, Dict[str, int]]:
    usages: Dict[Path, Dict[str, int]] = {}
    for py in path.rglob("*.py"):
        try:
            src = py.read_text(encoding="utf8")
        except Exception:
            continue
        tree = ast.parse(src)
        counts: Dict[str, int] = {}

        class Visitor(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name) -> None:
                if node.id in ALIAS_MAP:
                    counts[node.id] = counts.get(node.id, 0) + 1
                self.generic_visit(node)

        Visitor().visit(tree)
        if counts:
            usages[py] = counts
    return usages


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate_aliases.py PATH")
        return 2
    target = Path(sys.argv[1])
    if not target.exists():
        print("Path does not exist:", target)
        return 2
    usages = find_alias_usages(target)
    if not usages:
        print("No deprecated alias usages found under", target)
        return 0
    for f, counts in usages.items():
        print(f"\nFile: {f}")
        for alias, count in counts.items():
            print(f"  {alias}: {count} occurrence(s) -> suggested: {ALIAS_MAP[alias]}")
    print("\nNote: This script only suggests replacements. Review and apply changes manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
