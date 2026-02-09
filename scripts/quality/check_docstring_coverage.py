"""Docstring coverage reporting utility for Standard-018 guardrails."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import pathlib
from typing import Iterable, Tuple


@dataclasses.dataclass
class CoverageCounter:
    """Track documented/total counts for a documentation category."""

    documented: int = 0
    total: int = 0

    def update(self, has_docstring: bool) -> None:
        self.total += 1
        if has_docstring:
            self.documented += 1

    @property
    def coverage(self) -> float:
        if self.total == 0:
            return 100.0
        return (self.documented / self.total) * 100


@dataclasses.dataclass
class CoverageReport:
    """Aggregate docstring coverage across multiple categories."""

    module: CoverageCounter = dataclasses.field(default_factory=CoverageCounter)
    classes: CoverageCounter = dataclasses.field(default_factory=CoverageCounter)
    functions: CoverageCounter = dataclasses.field(default_factory=CoverageCounter)
    methods: CoverageCounter = dataclasses.field(default_factory=CoverageCounter)

    def rows(self) -> Iterable[Tuple[str, CoverageCounter]]:
        return (
            ("Modules", self.module),
            ("Classes", self.classes),
            ("Functions", self.functions),
            ("Methods", self.methods),
        )

    @property
    def overall(self) -> float:
        total_documented = sum(counter.documented for _, counter in self.rows())
        total_items = sum(counter.total for _, counter in self.rows())
        if total_items == 0:
            return 100.0
        return (total_documented / total_items) * 100


def _iter_python_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in root.rglob("*.py"):
        if any(part.startswith(".") for part in path.parts):
            continue
        yield path


def _has_docstring(node: ast.AST) -> bool:
    try:
        return bool(ast.get_docstring(node, clean=False))
    except TypeError:
        # Some nodes (e.g., ast.Module with non-string first statement) may raise.
        return False


def _analyze_class(node: ast.ClassDef, report: CoverageReport) -> None:
    report.classes.update(_has_docstring(node))
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            report.methods.update(_has_docstring(child))
        elif isinstance(child, ast.ClassDef):
            _analyze_class(child, report)


def _analyze_module(tree: ast.Module, report: CoverageReport) -> None:
    report.module.update(_has_docstring(tree))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            _analyze_class(node, report)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            report.functions.update(_has_docstring(node))


def build_report(root: pathlib.Path) -> CoverageReport:
    report = CoverageReport()
    for file_path in _iter_python_files(root):
        with file_path.open("r", encoding="utf-8") as handle:
            try:
                tree = ast.parse(handle.read(), filename=str(file_path))
            except SyntaxError:
                continue
        _analyze_module(tree, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="src/calibrated_explanations",
        help="Directory to scan for docstring coverage (default: src/calibrated_explanations)",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Optional coverage threshold (0-100). Exit non-zero if overall coverage falls below.",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.path).resolve()
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    report = build_report(root)

    print("Docstring coverage summary (Standard-018 baseline)")
    print("=" * 48)
    for label, counter in report.rows():
        print(f"{label:10s}: {counter.documented:4d}/{counter.total:4d} ({counter.coverage:6.2f}%)")
    print("-" * 48)
    print(f"Overall    : {report.overall:6.2f}%")

    if args.fail_under is not None and report.overall < args.fail_under:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
