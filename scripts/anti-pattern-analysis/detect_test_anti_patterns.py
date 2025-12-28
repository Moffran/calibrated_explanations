"""Simple anti-pattern scanner for the test suite."""

from __future__ import annotations

import argparse
import ast
import csv
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    """Represents a single anti-pattern finding."""

    path: Path
    line: int
    pattern: str
    snippet: str


class AntiPatternVisitor(ast.NodeVisitor):
    """AST visitor that records anti-pattern findings."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[Finding] = []
        self.lines = path.read_text(encoding="utf-8").splitlines()

    def visit_call(self, node: ast.Call) -> None:
        """Handle function calls that might trigger anti-patterns."""
        if self._is_private_helper_call(node):
            self._record(node, "private helper call")
        if self._is_pytest_frozen_instance(node):
            self._record(node, "pytest.raises(FrozenInstanceError)")
        self.generic_visit(node)

    visit_Call = visit_call

    def visit_with(self, node: ast.With) -> None:
        """Capture context managers that raise FrozenInstanceError."""
        for item in node.items:
            if self._is_pytest_frozen_instance(item.context_expr):
                self._record(item.context_expr, "pytest.raises(FrozenInstanceError)")
        self.generic_visit(node)

    visit_With = visit_with

    def visit_subscript(self, node: ast.Subscript) -> None:
        """Detect dictionary key access on serialized payloads."""
        if isinstance(node.value, ast.Call) and self._is_to_dict_call(node.value):
            self._record(node, "to_dict() dict key access")
        self.generic_visit(node)

    visit_Subscript = visit_subscript

    def visit_compare(self, node: ast.Compare) -> None:
        """Catch exact list comparisons that include file paths."""
        candidates = [node.left] + node.comparators
        if any(self._is_path_list(element) for element in candidates):
            self._record(node, "exact path list comparison")
        self.generic_visit(node)

    visit_Compare = visit_compare

    def _record(self, node: ast.AST, pattern: str) -> None:
        lineno = getattr(node, "lineno", 0)
        snippet = self._format_snippet(lineno)
        self.findings.append(Finding(self.path, lineno, pattern, snippet))

    def _format_snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return textwrap.shorten(self.lines[lineno - 1].strip(), width=120)
        return ""

    def _is_private_helper_call(self, node: ast.Call) -> bool:
        func = node.func
        match func:
            case ast.Name(id=name):
                return name.startswith("_") and not name.startswith("__")
            case ast.Attribute(attr=attr):
                return attr.startswith("_") and not attr.startswith("__")
            case _:
                return False

    def _is_pytest_frozen_instance(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        if not self._is_pytest_raises(node.func):
            return False
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id == "FrozenInstanceError":
                return True
        return False

    def _is_to_dict_call(self, call: ast.Call) -> bool:
        func = call.func
        return isinstance(func, ast.Attribute) and func.attr == "to_dict"

    def _is_path_list(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.List):
            return False
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                if "/" in element.value or "\\" in element.value:
                    return True
        return False

    def _is_pytest_raises(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute) and node.attr == "raises":
            return True
        if isinstance(node, ast.Name) and node.id == "raises":
            return True
        return False


def scan_tests(tree_root: Path) -> list[Finding]:
    """Walk the test tree and collect anti-pattern findings."""
    findings: list[Finding] = []
    tree_root = tree_root.resolve()
    for path in tree_root.rglob("*.py"):
        if not path.is_file():
            continue
        if not path.is_relative_to(tree_root):
            continue
        resolved = path.resolve()
        visitor = AntiPatternVisitor(resolved)
        try:
            visitor.visit(ast.parse(resolved.read_text(encoding="utf-8")))
        except SyntaxError as exc:
            print(f"Unable to parse {path}: {exc}")
            continue
        findings.extend(visitor.findings)
    return findings


def write_report(findings: list[Finding], output_path: Path, root: Path | None = None) -> None:
    """Write the findings into a CSV report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "line", "pattern", "snippet"])
        for finding in findings:
            try:
                subject = finding.path.relative_to(root) if root else finding.path
            except ValueError:
                subject = finding.path.relative_to(Path.cwd())
            writer.writerow([
                str(subject),
                finding.line,
                finding.pattern,
                finding.snippet,
            ])


def main() -> int:
    """Parse CLI args, run the scan, and output the report."""
    parser = argparse.ArgumentParser(description="Detect test anti-patterns.")
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=Path("tests"),
        help="Path to the tests directory to scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/anti-pattern-analysis/test_anti_pattern_report.csv"),
        help="CSV path for the generated report.",
    )
    args = parser.parse_args()
    findings = scan_tests(args.tests_dir)
    write_report(findings, args.output, args.tests_dir.resolve())
    print(f"Found {len(findings)} anti-patterns.")
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
