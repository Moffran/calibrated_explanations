"""Enforce minimum logger-domain policy for library code.

This guard implements a narrow STD-005/ADR-028 check:
- library loggers must live under the ``calibrated_explanations`` namespace,
  or use ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Violation:
    """A logger-domain violation record."""

    file: str
    line: int
    logger_name: str
    reason: str

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serialisable record."""
        return {
            "file": self.file,
            "line": self.line,
            "logger_name": self.logger_name,
            "reason": self.reason,
        }


def _resolve_package_root(root: Path) -> Path:
    """Resolve ``src/calibrated_explanations`` package path from *root*."""
    resolved = root.resolve()
    if resolved.name == "calibrated_explanations":
        return resolved
    candidate = resolved / "calibrated_explanations"
    if candidate.is_dir():
        return candidate
    raise ValueError(
        "root must be 'src/calibrated_explanations' or a parent directory containing it."
    )


def _relative(path: Path, package_root: Path) -> str:
    return str(path.resolve().relative_to(package_root.parent.resolve())).replace("\\", "/")


def _is_logging_getlogger(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "getLogger":
        return False
    return isinstance(func.value, ast.Name) and func.value.id == "logging"


def _extract_logger_name(node: ast.Call) -> tuple[str | None, str | None]:
    """Extract logger name and violation reason for a ``logging.getLogger`` call."""
    if not node.args:
        return "", "empty logger name is not allowed in library code"
    arg0 = node.args[0]
    if isinstance(arg0, ast.Name) and arg0.id == "__name__":
        return None, None
    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
        value = arg0.value
        if value.startswith("calibrated_explanations"):
            return None, None
        return value, "logger literal must start with 'calibrated_explanations'"
    # Minimum enforcement only validates literal names. Dynamic names are allowed.
    return None, None


def scan_package(package_root: Path) -> list[Violation]:
    """Scan package files for logger-domain violations."""
    violations: list[Violation] = []
    for path in sorted(package_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        rel = _relative(path, package_root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_logging_getlogger(node):
                continue
            logger_name, reason = _extract_logger_name(node)
            if reason is None:
                continue
            violations.append(
                Violation(
                    file=rel,
                    line=getattr(node, "lineno", 1),
                    logger_name=logger_name or "",
                    reason=reason,
                )
            )
    return sorted(violations, key=lambda item: (item.file, item.line, item.logger_name))


def write_report(report_path: Path, package_root: Path, violations: list[Violation]) -> None:
    """Write deterministic JSON report."""
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "package_root": str(package_root).replace("\\", "/"),
        "total_violations": len(violations),
        "violations": [v.to_record() for v in violations],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    """Run logger-domain enforcement check."""
    parser = argparse.ArgumentParser(
        description="Check that library logger domains follow calibrated_explanations namespace rules.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("src/calibrated_explanations"),
        help="Path to package root or parent directory containing calibrated_explanations.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/quality/logging_domain_report.json"),
        help="JSON report output path.",
    )
    args = parser.parse_args(argv)

    try:
        package_root = _resolve_package_root(args.root)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    violations = scan_package(package_root)
    write_report(args.report, package_root, violations)
    if violations:
        print("Logger domain violations detected:")
        for violation in violations:
            print(
                f"- {violation.file}:{violation.line} [{violation.logger_name}] {violation.reason}"
            )
        print(f"Report written to {args.report}")
        return 1

    print("Logger domain check passed (no violations).")
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
