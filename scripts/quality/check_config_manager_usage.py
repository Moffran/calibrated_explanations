"""Enforce ConfigManager usage in migrated runtime modules (ADR-034 Task 15).

Two checks are performed:

§6 Boundary rule:
    Migrated modules must not call os.environ, os.getenv, or read_pyproject_section
    directly. Only the sanctioned boundary modules (config_manager.py, config_helpers.py)
    may ingest raw config sources.

§3 Lifecycle rule:
    ConfigManager.from_sources() must be called only in __init__ methods or at module
    level (singleton). Calling from_sources() in other methods reintroduces live-read
    behavior through repeated snapshot recreation.

LIMITATION: This checker detects direct ConfigManager.from_sources() calls by name.
It cannot detect:
- Aliases (mgr = ConfigManager.from_sources stored in a field and passed around)
- Retained managers that are invalidated and re-created inside methods
- Lifecycle compliance for third-party code that holds a ConfigManager reference

Ownership compliance for non-detectable patterns must be verified in code review
and confirmed through the injection wiring tests.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

TARGETS = {
    "cli.py",
    "cache/cache.py",
    "parallel/parallel.py",
    "core/explain/_feature_filter.py",
    "core/prediction/orchestrator.py",
    "logging.py",
    "plotting.py",
    "plugins/manager.py",
    "plugins/registry.py",
    "plugins/cli.py",
}

SANCTIONED_BOUNDARY_MODULES = {
    "core/config_manager.py",
    "core/config_helpers.py",
}

# Runtime-scope scan exclusions are explicit and temporary where possible.
RUNTIME_SCOPE_EXCLUDE = {
    "utils/deprecations.py",  # ADR-011 control path, not runtime configuration resolution.
    "plugins/_trust.py",  # Dedicated trust debug toggle helper.
}

_LIFECYCLE_ALLOWLIST: tuple[tuple[str, str, str, str], ...] = ()
_ROOT_CLI_LIFECYCLE_ALLOWED_FUNCTIONS = frozenset(
    {
        "_cmd_config_show",
        "_cmd_config_export",
    }
)

_PLUGIN_CLI_LIFECYCLE_ALLOWED_FUNCTIONS = frozenset({"_cmd_trust"})


@dataclass(frozen=True)
class Violation:
    """A disallowed runtime configuration access site."""

    file: str
    line: int
    kind: str

    def to_dict(self) -> dict[str, object]:
        rule = (
            "§6-boundary"
            if self.kind in {"os.environ", "os.getenv", "read_pyproject_section"}
            else "§3-lifecycle"
        )
        return {"file": self.file, "line": self.line, "kind": self.kind, "adr_rule": rule}


def _resolve_package_root(root: Path) -> Path:
    resolved = root.resolve()
    if resolved.name == "calibrated_explanations":
        return resolved
    candidate = resolved / "calibrated_explanations"
    if candidate.is_dir():
        return candidate
    raise ValueError(
        "root must be 'src/calibrated_explanations' or a parent directory containing it."
    )


def _is_target(rel_path: str, *, scope: str) -> bool:
    if rel_path in SANCTIONED_BOUNDARY_MODULES:
        return False
    if scope == "targeted":
        return rel_path in TARGETS
    if scope == "runtime":
        return rel_path not in RUNTIME_SCOPE_EXCLUDE
    raise ValueError(f"Unsupported scope: {scope}")


def _relative(path: Path, package_root: Path) -> str:
    return str(path.resolve().relative_to(package_root)).replace("\\", "/")


def _report_package_root(package_root: Path) -> str:
    """Return a repo-relative package root for report payloads."""
    base = package_root.parent.parent if package_root.parent.name == "src" else package_root.parent
    return str(package_root.resolve().relative_to(base.resolve())).replace("\\", "/")


def _is_os_env_access(node: ast.Attribute) -> bool:
    if node.attr != "environ":
        return False
    value = node.value
    return isinstance(value, ast.Name) and value.id == "os"


def _is_os_getenv_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != "getenv":
        return False
    value = node.func.value
    return isinstance(value, ast.Name) and value.id == "os"


def _is_pytest_current_test_probe(node: ast.Call) -> bool:
    """Return True for explicit pytest harness probe via os.getenv('PYTEST_CURRENT_TEST')."""
    if not _is_os_getenv_call(node):
        return False
    if not node.args:
        return False
    first_arg = node.args[0]
    return isinstance(first_arg, ast.Constant) and first_arg.value == "PYTEST_CURRENT_TEST"


def _is_read_pyproject_section_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "read_pyproject_section"
    if isinstance(func, ast.Attribute):
        return func.attr == "read_pyproject_section"
    return False


def _is_from_sources_call(node: ast.Call) -> bool:
    """Return True if node is ConfigManager.from_sources(...)."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr != "from_sources":
        return False
    return isinstance(func.value, ast.Name) and func.value.id == "ConfigManager"


def _lifecycle_allowed_methods() -> frozenset[tuple[str, str]]:
    """Return (rel_path, method_name) pairs that are transitionally allowed."""
    return frozenset((entry[0], entry[1]) for entry in _LIFECYCLE_ALLOWLIST)


class _LifecycleViolationFinder(ast.NodeVisitor):
    """Detect ConfigManager.from_sources() calls outside compliant ownership scopes."""

    def __init__(self, rel_path: str, allowed: frozenset[tuple[str, str]]) -> None:
        self.rel_path = rel_path
        self._allowed = allowed
        self.violations: list[Violation] = []
        self._scope_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        if _is_from_sources_call(node):
            scope = self._scope_stack[-1] if self._scope_stack else None
            if not self._is_compliant(scope):
                label = scope if scope else "<module-level>"
                self.violations.append(
                    Violation(
                        self.rel_path,
                        getattr(node, "lineno", 1),
                        f"from_sources() in '{label}' violates ADR-034 §3 construct-once",
                    )
                )
        self.generic_visit(node)

    def _is_compliant(self, scope: str | None) -> bool:
        if scope is None:
            return True
        if scope == "__init__":
            return True
        if (self.rel_path, scope) in {
            ("logging.py", "_get_module_config_manager"),
            ("plotting.py", "_get_plotting_config_manager"),
            ("plugins/registry.py", "_config_manager"),
            ("cache/cache.py", "_get_cache_config_manager"),
            ("parallel/parallel.py", "_get_parallel_config_manager"),
            ("core/explain/_feature_filter.py", "_get_feature_filter_config_manager"),
            ("utils/perturbation.py", "_get_perturbation_config_manager"),
        }:
            return True
        if self.rel_path == "cli.py" and scope in _ROOT_CLI_LIFECYCLE_ALLOWED_FUNCTIONS:
            return True
        if self.rel_path == "plugins/cli.py" and scope in _PLUGIN_CLI_LIFECYCLE_ALLOWED_FUNCTIONS:
            return True
        return (self.rel_path, scope) in self._allowed


def scan_package(package_root: Path, *, scope: str = "targeted") -> list[Violation]:
    """Return boundary and lifecycle violations for migrated modules."""
    allowed = _lifecycle_allowed_methods()
    violations: list[Violation] = []
    for path in sorted(package_root.rglob("*.py")):
        rel = _relative(path, package_root)
        if not _is_target(rel, scope=scope):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and _is_os_env_access(node):
                violations.append(Violation(rel, getattr(node, "lineno", 1), "os.environ"))
            elif isinstance(node, ast.Call) and _is_os_getenv_call(node):
                if _is_pytest_current_test_probe(node):
                    continue
                violations.append(Violation(rel, getattr(node, "lineno", 1), "os.getenv"))
            elif isinstance(node, ast.Call) and _is_read_pyproject_section_call(node):
                violations.append(
                    Violation(rel, getattr(node, "lineno", 1), "read_pyproject_section")
                )
        finder = _LifecycleViolationFinder(rel, allowed)
        finder.visit(tree)
        violations.extend(finder.violations)
    return sorted(violations, key=lambda item: (item.file, item.line, item.kind))


def write_report(path: Path, package_root: Path, violations: list[Violation]) -> None:
    """Write deterministic JSON output."""
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "package_root": _report_package_root(package_root),
        "total_violations": len(violations),
        "violations": [item.to_dict() for item in violations],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run ConfigManager usage checker."""
    parser = argparse.ArgumentParser(
        description="Ensure migrated runtime modules read config via ConfigManager only.",
    )
    parser.add_argument("--root", type=Path, default=Path("src/calibrated_explanations"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/quality/config_manager_usage_report.json"),
    )
    parser.add_argument(
        "--scope",
        choices=("targeted", "runtime"),
        default="targeted",
        help="Scanning scope: 'targeted' checks ADR-034 Phase A modules, 'runtime' scans all runtime modules except explicit exclusions.",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    try:
        package_root = _resolve_package_root(args.root)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    violations = scan_package(package_root, scope=args.scope)
    write_report(args.report, package_root, violations)
    boundary = [
        v for v in violations if v.kind in {"os.environ", "os.getenv", "read_pyproject_section"}
    ]
    lifecycle = [v for v in violations if "from_sources" in v.kind]

    if boundary:
        print("ADR-034 §6 boundary violations (direct env/pyproject reads):")
        for violation in boundary:
            print(f"  {violation.file}:{violation.line} [{violation.kind}]")
    if lifecycle:
        print("ADR-034 §3 lifecycle violations (per-lookup from_sources):")
        for violation in lifecycle:
            print(f"  {violation.file}:{violation.line} [{violation.kind}]")

    if violations:
        print(f"Report written to {args.report}")
        if _LIFECYCLE_ALLOWLIST:
            print("Active lifecycle allowlist entries (remove when injection is complete):")
            for path, method, justification, expiry in _LIFECYCLE_ALLOWLIST:
                print(f"  [{expiry}] {path}::{method} — {justification}")
        return 1 if args.check else 0

    print("ConfigManager usage check passed (boundary + lifecycle).")
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
