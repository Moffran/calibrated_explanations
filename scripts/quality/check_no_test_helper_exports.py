"""Block production exports that expose test-helper wrappers.

This guard enforces ADR-030 quality intent: test scaffolding should not be
published as production API surface in ``src/calibrated_explanations``.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

TEST_HELPER_PHRASES = (
    "testing helper",
    "test helper",
    "used by tests",
    "used in tests",
    "for tests",
    "for testing",
    "testing purposes",
)

IGNORED_MODULE_PREFIXES = (
    "calibrated_explanations.testing",
)

# Registry symbols explicitly identified as over-scoped/anti-pattern exports.
BANNED_EXPORTS_BY_MODULE: dict[str, set[str]] = {
    "calibrated_explanations.plugins.registry": {
        "clear_explanation_plugins",
        "clear_interval_plugins",
        "clear_plot_plugins",
        "find_plot_plugin_trusted",
        "find_plot_renderer_trusted",
        "mark_plot_builder_trusted",
        "mark_plot_builder_untrusted",
        "mark_plot_renderer_trusted",
        "mark_plot_renderer_untrusted",
    },
    "calibrated_explanations.plugins": {
        "_EXPLANATION_PLUGINS",
        "_INTERVAL_PLUGINS",
        "_PLOT_BUILDERS",
        "_PLOT_RENDERERS",
        "_PLOT_STYLES",
        "clear_explanation_plugins",
        "clear_interval_plugins",
        "clear_plot_plugins",
        "find_plot_plugin_trusted",
    },
}


@dataclass(frozen=True)
class Violation:
    """A single blocked export violation."""

    module: str
    file: str
    symbol: str
    line: int
    reason: str

    def to_record(self) -> dict[str, object]:
        """Return a JSON-safe record."""
        return {
            "module": self.module,
            "file": self.file,
            "symbol": self.symbol,
            "line": self.line,
            "reason": self.reason,
        }


def _resolve_package_root(root: Path) -> Path:
    """Return the ``calibrated_explanations`` package directory under *root*."""
    root = root.resolve()
    if root.name == "calibrated_explanations":
        return root
    candidate = root / "calibrated_explanations"
    if candidate.is_dir():
        return candidate.resolve()
    raise ValueError(
        "root must point to 'src/calibrated_explanations' or a parent containing it."
    )


def _module_name(path: Path, package_root: Path) -> str:
    """Convert *path* into dotted module form."""
    rel = path.resolve().relative_to(package_root.parent.resolve()).with_suffix("")
    return ".".join(rel.parts)


def _relative_path(path: Path, package_root: Path) -> str:
    """Return a stable forward-slash path for reports."""
    return str(path.resolve().relative_to(package_root.parent.resolve())).replace("\\", "/")


def _extract_exports(tree: ast.Module) -> dict[str, int]:
    """Extract static ``__all__`` string entries from a module AST."""
    exports: dict[str, int] = {}
    for node in tree.body:
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            value = node.value
        if value is None:
            continue
        if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            continue
        for element in value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                exports[element.value] = getattr(element, "lineno", node.lineno)
    return exports


def _collect_public_defs(tree: ast.Module) -> dict[str, ast.AST]:
    """Map top-level public definition names to AST nodes."""
    definitions: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
    return definitions


def _contains_test_helper_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in TEST_HELPER_PHRASES)


def _contains_helper_intent(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("helper", "wrapper", "alias", "shim"))


def _looks_like_test_helper_name(symbol: str) -> bool:
    lowered = symbol.lower()
    return any(token in lowered for token in ("for_testing", "testing_helper", "test_helper"))


def _is_private_delegate(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether function body is a thin call-through to a private helper."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1:
        return False
    statement = body[0]
    call: ast.AST | None = None
    if isinstance(statement, (ast.Return, ast.Expr)):
        call = statement.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if isinstance(func, ast.Name):
        return func.id.startswith("_") and not func.id.startswith("__")
    if isinstance(func, ast.Attribute):
        return func.attr.startswith("_") and not func.attr.startswith("__")
    return False


def _should_ignore_module(module: str) -> bool:
    return any(module.startswith(prefix) for prefix in IGNORED_MODULE_PREFIXES)


def _find_dynamic_test_imports(
    tree: ast.Module,
    module: str,
    report_file: str,
    seen: set[tuple[str, str, str]],
) -> list[Violation]:
    """Return violations for importlib.import_module('tests....') calls in src/."""
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_import_module = (
            (isinstance(func, ast.Attribute) and func.attr == "import_module")
            or (isinstance(func, ast.Name) and func.id == "import_module")
        )
        if not is_import_module or not node.args:
            continue
        first_arg = node.args[0]
        if not (isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str)):
            continue
        target = first_arg.value
        if target.startswith("tests.") or target.startswith("tests/"):
            reason = "dynamic-import targeting tests namespace"
            key = (module, target, reason)
            if key not in seen:
                seen.add(key)
                violations.append(
                    Violation(
                        module=module,
                        file=report_file,
                        symbol=target,
                        line=node.lineno,
                        reason=reason,
                    )
                )
    return violations


def _find_banned_import_reexports(
    tree: ast.Module,
    module: str,
    report_file: str,
    banned_exports: set[str],
    seen: set[tuple[str, str, str]],
) -> list[Violation]:
    """Return violations for import-level re-exports of banned symbols."""
    violations: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            imported_name = alias.name
            if imported_name not in banned_exports:
                continue
            reason = "import-level banned symbol re-export"
            key = (module, imported_name, reason)
            if key not in seen:
                seen.add(key)
                violations.append(
                    Violation(
                        module=module,
                        file=report_file,
                        symbol=imported_name,
                        line=node.lineno,
                        reason=reason,
                    )
                )
    return violations


def _symbol_reasons(
    symbol: str,
    export_line: int,
    banned_exports: set[str],
    public_defs: dict[str, ast.AST],
) -> list[tuple[str, int]]:
    """Collect violation reasons for a single exported symbol."""
    reasons: list[tuple[str, int]] = []
    if symbol in banned_exports:
        reasons.append(("explicitly banned export for this module", export_line))
    if _looks_like_test_helper_name(symbol):
        reasons.append(("symbol name indicates test-only helper intent", export_line))
    node = public_defs.get(symbol)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        docstring = ast.get_docstring(node) or ""
        if _contains_test_helper_phrase(docstring) and _contains_helper_intent(docstring):
            reasons.append(("docstring labels symbol as testing helper", node.lineno))
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _is_private_delegate(node)
            and _contains_test_helper_phrase(docstring)
        ):
            label = "thin public wrapper around private helper for tests"
            reasons.append((label, node.lineno))
    return reasons


def _find_banned_exported_symbols(
    tree: ast.Module,
    module: str,
    report_file: str,
    banned_exports: set[str],
    seen: set[tuple[str, str, str]],
) -> list[Violation]:
    """Return violations for exported symbols that look like test helpers."""
    exports = _extract_exports(tree)
    if not exports:
        return []
    public_defs = _collect_public_defs(tree)
    violations: list[Violation] = []
    for symbol, export_line in exports.items():
        reasons = _symbol_reasons(symbol, export_line, banned_exports, public_defs)
        for reason, reason_line in reasons:
            key = (module, symbol, reason)
            if key not in seen:
                seen.add(key)
                violations.append(
                    Violation(
                        module=module,
                        file=report_file,
                        symbol=symbol,
                        line=reason_line,
                        reason=reason,
                    )
                )
    return violations


def scan_package(package_root: Path) -> list[Violation]:
    """Scan package modules for banned test-helper exports."""
    violations: list[Violation] = []
    seen: set[tuple[str, str, str]] = set()

    for path in sorted(package_root.rglob("*.py")):
        module = _module_name(path, package_root)
        if _should_ignore_module(module):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            raise RuntimeError(f"Unable to parse {path}: {exc}") from exc

        banned_exports = BANNED_EXPORTS_BY_MODULE.get(module, set())
        report_file = _relative_path(path, package_root)

        violations.extend(_find_dynamic_test_imports(tree, module, report_file, seen))
        if banned_exports:
            violations.extend(
                _find_banned_import_reexports(tree, module, report_file, banned_exports, seen)
            )
        violations.extend(
            _find_banned_exported_symbols(tree, module, report_file, banned_exports, seen)
        )

    return sorted(
        violations,
        key=lambda item: (item.file, item.line, item.symbol, item.reason),
    )


def write_report(report_path: Path, package_root: Path, violations: list[Violation]) -> None:
    """Write a deterministic JSON report."""
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "package_root": str(package_root).replace("\\", "/"),
        "total_violations": len(violations),
        "violations": [violation.to_record() for violation in violations],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the export leakage guard."""
    parser = argparse.ArgumentParser(
        description="Block production __all__ exports that expose test-helper wrappers.",
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
        default=Path("reports/anti-pattern-analysis/test_helper_wrapper_report.json"),
        help="JSON report path.",
    )
    args = parser.parse_args(argv)

    try:
        package_root = _resolve_package_root(args.root)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    try:
        violations = scan_package(package_root)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    write_report(args.report, package_root, violations)

    if violations:
        print("Found prohibited test-helper exports:")
        for violation in violations:
            print(
                f"- {violation.file}:{violation.line} "
                f"[{violation.symbol}] {violation.reason}"
            )
        print(f"Report written to {args.report}")
        return 1

    print("No prohibited test-helper exports detected.")
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
