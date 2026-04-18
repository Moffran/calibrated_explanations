"""Detect Standard-001 nomenclature violations and generate an inventory report."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

APPROVED_COMPATIBILITY_BRIDGES: dict[str, set[str]] = {
    "calibrated_explanations/calibration/venn_abers.py": {
        "__is_multiclass",
        "__predict_proba_with_difficulty",
    },
    "calibrated_explanations/calibration/state.py": {
        "__X_cal",
    },
    "calibrated_explanations/core/calibrated_explainer.py": {
        "__initialized",
        "__fast",
        "__noise_type",
        "__scale_factor",
        "__severity",
        "__initialize_interval_learner_for_fast_explainer",
    },
    "calibrated_explanations/explanations/explanation.py": {
        "__append_rule",
        "__extracted_non_conjunctive_rules",
        "__filter_rules",
        "__is_counter_explanation",
        "__is_semi_explanation",
        "__is_super_explanation",
        "__pareto_filter_rules",
        "__pareto_rule_indexes",
        "__set_up_result",
    },
    "calibrated_explanations/explanations/explanations.py": {
        "__convert_to_alternative_explanations",
    },
}

APPROVED_TRANSITIONAL_SHIMS: dict[str, set[str]] = {
    "calibrated_explanations/plotting.py": {
        "__plot_proba_triangle",
        "__require_matplotlib",
        "__setup_plot_style",
    }
}

APPROVED_UTILITY_IMPORT_BRIDGES: set[str] = {
    "calibrated_explanations/calibration/interval_learner.py",
    "calibrated_explanations/core/__init__.py",
    "calibrated_explanations/core/explain/_computation.py",
    "calibrated_explanations/explanations/explanation.py",
    "calibrated_explanations/explanations/explanations.py",
    "calibrated_explanations/explanations/legacy_conjunctions.py",
    "calibrated_explanations/utils/__init__.py",
    "calibrated_explanations/utils/discretizers.py",
}

APPROVED_SHIM_SURFACES: dict[str, dict[str, str]] = {
    "calibrated_explanations/serialization.py": {
        "validate_payload": "schema.validate_payload_compat_wrapper",
    },
    "calibrated_explanations/viz/builders.py": {
        "legacy_get_fill_color": "legacy_color_api_alias",
    },
}

_MANGLED_PATTERN = re.compile(r"^_[A-Za-z][A-Za-z0-9]*__([_A-Za-z][A-Za-z0-9_]*)$")


@dataclass(frozen=True)
class ViolationRecord:
    """A Standard-001 scan result row."""

    file: str
    line: int
    symbol: str
    violation_kind: str
    allowed_reason: str
    workstream: str

    @property
    def is_allowed(self) -> bool:
        return self.allowed_reason != ""

    def to_dict(self) -> dict[str, object]:
        """Return a report-compatible dictionary."""
        return {
            "file": self.file,
            "line": self.line,
            "symbol": self.symbol,
            "violation_kind": self.violation_kind,
            "allowed_reason": self.allowed_reason,
            "workstream": self.workstream,
            "is_allowed": self.is_allowed,
        }


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


def _relative(path: Path, package_root: Path) -> str:
    return str(path.resolve().relative_to(package_root.parent.resolve())).replace("\\", "/")


def _report_package_root(package_root: Path) -> str:
    base = package_root.parent.parent if package_root.parent.name == "src" else package_root.parent
    return str(package_root.resolve().relative_to(base.resolve())).replace("\\", "/")


def _is_non_protocol_dunder(symbol: str) -> bool:
    return symbol.startswith("__") and not symbol.endswith("__")


def _normalize_dunder_symbol(symbol: str) -> str:
    if _is_non_protocol_dunder(symbol):
        return symbol
    match = _MANGLED_PATTERN.match(symbol)
    if match:
        return f"__{match.group(1)}"
    return symbol


def _classify_dunder_violation(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "non_legacy_dunder_definition"
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return "non_legacy_dunder_mutation"
    if isinstance(node, (ast.Attribute, ast.Name)) and isinstance(node.ctx, ast.Store):
        return "non_legacy_dunder_mutation"
    return "non_legacy_dunder_access"


def _allowed_for_dunder(rel: str, symbol: str) -> tuple[str, str, str]:
    if "/legacy/" in rel:
        return (
            "non_legacy_dunder_access",
            "legacy_namespace_exception",
            "legacy_namespace",
        )
    if symbol in APPROVED_TRANSITIONAL_SHIMS.get(rel, set()):
        return (
            "non_legacy_transitional_shim",
            "approved_transitional_shim_remove_by_v0.11.3",
            "shim_confinement",
        )
    if symbol in APPROVED_COMPATIBILITY_BRIDGES.get(rel, set()):
        return (
            "compatibility_bridge",
            "approved_compatibility_bridge_remove_by_v0.11.3",
            "compatibility_bridge",
        )
    return ("", "", "")


def _iter_dunder_records(tree: ast.AST, rel: str) -> list[ViolationRecord]:
    records: list[ViolationRecord] = []
    seen: set[tuple[str, int, str, str]] = set()
    for node in ast.walk(tree):
        symbol = ""
        if isinstance(node, ast.Attribute):
            symbol = node.attr
        elif isinstance(node, ast.Name):
            symbol = node.id
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol = node.name
        else:
            continue

        symbol = _normalize_dunder_symbol(symbol)
        if not _is_non_protocol_dunder(symbol):
            continue

        line = getattr(node, "lineno", 1)
        violation_kind = _classify_dunder_violation(node)
        override_kind, allowed_reason, workstream = _allowed_for_dunder(rel, symbol)
        if override_kind:
            violation_kind = override_kind

        key = (rel, line, symbol, violation_kind)
        if key in seen:
            continue
        seen.add(key)

        records.append(
            ViolationRecord(
                file=rel,
                line=line,
                symbol=symbol,
                violation_kind=violation_kind,
                allowed_reason=allowed_reason,
                workstream=workstream,
            )
        )
    return records


def _iter_import_bridge_records(tree: ast.AST, rel: str) -> list[ViolationRecord]:
    records: list[ViolationRecord] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if not module.endswith("utils.helper") and module != ".helper":
            continue
        allowed = rel in APPROVED_UTILITY_IMPORT_BRIDGES
        records.append(
            ViolationRecord(
                file=rel,
                line=getattr(node, "lineno", 1),
                symbol=module,
                violation_kind="utility_import_bridge",
                allowed_reason="approved_helper_split_bridge"
                if allowed
                else "",
                workstream="utility_split" if allowed else "",
            )
        )
    return records


def _iter_shim_surface_records(tree: ast.AST, rel: str) -> list[ViolationRecord]:
    if rel not in APPROVED_SHIM_SURFACES:
        return []
    remaining = dict(APPROVED_SHIM_SURFACES[rel])
    records: list[ViolationRecord] = []
    for node in ast.walk(tree):
        candidate_name = ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            candidate_name = node.name
        elif isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Name)
            ):
                candidate_name = node.targets[0].id
        if candidate_name not in remaining:
            continue
        thin = True
        if rel.endswith("serialization.py") and candidate_name == "validate_payload":
            thin = _is_thin_serialization_wrapper(node)
        if rel.endswith("viz/builders.py") and candidate_name == "legacy_get_fill_color":
            thin = _is_thin_builders_alias(node)
        records.append(
            ViolationRecord(
                file=rel,
                line=getattr(node, "lineno", 1),
                symbol=f"{candidate_name}:{remaining.pop(candidate_name)}",
                violation_kind="non_legacy_transitional_shim",
                allowed_reason="approved_transitional_shim_remove_by_v0.11.3" if thin else "",
                workstream="shim_confinement" if thin else "",
            )
        )
    return records


def _is_thin_serialization_wrapper(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    # Allow optional docstring expr, then require a single return delegating to
    # _schema_validate_payload(obj).
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    call = body[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return False
    return call.func.id == "_schema_validate_payload"


def _is_thin_builders_alias(node: ast.AST) -> bool:
    if not isinstance(node, ast.Assign):
        return False
    if len(node.targets) != 1:
        return False
    target = node.targets[0]
    return (
        isinstance(target, ast.Name)
        and target.id == "legacy_get_fill_color"
        and isinstance(node.value, ast.Name)
        and node.value.id == "_legacy_get_fill_color"
    )


def scan_package(package_root: Path) -> list[ViolationRecord]:
    """Scan source files for Standard-001 violations and approved exceptions."""
    records: list[ViolationRecord] = []
    for path in sorted(package_root.rglob("*.py")):
        rel = _relative(path, package_root)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        records.extend(_iter_dunder_records(tree, rel))
        records.extend(_iter_import_bridge_records(tree, rel))
        records.extend(_iter_shim_surface_records(tree, rel))

    return sorted(
        records,
        key=lambda rec: (rec.file, rec.line, rec.violation_kind, rec.symbol),
    )


def write_report(report_path: Path, package_root: Path, records: list[ViolationRecord]) -> None:
    """Write a deterministic JSON report."""
    payload = {
        "version": 1,
        "package_root": _report_package_root(package_root),
        "total_records": len(records),
        "total_violations": sum(1 for rec in records if not rec.is_allowed),
        "records": [record.to_dict() for record in records],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the nomenclature checker."""
    parser = argparse.ArgumentParser(
        description="Detect non-legacy dunder usage and utility/shim bridges for STD-001.",
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
        default=Path("reports/nomenclature_violation_inventory.json"),
        help="Output report path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail on disallowed violations.",
    )
    args = parser.parse_args(argv)

    try:
        package_root = _resolve_package_root(args.root)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    records = scan_package(package_root)
    write_report(args.report, package_root, records)

    violations = [rec for rec in records if not rec.is_allowed]
    if violations:
        print("STD-001 nomenclature violations detected:")
        for rec in violations:
            print(f"- {rec.file}:{rec.line} [{rec.violation_kind}] {rec.symbol}")
        print(f"Report written to {args.report}")
        return 1 if args.check else 0

    print("STD-001 nomenclature check passed (no disallowed violations).")
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
