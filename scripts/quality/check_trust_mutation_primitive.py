"""Enforce trust-state mutations route through plugins._trust primitives."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MUTATION_METHODS = {
    "add",
    "discard",
    "clear",
    "remove",
    "update",
    "difference_update",
}


@dataclass(frozen=True)
class MutationRecord:
    """A scanned trust-mutation record."""

    file: str
    line: int
    symbol: str
    mutation_kind: str
    is_allowed: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "file": self.file,
            "line": self.line,
            "symbol": self.symbol,
            "mutation_kind": self.mutation_kind,
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
    """Return a repo-relative package root for report payloads."""
    base = package_root.parent.parent if package_root.parent.name == "src" else package_root.parent
    return str(package_root.resolve().relative_to(base.resolve())).replace("\\", "/")


def _is_allowed(path: str) -> bool:
    return path.endswith("plugins/_trust.py")


def _record_trusted_set_mutation(
    node: ast.Call,
    *,
    rel: str,
) -> MutationRecord | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name):
        return None
    if not func.value.id.startswith("_TRUSTED_"):
        return None
    if func.attr not in MUTATION_METHODS:
        return None
    return MutationRecord(
        file=rel,
        line=getattr(node, "lineno", 1),
        symbol=func.value.id,
        mutation_kind=f"trusted_set.{func.attr}",
        is_allowed=_is_allowed(rel),
    )


def _record_trusted_attr_write(
    node: ast.Assign | ast.AnnAssign,
    *,
    rel: str,
) -> list[MutationRecord]:
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    else:
        targets = [node.target]

    records: list[MutationRecord] = []
    for target in targets:
        if not isinstance(target, ast.Attribute):
            continue
        if target.attr != "trusted":
            continue
        records.append(
            MutationRecord(
                file=rel,
                line=getattr(node, "lineno", 1),
                symbol=ast.unparse(target) if hasattr(ast, "unparse") else "trusted",
                mutation_kind="attribute_write.trusted",
                is_allowed=_is_allowed(rel),
            )
        )
    return records


def _record_primitive_call(node: ast.Call, *, rel: str) -> MutationRecord | None:
    func = node.func
    if not isinstance(func, ast.Name):
        return None
    if func.id not in {"update_trusted_identifier", "clear_trusted_identifiers"}:
        return None
    return MutationRecord(
        file=rel,
        line=getattr(node, "lineno", 1),
        symbol=func.id,
        mutation_kind=f"primitive_call.{func.id}",
        is_allowed=True,
    )


def scan_package(package_root: Path) -> list[MutationRecord]:
    """Scan package for trust-state mutation sites."""
    records: list[MutationRecord] = []
    for path in sorted(package_root.rglob("*.py")):
        rel = _relative(path, package_root)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                record = _record_trusted_set_mutation(node, rel=rel)
                if record is not None:
                    records.append(record)
                    continue
                primitive_record = _record_primitive_call(node, rel=rel)
                if primitive_record is not None:
                    records.append(primitive_record)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                records.extend(_record_trusted_attr_write(node, rel=rel))

    return sorted(records, key=lambda r: (r.file, r.line, r.symbol, r.mutation_kind))


def write_report(report_path: Path, package_root: Path, records: list[MutationRecord]) -> None:
    """Write deterministic inventory report."""
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "package_root": _report_package_root(package_root),
        "total_records": len(records),
        "total_violations": sum(1 for record in records if not record.is_allowed),
        "records": [record.to_dict() for record in records],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    """Run trust-mutation inventory and enforcement."""
    parser = argparse.ArgumentParser(
        description="Inventory trust-state mutation sites and enforce use of plugins._trust.",
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
        default=Path("reports/trust_mutation_inventory.json"),
        help="JSON inventory report output path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when disallowed trust mutations are detected.",
    )
    args = parser.parse_args(argv)

    try:
        package_root = _resolve_package_root(args.root)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    records = scan_package(package_root)
    write_report(args.report, package_root, records)

    violations = [record for record in records if not record.is_allowed]
    if violations:
        print("Disallowed trust mutation sites detected:")
        for record in violations:
            print(f"- {record.file}:{record.line} [{record.symbol}] {record.mutation_kind}")
        print(f"Report written to {args.report}")
        return 1 if args.check else 0

    print("Trust mutation primitive check passed (no disallowed mutations).")
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
