"""Check test marker hygiene per ADR-030 Phase 3.

Auto-infers expected test layer (unit, integration) from directory paths and
verifies that explicit markers (slow, viz) are present where required.

Usage
-----
    python scripts/quality/check_marker_hygiene.py               # report mode
    python scripts/quality/check_marker_hygiene.py --check        # CI gate mode
    python scripts/quality/check_marker_hygiene.py --rebaseline   # regenerate baseline

Exit codes (--check):
    0  no new violations versus baseline
    1  new violations detected (or baseline missing)
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

BASELINE_RATIONALE = "Existing debt accepted during ADR-030 Phase 3 rollout."

# ---------------------------------------------------------------------------
# Directory → expected layer mapping
# ---------------------------------------------------------------------------
LAYER_BY_DIR: dict[str, str] = {
    "unit": "unit",
    "integration": "integration",
}

# Markers that should be explicit (not auto-inferred)
EXPLICIT_MARKERS = {"slow", "viz", "viz_render"}

# Imports that signal a test needs the viz marker
VIZ_IMPORT_SIGNALS = {
    "matplotlib",
    "matplotlib.pyplot",
}


@dataclass(frozen=True)
class Finding:
    """A single marker hygiene finding."""

    path: Path
    stable_file: str
    line: int
    pattern: str
    detail: str
    file_hash: str

    @property
    def finding_id(self) -> str:
        """Return deterministic fingerprint for this finding."""
        payload = f"{self.stable_file}|{self.line}|{self.pattern}|{self.detail}"
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_record(self, root: Path | None = None) -> dict[str, object]:
        """Return a JSON-safe report record."""
        return {
            "id": self.finding_id,
            "file": self.stable_file,
            "line": self.line,
            "pattern": self.pattern,
            "detail": self.detail,
            "file_hash": self.file_hash,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_relative(path: Path, root: Path | None = None) -> str:
    try:
        if root is not None:
            return str(path.relative_to(root)).replace("\\", "/")
        return str(path.relative_to(Path.cwd())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _hash_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _infer_layer(path: Path, tests_root: Path) -> str | None:
    """Infer the expected test layer from the directory structure."""
    try:
        relative = path.relative_to(tests_root)
    except ValueError:
        return None
    parts = relative.parts
    for part in parts:
        if part in LAYER_BY_DIR:
            return LAYER_BY_DIR[part]
    return None


def _collect_markers(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Extract pytest marker names from decorator list."""
    markers: set[str] = set()
    for dec in node.decorator_list:
        # @pytest.mark.slow
        if isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Attribute):
            if (
                isinstance(dec.value.value, ast.Name)
                and dec.value.value.id == "pytest"
                and dec.value.attr == "mark"
            ):
                markers.add(dec.attr)
        # @pytest.mark.slow(...)  (call form)
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
            inner = dec.func
            if isinstance(inner.value, ast.Attribute):
                if (
                    isinstance(inner.value.value, ast.Name)
                    and inner.value.value.id == "pytest"
                    and inner.value.attr == "mark"
                ):
                    markers.add(inner.attr)
    return markers


def _collect_module_markers(tree: ast.Module) -> set[str]:
    """Collect module-level pytestmark assignments."""
    markers: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "pytestmark":
                    # pytestmark = pytest.mark.viz
                    if isinstance(node.value, ast.Attribute):
                        markers.add(node.value.attr)
                    # pytestmark = [pytest.mark.viz, pytest.mark.slow]
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Attribute):
                                markers.add(elt.attr)
    return markers


def _has_viz_imports(tree: ast.Module) -> bool:
    """Check if module imports matplotlib or related viz packages."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in VIZ_IMPORT_SIGNALS or alias.name.startswith("matplotlib"):
                    return True
        if isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module in VIZ_IMPORT_SIGNALS or node.module.startswith("matplotlib")
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_marker_hygiene(tests_root: Path) -> list[Finding]:
    """Walk the test tree and collect marker hygiene findings."""
    findings: list[Finding] = []
    tests_root = tests_root.resolve()

    for path in sorted(tests_root.rglob("test_*.py")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        file_hash = _hash_file(resolved)
        stable_file = _to_relative(resolved, tests_root)
        try:
            tree = ast.parse(resolved.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        expected_layer = _infer_layer(resolved, tests_root)
        module_markers = _collect_module_markers(tree)
        has_viz = _has_viz_imports(tree)

        # Check 1: integration tests under unit/ directory (layer mismatch)
        if expected_layer == "unit" and "integration" in module_markers:
            findings.append(
                Finding(
                    resolved,
                    stable_file,
                    1,
                    "layer-marker-mismatch",
                    "File in unit/ directory but has pytestmark = integration",
                    file_hash,
                )
            )

        # Check 2: viz imports without viz marker
        if has_viz and "viz" not in module_markers and "viz_render" not in module_markers:
            # Check if file is already in a viz/ subdirectory (which might auto-infer)
            relative_parts = resolved.relative_to(tests_root).parts
            if "viz" not in relative_parts:
                findings.append(
                    Finding(
                        resolved,
                        stable_file,
                        1,
                        "missing-viz-marker",
                        "File imports matplotlib but lacks @pytest.mark.viz or pytestmark",
                        file_hash,
                    )
                )

        # Check 3: per-function marker checks
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue

            func_markers = _collect_markers(node)
            all_markers = module_markers | func_markers

            # Integration test not in integration/ and missing marker
            if expected_layer == "unit" and "integration" in func_markers:
                findings.append(
                    Finding(
                        resolved,
                        stable_file,
                        node.lineno,
                        "layer-marker-mismatch",
                        f"Test {node.name} in unit/ dir marked @pytest.mark.integration",
                        file_hash,
                    )
                )

            # Tests in integration/ should have integration marker
            # (module-level or function-level)
            if expected_layer == "integration" and "integration" not in all_markers:
                findings.append(
                    Finding(
                        resolved,
                        stable_file,
                        node.lineno,
                        "missing-integration-marker",
                        f"Test {node.name} in integration/ dir lacks "
                        "@pytest.mark.integration (module or function level)",
                        file_hash,
                    )
                )

    return findings


# ---------------------------------------------------------------------------
# Baseline + reporting
# ---------------------------------------------------------------------------

def load_baseline(path: Path) -> dict[str, dict[str, object]]:
    """Load baseline entries indexed by finding id."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    entries = payload.get("entries", [])
    indexed: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        fid = entry.get("id")
        if isinstance(fid, str):
            indexed[fid] = entry
    return indexed


def find_new_violations(
    findings: list[Finding], baseline: dict[str, dict[str, object]]
) -> list[Finding]:
    """Return findings not present in the baseline."""
    return [f for f in findings if f.finding_id not in baseline]


def write_baseline(path: Path, findings: list[Finding], root: Path) -> None:
    """Write committed baseline file."""
    entries = []
    for f in findings:
        record = f.to_record(root=root)
        record["rationale"] = BASELINE_RATIONALE
        entries.append(record)
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "tests_dir": str(root).replace("\\", "/"),
        "entries": sorted(
            entries, key=lambda r: (str(r["file"]), int(r["line"]), str(r["pattern"]))
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_json_report(
    findings: list[Finding],
    output: Path,
    root: Path,
    baseline_path: Path,
    new_ids: set[str] | None = None,
) -> None:
    """Write rich JSON report for CI artifacts."""
    records = [f.to_record(root=root) for f in findings]
    for rec in records:
        rec["is_new"] = bool(new_ids and rec["id"] in new_ids)
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "tests_dir": str(root).replace("\\", "/"),
        "baseline_path": str(baseline_path).replace("\\", "/"),
        "total_findings": len(records),
        "new_violations": len(new_ids or set()),
        "findings": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def print_rebaseline_diff(
    old: dict[str, dict[str, object]], findings: list[Finding], root: Path
) -> None:
    """Print diff summary for rebaseline review."""
    current = {f.finding_id: f.to_record(root=root) for f in findings}
    old_ids = set(old.keys())
    cur_ids = set(current.keys())
    added = sorted(cur_ids - old_ids)
    removed = sorted(old_ids - cur_ids)
    print("Rebaseline diff summary:")
    print(f"  Added: {len(added)}")
    print(f"  Removed: {len(removed)}")
    for fid in added[:10]:
        r = current[fid]
        print(f"  + {r['file']}:{r['line']} [{r['pattern']}]")
    for fid in removed[:10]:
        r = old[fid]
        print(f"  - {r.get('file')}:{r.get('line')} [{r.get('pattern')}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """Run marker hygiene analysis."""
    parser = argparse.ArgumentParser(description="Check test marker hygiene (ADR-030 Phase 3).")
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=Path("tests"),
        help="Path to the tests directory to scan.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/marker-hygiene/marker_hygiene_report.json"),
        help="JSON path for full report output.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(".github/marker-hygiene-baseline.json"),
        help="Baseline file for --check / --rebaseline.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if new violations exist versus baseline.",
    )
    parser.add_argument(
        "--rebaseline",
        action="store_true",
        help="Regenerate baseline from current findings.",
    )
    args = parser.parse_args()

    tests_dir = args.tests_dir.resolve()
    findings = scan_marker_hygiene(tests_dir)
    baseline = load_baseline(args.baseline)
    new_violations = find_new_violations(findings, baseline)
    new_ids = {f.finding_id for f in new_violations}

    write_json_report(findings, args.report, tests_dir, args.baseline, new_ids)

    if args.rebaseline:
        print_rebaseline_diff(baseline, findings, tests_dir)
        write_baseline(args.baseline, findings, tests_dir)
        print(f"Baseline written to {args.baseline}")

    print(f"Found {len(findings)} marker hygiene findings.")
    print(f"JSON report written to {args.report}")

    if args.check:
        if not args.baseline.exists():
            print(f"ERROR: baseline not found at {args.baseline}")
            return 1
        if new_violations:
            print(f"New violations versus baseline: {len(new_violations)}")
            for f in new_violations[:25]:
                file_path = _to_relative(f.path, tests_dir)
                print(f"  {file_path}:{f.line} [{f.pattern}] {f.detail}")
            return 1
        print("No new marker hygiene violations versus baseline.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
