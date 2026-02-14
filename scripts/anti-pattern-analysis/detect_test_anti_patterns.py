"""Test anti-pattern detector with ADR-030 baseline enforcement support."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


RNG_METHODS = {
    "random",
    "randint",
    "randrange",
    "choice",
    "choices",
    "shuffle",
    "sample",
    "uniform",
    "triangular",
    "betavariate",
    "expovariate",
    "gammavariate",
    "gauss",
    "lognormvariate",
    "normalvariate",
    "vonmisesvariate",
    "paretovariate",
    "weibullvariate",
    "getrandbits",
}

TIME_CALLS = {
    ("time", "time"),
    ("time", "sleep"),
    ("time", "monotonic"),
    ("time", "perf_counter"),
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("date", "today"),
}

NETWORK_CALLS = {
    ("requests", "get"),
    ("requests", "post"),
    ("requests", "put"),
    ("requests", "delete"),
    ("requests", "patch"),
    ("requests", "request"),
    ("httpx", "get"),
    ("httpx", "post"),
    ("httpx", "put"),
    ("httpx", "delete"),
    ("httpx", "patch"),
    ("httpx", "request"),
    ("urllib.request", "urlopen"),
    ("socket", "socket"),
    ("socket", "create_connection"),
}

BASELINE_RATIONALE = "Existing debt accepted during incremental ADR-030 rollout."


@dataclass(frozen=True)
class Finding:
    """Represents a single anti-pattern finding."""

    path: Path
    line: int
    pattern: str
    snippet: str
    file_hash: str

    @property
    def finding_id(self) -> str:
        """Return deterministic fingerprint for this finding."""
        payload = f"{self.path.as_posix()}|{self.line}|{self.pattern}|{self.snippet}"
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_record(self, root: Path | None = None) -> dict[str, object]:
        """Return a JSON-safe report record."""
        relative_file = _to_relative(self.path, root)
        return {
            "id": self.finding_id,
            "file": relative_file,
            "line": self.line,
            "pattern": self.pattern,
            "snippet": self.snippet,
            "file_hash": self.file_hash,
        }


def _safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _call_name(node: ast.Call) -> tuple[str, str]:
    func = node.func
    if isinstance(func, ast.Name):
        return ("", func.id)
    if isinstance(func, ast.Attribute):
        return (_safe_unparse(func.value), func.attr)
    return ("", "")


def _is_test_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return node.name.startswith("test_")


class AssertionProbe(ast.NodeVisitor):
    """Detect whether a test body contains meaningful assertions."""

    def __init__(self) -> None:
        self.has_assertion = False

    def visit_Assert(self, node: ast.Assert) -> None:  # noqa: N802
        self.has_assertion = True
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        if any(_is_pytest_assertion_context(item.context_expr) for item in node.items):
            self.has_assertion = True
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:  # noqa: N802
        if any(_is_pytest_assertion_context(item.context_expr) for item in node.items):
            self.has_assertion = True
        self.generic_visit(node)


class FunctionProbe(ast.NodeVisitor):
    """Collect behavior hints for determinism checks in a test function."""

    def __init__(
        self,
        random_modules: set[str],
        random_functions: dict[str, str],
        time_aliases: set[str],
        datetime_aliases: set[str],
        date_aliases: set[str],
        requests_aliases: set[str],
        httpx_aliases: set[str],
        urllib_request_aliases: set[str],
        socket_aliases: set[str],
    ) -> None:
        self.random_modules = random_modules
        self.random_functions = random_functions
        self.time_aliases = time_aliases
        self.datetime_aliases = datetime_aliases
        self.date_aliases = date_aliases
        self.requests_aliases = requests_aliases
        self.httpx_aliases = httpx_aliases
        self.urllib_request_aliases = urllib_request_aliases
        self.socket_aliases = socket_aliases
        self.random_calls: list[ast.Call] = []
        self.seed_calls: list[ast.Call] = []
        self.time_or_network_calls: list[ast.Call] = []
        self.patch_calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        base, attr = _call_name(node)

        if self._is_random_call(base, attr):
            self.random_calls.append(node)
        if self._is_seed_call(base, attr):
            self.seed_calls.append(node)
        if self._is_time_call(base, attr) or self._is_network_call(base, attr):
            self.time_or_network_calls.append(node)
        if self._is_patch_call(base, attr):
            self.patch_calls.append(node)
        self.generic_visit(node)

    def _is_random_call(self, base: str, attr: str) -> bool:
        if base in self.random_modules and attr in RNG_METHODS:
            return True
        if not base and attr in self.random_functions and self.random_functions[attr] in RNG_METHODS:
            return True
        return False

    def _is_seed_call(self, base: str, attr: str) -> bool:
        if base in self.random_modules and attr == "seed":
            return True
        if not base and attr in self.random_functions and self.random_functions[attr] == "seed":
            return True
        return False

    def _is_time_call(self, base: str, attr: str) -> bool:
        if (base, attr) in TIME_CALLS:
            return True
        if base in self.time_aliases and attr in {"time", "sleep", "monotonic", "perf_counter"}:
            return True
        if base in self.datetime_aliases and attr in {"now", "utcnow"}:
            return True
        if base in self.date_aliases and attr == "today":
            return True
        return False

    def _is_network_call(self, base: str, attr: str) -> bool:
        if (base, attr) in NETWORK_CALLS:
            return True
        if base in self.requests_aliases and attr in {"get", "post", "put", "delete", "patch", "request"}:
            return True
        if base in self.httpx_aliases and attr in {"get", "post", "put", "delete", "patch", "request"}:
            return True
        if base in self.urllib_request_aliases and attr == "urlopen":
            return True
        if base in self.socket_aliases and attr in {"socket", "create_connection"}:
            return True
        return False

    def _is_patch_call(self, base: str, attr: str) -> bool:
        if attr == "setattr" and base in {"monkeypatch", "pytest.MonkeyPatch"}:
            return True
        if attr == "patch":
            return True
        if base == "mocker" and attr in {"patch", "spy"}:
            return True
        return False


class AntiPatternVisitor(ast.NodeVisitor):
    """AST visitor that records anti-pattern findings."""

    def __init__(self, path: Path, file_hash: str) -> None:
        self.path = path
        self.file_hash = file_hash
        self.findings: list[Finding] = []
        self.lines = path.read_text(encoding="utf-8").splitlines()
        self.random_modules: set[str] = {"random"}
        self.random_functions: dict[str, str] = {}
        self.time_aliases: set[str] = {"time"}
        self.datetime_aliases: set[str] = {"datetime"}
        self.date_aliases: set[str] = {"date"}
        self.requests_aliases: set[str] = {"requests"}
        self.httpx_aliases: set[str] = {"httpx"}
        self.urllib_request_aliases: set[str] = {"urllib.request"}
        self.socket_aliases: set[str] = {"socket"}

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            name = alias.name
            as_name = alias.asname or alias.name
            if name == "random":
                self.random_modules.add(as_name)
            if name == "time":
                self.time_aliases.add(as_name)
            if name == "requests":
                self.requests_aliases.add(as_name)
            if name == "httpx":
                self.httpx_aliases.add(as_name)
            if name == "socket":
                self.socket_aliases.add(as_name)
            if name == "urllib.request":
                self.urllib_request_aliases.add(as_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        for alias in node.names:
            as_name = alias.asname or alias.name
            if module == "random":
                self.random_functions[as_name] = alias.name
            if module == "time":
                self.time_aliases.add(as_name)
            if module == "datetime":
                if alias.name == "datetime":
                    self.datetime_aliases.add(as_name)
                if alias.name == "date":
                    self.date_aliases.add(as_name)
            if module == "requests":
                self.requests_aliases.add(as_name)
            if module == "httpx":
                self.httpx_aliases.add(as_name)
            if module == "socket":
                self.socket_aliases.add(as_name)
            if module == "urllib.request":
                self.urllib_request_aliases.add(as_name)
        self.generic_visit(node)

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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._check_test_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._check_test_function(node)
        self.generic_visit(node)

    def _check_test_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if not _is_test_function(node):
            return

        assertion_probe = AssertionProbe()
        for stmt in node.body:
            assertion_probe.visit(stmt)
        if not assertion_probe.has_assertion:
            self._record(node, "test without assertion")

        behavior_probe = FunctionProbe(
            random_modules=self.random_modules,
            random_functions=self.random_functions,
            time_aliases=self.time_aliases,
            datetime_aliases=self.datetime_aliases,
            date_aliases=self.date_aliases,
            requests_aliases=self.requests_aliases,
            httpx_aliases=self.httpx_aliases,
            urllib_request_aliases=self.urllib_request_aliases,
            socket_aliases=self.socket_aliases,
        )
        for stmt in node.body:
            behavior_probe.visit(stmt)

        if behavior_probe.random_calls and not behavior_probe.seed_calls:
            self._record(behavior_probe.random_calls[0], "random usage without explicit seeding")

        if behavior_probe.time_or_network_calls and not behavior_probe.patch_calls:
            self._record(
                behavior_probe.time_or_network_calls[0],
                "time/network usage without patching",
            )

    def _record(self, node: ast.AST, pattern: str) -> None:
        lineno = getattr(node, "lineno", 0)
        snippet = self._format_snippet(lineno)
        self.findings.append(Finding(self.path, lineno, pattern, snippet, self.file_hash))

    def _format_snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return textwrap.shorten(self.lines[lineno - 1].strip(), width=160)
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
        if isinstance(node, ast.Attribute) and node.attr in {"raises", "warns"}:
            return True
        if isinstance(node, ast.Name) and node.id in {"raises", "warns"}:
            return True
        return False


def _hash_file(path: Path) -> str:
    content = path.read_bytes()
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _to_relative(path: Path, root: Path | None = None) -> str:
    try:
        if root is not None:
            return str(path.relative_to(root)).replace("\\", "/")
        return str(path.relative_to(Path.cwd())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


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
        file_hash = _hash_file(resolved)
        visitor = AntiPatternVisitor(resolved, file_hash=file_hash)
        try:
            visitor.visit(ast.parse(resolved.read_text(encoding="utf-8")))
        except SyntaxError as exc:
            print(f"Unable to parse {path}: {exc}")
            continue
        findings.extend(visitor.findings)
    return findings


def write_csv_report(findings: Iterable[Finding], output_path: Path, root: Path | None = None) -> None:
    """Write findings to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "file", "line", "pattern", "snippet", "file_hash"])
        for finding in findings:
            row = finding.to_record(root=root)
            writer.writerow(
                [
                    row["id"],
                    row["file"],
                    row["line"],
                    row["pattern"],
                    row["snippet"],
                    row["file_hash"],
                ]
            )


def write_json_report(
    findings: Iterable[Finding],
    output_path: Path,
    root: Path,
    baseline_path: Path,
    new_ids: set[str] | None = None,
) -> None:
    """Write rich JSON report for CI artifacts and local triage."""
    records = [finding.to_record(root=root) for finding in findings]
    for record in records:
        record["is_new"] = bool(new_ids and record["id"] in new_ids)

    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "tests_dir": str(root).replace("\\", "/"),
        "baseline_path": str(baseline_path).replace("\\", "/"),
        "total_findings": len(records),
        "new_violations": len(new_ids or set()),
        "findings": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
        finding_id = entry.get("id")
        if isinstance(finding_id, str):
            indexed[finding_id] = entry
    return indexed


def find_new_violations(findings: Iterable[Finding], baseline: dict[str, dict[str, object]]) -> list[Finding]:
    """Return findings not covered by baseline id."""
    new_findings: list[Finding] = []
    for finding in findings:
        if baseline.get(finding.finding_id) is None:
            new_findings.append(finding)
    return new_findings


def write_baseline(path: Path, findings: Iterable[Finding], root: Path) -> None:
    """Write committed baseline with required rationale/hash fields."""
    entries = []
    for finding in findings:
        record = finding.to_record(root=root)
        record["rationale"] = BASELINE_RATIONALE
        entries.append(record)

    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "tests_dir": str(root).replace("\\", "/"),
        "entries": sorted(entries, key=lambda item: (str(item["file"]), int(item["line"]), str(item["pattern"]))),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_rebaseline_diff(old: dict[str, dict[str, object]], findings: list[Finding], root: Path) -> None:
    """Print diff summary for rebaseline workflows."""
    current = {finding.finding_id: finding.to_record(root=root) for finding in findings}
    old_ids = set(old.keys())
    current_ids = set(current.keys())
    added = sorted(current_ids - old_ids)
    removed = sorted(old_ids - current_ids)
    changed_hash = sorted(
        finding_id
        for finding_id in current_ids & old_ids
        if old[finding_id].get("file_hash") != current[finding_id].get("file_hash")
    )

    print("Rebaseline diff summary:")
    print(f"  Added: {len(added)}")
    print(f"  Removed: {len(removed)}")
    print(f"  Hash-changed: {len(changed_hash)}")
    for finding_id in added[:10]:
        row = current[finding_id]
        print(f"  + {row['file']}:{row['line']} [{row['pattern']}]")
    for finding_id in removed[:10]:
        row = old[finding_id]
        print(f"  - {row.get('file')}:{row.get('line')} [{row.get('pattern')}]")
    for finding_id in changed_hash[:10]:
        row = current[finding_id]
        print(f"  * {row['file']}:{row['line']} [{row['pattern']}] file_hash changed")


def _is_pytest_assertion_context(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    base, attr = _call_name(node)
    if base in {"pytest", "py.test"} and attr in {"raises", "warns"}:
        return True
    if not base and attr in {"raises", "warns"}:
        return True
    return False


def main() -> int:
    """Parse CLI args, run the scan, and output reports."""
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
        help="CSV path for report output.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/anti-pattern-analysis/test_quality_report.json"),
        help="JSON path for full report output.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(".github/test-quality-baseline.json"),
        help="Baseline file used for --check / --rebaseline.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if findings include violations not present in baseline.",
    )
    parser.add_argument(
        "--rebaseline",
        action="store_true",
        help="Regenerate baseline from current findings and print a diff summary.",
    )
    args = parser.parse_args()

    tests_dir = args.tests_dir.resolve()
    findings = scan_tests(tests_dir)
    baseline = load_baseline(args.baseline)
    new_violations = find_new_violations(findings, baseline)
    new_ids = {finding.finding_id for finding in new_violations}

    write_csv_report(findings, args.output, tests_dir)
    write_json_report(findings, args.report, tests_dir, args.baseline, new_ids)

    if args.rebaseline:
        print_rebaseline_diff(baseline, findings, tests_dir)
        write_baseline(args.baseline, findings, tests_dir)
        print(f"Baseline written to {args.baseline}")

    print(f"Found {len(findings)} anti-patterns.")
    print(f"CSV report written to {args.output}")
    print(f"JSON report written to {args.report}")

    if args.check:
        if not args.baseline.exists():
            print(f"ERROR: baseline not found at {args.baseline}")
            return 1
        if new_violations:
            print(f"New violations versus baseline: {len(new_violations)}")
            for finding in new_violations[:25]:
                file_path = _to_relative(finding.path, tests_dir)
                print(f"  {file_path}:{finding.line} [{finding.pattern}]")
            return 1
        print("No new violations versus baseline.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
