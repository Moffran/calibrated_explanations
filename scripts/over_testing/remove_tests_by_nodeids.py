"""Remove pytest tests by nodeid safely, including decorators.

This utility removes test functions/methods addressed by nodeids like:
  tests/unit/test_mod.py::test_name
  tests/unit/test_mod.py::TestClass::test_name

Only target tests are removed. Decorators are included in the removed span.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_nodeid(nodeid: str) -> Tuple[str, str | None, str]:
    parts = nodeid.split("::")
    if len(parts) == 2:
        file_path, test_name = parts
        return file_path, None, test_name
    if len(parts) == 3:
        file_path, class_name, test_name = parts
        return file_path, class_name, test_name
    raise ValueError(f"Unsupported nodeid format: {nodeid}")


def _node_span(node: ast.AST) -> Tuple[int, int]:
    start = getattr(node, "lineno")
    end = getattr(node, "end_lineno")
    decorators = getattr(node, "decorator_list", None) or []
    if decorators:
        start = min(start, min(d.lineno for d in decorators))
    return int(start), int(end)


def _collect_targets(tree: ast.Module, class_name: str | None, test_name: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    if class_name is None:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == test_name:
                spans.append(_node_span(node))
        return spans

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == test_name:
                    spans.append(_node_span(child))
    return spans


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _remove_spans(lines: List[str], spans: List[Tuple[int, int]]) -> List[str]:
    keep = [True] * len(lines)
    for start, end in spans:
        for idx in range(start - 1, end):
            if 0 <= idx < len(keep):
                keep[idx] = False
    return [line for i, line in enumerate(lines) if keep[i]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodeids-file", required=True, help="Path to text file containing one nodeid per line")
    parser.add_argument("--write-applied", help="Optional output file for successfully removed nodeids")
    parser.add_argument("--write-missing", help="Optional output file for nodeids not found")
    args = parser.parse_args()

    root = Path(".").resolve()
    nodeids = [
        line.strip()
        for line in Path(args.nodeids_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    file_targets: Dict[Path, List[Tuple[str | None, str, str]]] = {}
    for nodeid in nodeids:
        try:
            rel_path, class_name, test_name = _parse_nodeid(nodeid)
        except ValueError:
            continue
        path = (root / rel_path).resolve()
        file_targets.setdefault(path, []).append((class_name, test_name, nodeid))

    applied: List[str] = []
    missing: List[str] = []

    for path, targets in file_targets.items():
        if not path.exists():
            missing.extend(nodeid for _, _, nodeid in targets)
            continue

        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            missing.extend(nodeid for _, _, nodeid in targets)
            continue

        spans: List[Tuple[int, int]] = []
        matched_ids = set()
        for class_name, test_name, nodeid in targets:
            found = _collect_targets(tree, class_name, test_name)
            if found:
                spans.extend(found)
                matched_ids.add(nodeid)
            else:
                missing.append(nodeid)

        if not spans:
            continue

        merged = _merge_spans(spans)
        new_lines = _remove_spans(source.splitlines(keepends=True), merged)
        path.write_text("".join(new_lines), encoding="utf-8")
        applied.extend(sorted(matched_ids))

    if args.write_applied:
        Path(args.write_applied).write_text("\n".join(applied) + ("\n" if applied else ""), encoding="utf-8")
    if args.write_missing:
        Path(args.write_missing).write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

    print(f"removed {len(applied)} tests; missing {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
