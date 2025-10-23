"""Lightweight coverage stub using Python's tracing hooks."""

from __future__ import annotations

import ast
import fnmatch
import os
import sys
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

__all__ = ["Coverage"]


@dataclass
class _FileStats:
    path: str
    executable: Set[int]
    executed: Set[int]

    @property
    def total(self) -> int:
        return len(self.executable)

    @property
    def covered(self) -> int:
        return len(self.executable & self.executed)

    @property
    def missing(self) -> Set[int]:
        return self.executable - self.executed


def _discover_source_files(sources: Sequence[str]) -> List[str]:
    files: List[str] = []
    for src in sources:
        if not src:
            continue
        path = src
        if os.path.isdir(path):
            for root, _dirs, filenames in os.walk(path):
                for name in filenames:
                    if name.endswith(".py"):
                        files.append(os.path.abspath(os.path.join(root, name)))
        else:
            if os.path.isfile(path):
                files.append(os.path.abspath(path))
    return sorted(set(files))


def _statement_lines(path: str) -> Set[int]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return set()

    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        return set()

    pragma_lines = {
        idx + 1 for idx, line in enumerate(text.splitlines()) if "# pragma: no cover" in line
    }

    lines: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            if (
                isinstance(node, ast.Expr)
                and isinstance(getattr(node, "value", None), ast.Constant)
                and isinstance(node.value.value, str)
            ):
                # Ignore docstrings.
                continue

            lineno = getattr(node, "lineno", None)
            if lineno is None:
                continue
            if int(lineno) in pragma_lines:
                continue
            lines.add(int(lineno))
    return lines


class Coverage:  # pragma: no cover - integration harness
    """Minimal drop-in replacement for ``coverage.Coverage`` used in tests."""

    def __init__(
        self,
        source: Optional[Sequence[str]] = None,
        omit: Optional[Sequence[str]] = None,
        include: Optional[Sequence[str]] = None,
        data_file: Optional[str] = None,
        **_kwargs: object,
    ) -> None:
        self._source_inputs = list(source or [])
        self._omit = list(omit or [])
        self._include = list(include or [])
        self._data_file = data_file

        self._files: Dict[str, _FileStats] = {}
        self._saved = False
        self._prepare_files()

    def _prepare_files(self) -> None:
        files = _discover_source_files(self._source_inputs) if self._source_inputs else []
        for path in files:
            if self._is_omitted(path):
                continue
            executable = _statement_lines(path)
            self._files[path] = _FileStats(path=path, executable=executable, executed=set())

    def _is_omitted(self, path: str) -> bool:
        return any(fnmatch.fnmatch(path, pattern) for pattern in self._omit)

    def start(self) -> None:
        if not self._files:
            return

        def tracer(frame, event, arg):
            filename = os.path.abspath(frame.f_code.co_filename)
            stats = self._files.get(filename)
            if stats is None:
                return None
            if event == "line":
                stats.executed.add(frame.f_lineno)
            return tracer

        self._tracer = tracer
        sys.settrace(tracer)
        threading.settrace(tracer)

    def stop(self) -> None:
        sys.settrace(None)
        threading.settrace(None)

    def save(self) -> None:
        self._saved = True

    def _iter_stats(self) -> Iterable[_FileStats]:
        return self._files.values()

    def report(
        self,
        show_missing: bool = False,
        file=None,
        fail_under: Optional[float] = None,
        **_kwargs: object,
    ) -> float:
        if file is None:
            file = sys.stdout

        stats_list = sorted(self._iter_stats(), key=lambda item: item.path)
        total_lines = sum(stat.total for stat in stats_list)
        total_hits = sum(stat.covered for stat in stats_list)
        overall = 100.0 if total_lines == 0 else (total_hits / total_lines) * 100.0

        header = f"{'Name':<70} {'Stmts':>6} {'Miss':>6} {'Cover':>6}"
        file.write(header + "\n")
        file.write("-" * len(header) + "\n")

        for stats in stats_list:
            total = stats.total
            hit = stats.covered
            miss = total - hit
            percent = 100.0 if total == 0 else (hit / total) * 100.0
            rel_path = os.path.relpath(stats.path)
            file.write(f"{rel_path:<70} {total:6d} {miss:6d} {percent:5.1f}%\n")
            if show_missing and miss:
                missing = ",".join(str(num) for num in sorted(stats.missing))
                file.write(f"    Missing: {missing}\n")

        file.write(
            "\nTOTAL                                        "
            f"{total_lines:6d} {total_lines - total_hits:6d} {overall:5.1f}%\n"
        )

        if fail_under is not None and overall < fail_under:
            raise RuntimeError(
                f"Coverage {overall:.1f}% fell below fail-under threshold {fail_under:.1f}%"
            )

        return overall

    def html_report(self, **_kwargs: object) -> None:
        return None

    def xml_report(self, **_kwargs: object) -> None:
        return None

    def erase(self) -> None:
        for stats in self._files.values():
            stats.executed.clear()
        self._saved = False
