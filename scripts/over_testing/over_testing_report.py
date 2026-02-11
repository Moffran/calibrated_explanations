"""Generate per-test coverage counts to detect over-testing hotspots."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


try:
    from coverage import Coverage, CoverageData
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "coverage is required. Install the dev dependencies (pytest-cov includes coverage)."
    ) from exc


@dataclass(frozen=True)
class LineCount:
    """Per-line coverage count entry.

    Parameters
    ----------
    path:
        Path to the source file (relative to repo root).
    line:
        Line number in the source file (1-based).
    count:
        Number of distinct test contexts that executed the line.
    """

    path: Path
    line: int
    count: int


@dataclass(frozen=True)
class BlockCount:
    """Contiguous block coverage count entry.

    Parameters
    ----------
    path:
        Path to the source file (relative to repo root).
    start_line:
        Starting line number of the block (1-based).
    end_line:
        Ending line number of the block (1-based).
    count:
        Number of distinct test contexts that executed the block.
    """

    path: Path
    start_line: int
    end_line: int
    count: int


@dataclass(frozen=True)
class SummaryStats:
    """Summary statistics for a single file.

    Parameters
    ----------
    path:
        Path to the source file (relative to repo root).
    lines_covered:
        Total number of executed lines in the file.
    mean_count:
        Mean number of tests per executed line.
    median_count:
        Median number of tests per executed line.
    max_count:
        Maximum number of tests hitting a single line.
    over_threshold:
        Count of executed lines at or above the over-testing threshold.
    """

    path: Path
    lines_covered: int
    mean_count: float
    median_count: float
    max_count: int
    over_threshold: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize per-test coverage counts using coverage.py contexts. "
            "Requires coverage data recorded with pytest-cov --cov-context=test."
        )
    )
    parser.add_argument(
        "--coverage-file",
        type=Path,
        default=Path(".coverage"),
        help="Path to the coverage data file (default: .coverage).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used for relative paths.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("src/calibrated_explanations"),
        help="Source root to include in the analysis.",
    )
    parser.add_argument(
        "--context-regex",
        type=str,
        default=None,
        help="Regex to filter coverage contexts (default: include all).",
    )
    parser.add_argument(
        "--require-multiple-contexts",
        action="store_true",
        help=(
            "Fail if fewer than two distinct contexts are detected. "
            "Use this to ensure pytest ran with --cov-context=test."
        ),
    )
    parser.add_argument(
        "--over-testing-threshold",
        type=int,
        default=20,
        help="Threshold for flagging heavily exercised lines.",
    )
    parser.add_argument(
        "--output-lines",
        type=Path,
        default=Path("reports/over_testing/line_coverage_counts.csv"),
        help="CSV output for per-line counts.",
    )
    parser.add_argument(
        "--output-blocks",
        type=Path,
        default=Path("reports/over_testing/block_coverage_counts.csv"),
        help="CSV output for per-block counts.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("reports/over_testing/summary.json"),
        help="JSON output for summary statistics.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("reports/over_testing/metadata.json"),
        help="JSON output for analysis metadata and warnings.",
    )
    return parser.parse_args()


def _load_coverage_data(path: Path) -> CoverageData:
    if not path.exists():
        raise FileNotFoundError(
            f"Coverage data file not found: {path}. "
            "Run pytest with --cov-context=test to generate it."
        )
    coverage = Coverage(data_file=str(path))
    coverage.load()
    return coverage.get_data()


def _filter_contexts(contexts: Iterable[str], pattern: re.Pattern[str] | None) -> set[str]:
    if pattern is None:
        return set(contexts)
    return {context for context in contexts if pattern.search(context)}


def _collect_contexts(
    data: CoverageData,
    repo_root: Path,
    source_root: Path,
    pattern: re.Pattern[str] | None,
) -> set[str]:
    repo_root = repo_root.resolve()
    source_root = (repo_root / source_root).resolve()
    contexts: set[str] = set()
    if hasattr(data, "contexts_by_lineno"):
        for filename in data.measured_files():
            file_path = Path(filename).resolve()
            if not file_path.is_relative_to(source_root):
                continue
            context_map = data.contexts_by_lineno(str(file_path))
            for line_contexts in context_map.values():
                contexts.update(_filter_contexts(line_contexts, pattern))
    return contexts


def _line_counts(
    data: CoverageData,
    repo_root: Path,
    source_root: Path,
    pattern: re.Pattern[str] | None,
) -> list[LineCount]:
    if not hasattr(data, "contexts_by_lineno"):
        raise RuntimeError(
            "coverage.py contexts not available. Upgrade coverage and run pytest "
            "with --cov-context=test."
        )

    repo_root = repo_root.resolve()
    source_root = (repo_root / source_root).resolve()

    entries: list[LineCount] = []
    for filename in data.measured_files():
        file_path = Path(filename).resolve()
        if not file_path.is_relative_to(source_root):
            continue
        context_map = data.contexts_by_lineno(str(file_path))
        for line, contexts in context_map.items():
            filtered = _filter_contexts(contexts, pattern)
            if not filtered:
                continue
            relative_path = file_path.relative_to(repo_root)
            entries.append(LineCount(relative_path, int(line), len(filtered)))
    return sorted(entries, key=lambda item: (str(item.path), item.line))


def _block_counts(line_counts: list[LineCount]) -> list[BlockCount]:
    blocks: list[BlockCount] = []
    if not line_counts:
        return blocks

    current = line_counts[0]
    start_line = current.line
    last_line = current.line
    last_count = current.count

    for entry in line_counts[1:]:
        if entry.path == current.path and entry.count == last_count and entry.line == last_line + 1:
            last_line = entry.line
            continue
        blocks.append(BlockCount(current.path, start_line, last_line, last_count))
        current = entry
        start_line = entry.line
        last_line = entry.line
        last_count = entry.count

    blocks.append(BlockCount(current.path, start_line, last_line, last_count))
    return blocks


def _summaries(line_counts: list[LineCount], threshold: int) -> list[SummaryStats]:
    if not line_counts:
        return []
    by_file: dict[Path, list[int]] = {}
    for entry in line_counts:
        by_file.setdefault(entry.path, []).append(entry.count)

    summaries: list[SummaryStats] = []
    for path, counts in sorted(by_file.items(), key=lambda item: str(item[0])):
        summaries.append(
            SummaryStats(
                path=path,
                lines_covered=len(counts),
                mean_count=float(statistics.mean(counts)),
                median_count=float(statistics.median(counts)),
                max_count=max(counts),
                over_threshold=sum(1 for count in counts if count >= threshold),
            )
        )
    return summaries


def _write_line_csv(entries: list[LineCount], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "line", "test_count"])
        for entry in entries:
            writer.writerow([str(entry.path), entry.line, entry.count])


def _write_block_csv(entries: list[BlockCount], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "start_line", "end_line", "test_count", "length"])
        for entry in entries:
            writer.writerow([
                str(entry.path),
                entry.start_line,
                entry.end_line,
                entry.count,
                entry.end_line - entry.start_line + 1,
            ])


def _write_summary_json(entries: list[SummaryStats], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "file": str(entry.path),
            "lines_covered": entry.lines_covered,
            "mean_count": round(entry.mean_count, 3),
            "median_count": round(entry.median_count, 3),
            "max_count": entry.max_count,
            "over_threshold": entry.over_threshold,
        }
        for entry in entries
    ]
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_metadata_json(metadata: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> int:
    """Run the over-testing analysis.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """

    args = _parse_args()
    pattern = re.compile(args.context_regex) if args.context_regex else None

    try:
        data = _load_coverage_data(args.coverage_file)
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        return 1

    contexts = _collect_contexts(
        data,
        repo_root=args.repo_root,
        source_root=args.source_root,
        pattern=pattern,
    )
    if args.require_multiple_contexts and len(contexts) < 2:
        print(
            "Fewer than two coverage contexts detected. "
            "Re-run pytest with --cov-context=test."
        )
        return 1

    line_counts = _line_counts(
        data,
        repo_root=args.repo_root,
        source_root=args.source_root,
        pattern=pattern,
    )
    block_counts = _block_counts(line_counts)
    summaries = _summaries(line_counts, args.over_testing_threshold)

    metadata = {
        "coverage_file": str(args.coverage_file),
        "contexts_detected": len(contexts),
        "context_regex": args.context_regex,
        "source_root": str(args.source_root),
        "over_testing_threshold": args.over_testing_threshold,
        "warnings": [],
    }
    if len(contexts) < 2:
        metadata["warnings"].append(
            "Only one coverage context detected. Over-testing analysis is unreliable; "
            "run pytest with --cov-context=test."
        )
    if not line_counts:
        metadata["warnings"].append("No covered lines found under the source root.")

    _write_line_csv(line_counts, args.output_lines)
    _write_block_csv(block_counts, args.output_blocks)
    _write_summary_json(summaries, args.output_summary)
    _write_metadata_json(metadata, args.output_metadata)

    print(f"Line counts written to {args.output_lines}")
    print(f"Block counts written to {args.output_blocks}")
    print(f"Summary written to {args.output_summary}")
    print(f"Metadata written to {args.output_metadata}")
    print(f"Total lines recorded: {len(line_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
