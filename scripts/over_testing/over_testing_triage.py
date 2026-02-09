"""Triage over-testing hotspots and coverage gaps for test-quality optimization."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from coverage import Coverage
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "coverage is required to include per-test contexts. Install dev dependencies."
    ) from exc


@dataclass(frozen=True)
class FileSummary:
    """Summary statistics per source file."""

    path: str
    lines_covered: int
    mean_count: float
    median_count: float
    max_count: int
    over_threshold: int


@dataclass(frozen=True)
class HotspotLine:
    """Single line hotspot entry."""

    path: str
    line: int
    test_count: int


@dataclass(frozen=True)
class HotspotBlock:
    """Block hotspot entry."""

    path: str
    start_line: int
    end_line: int
    test_count: int
    length: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Triage over-testing hotspots by summarizing per-test coverage counts. "
            "Run scripts/over_testing/over_testing_report.py first to produce inputs."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/over_testing/summary.json"),
        help="Path to summary.json from over_testing_report.py.",
    )
    parser.add_argument(
        "--lines",
        type=Path,
        default=Path("reports/over_testing/line_coverage_counts.csv"),
        help="Path to line_coverage_counts.csv from over_testing_report.py.",
    )
    parser.add_argument(
        "--blocks",
        type=Path,
        default=Path("reports/over_testing/block_coverage_counts.csv"),
        help="Path to block_coverage_counts.csv from over_testing_report.py.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("reports/over_testing/metadata.json"),
        help="Path to metadata.json from over_testing_report.py.",
    )
    parser.add_argument(
        "--coverage-file",
        type=Path,
        default=Path(".coverage"),
        help="Coverage data file to read test contexts from.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used to resolve file paths.",
    )
    parser.add_argument(
        "--context-regex",
        type=str,
        default=None,
        help="Regex to filter coverage contexts (default: include all).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Minimum test_count for hotspots.",
    )
    parser.add_argument(
        "--top-files",
        type=int,
        default=15,
        help="Number of files to report.",
    )
    parser.add_argument(
        "--top-lines",
        type=int,
        default=30,
        help="Number of line hotspots to report.",
    )
    parser.add_argument(
        "--top-blocks",
        type=int,
        default=30,
        help="Number of block hotspots to report.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/over_testing/triage.json"),
        help="JSON output with ranked hotspots.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/over_testing/triage.md"),
        help="Markdown output with ranked hotspots and process notes.",
    )
    parser.add_argument(
        "--include-contexts",
        action="store_true",
        help="Include per-test context lists for the top hotspots.",
    )
    parser.add_argument(
        "--contexts-per-hotspot",
        type=int,
        default=25,
        help="Number of test contexts to include per hotspot when enabled.",
    )
    parser.add_argument(
        "--output-contexts-json",
        type=Path,
        default=Path("reports/over_testing/hotspot_contexts.json"),
        help="JSON output for hotspot-to-test context mappings.",
    )
    parser.add_argument(
        "--output-contexts-md",
        type=Path,
        default=Path("reports/over_testing/hotspot_contexts.md"),
        help="Markdown output for hotspot-to-test context mappings.",
    )
    return parser.parse_args()


def _load_summary(path: Path) -> list[FileSummary]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        FileSummary(
            path=item["file"],
            lines_covered=item["lines_covered"],
            mean_count=float(item["mean_count"]),
            median_count=float(item["median_count"]),
            max_count=int(item["max_count"]),
            over_threshold=int(item["over_threshold"]),
        )
        for item in payload
    ]


def _load_line_hotspots(path: Path, min_count: int) -> list[HotspotLine]:
    hotspots: list[HotspotLine] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            count = int(row["test_count"])
            if count < min_count:
                continue
            hotspots.append(HotspotLine(row["file"], int(row["line"]), count))
    return hotspots


def _load_block_hotspots(path: Path, min_count: int) -> list[HotspotBlock]:
    hotspots: list[HotspotBlock] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            count = int(row["test_count"])
            if count < min_count:
                continue
            hotspots.append(
                HotspotBlock(
                    row["file"],
                    int(row["start_line"]),
                    int(row["end_line"]),
                    count,
                    int(row["length"]),
                )
            )
    return hotspots


def _filter_contexts(contexts: Iterable[str], pattern: str | None) -> list[str]:
    if pattern is None:
        return sorted(set(contexts))
    import re

    regex = re.compile(pattern)
    return sorted({context for context in contexts if regex.search(context)})


def _load_contexts(path: Path):
    coverage = Coverage(data_file=str(path))
    coverage.load()
    return coverage.get_data()


def _resolve_coverage_path(data, repo_root: Path, rel_path: str) -> str | None:
    target = (repo_root / Path(rel_path)).resolve()
    target_str = str(target)
    target_norm = target_str.casefold()
    for measured in data.measured_files():
        measured_norm = str(Path(measured)).casefold()
        if measured_norm == target_norm:
            return measured
    # Fallback: suffix match (handles differing roots)
    rel_norm = str(Path(rel_path)).casefold()
    for measured in data.measured_files():
        measured_path = Path(measured)
        if str(measured_path).casefold().endswith(rel_norm):
            return str(measured_path)
    return None


def _contexts_for_line(
    data,
    repo_root: Path,
    hotspot: HotspotLine,
    pattern: str | None,
) -> list[str]:
    resolved = _resolve_coverage_path(data, repo_root, hotspot.path)
    if resolved is None:
        return []
    contexts_by_line = data.contexts_by_lineno(str(resolved))
    contexts = contexts_by_line.get(hotspot.line, [])
    return _filter_contexts(contexts, pattern)


def _contexts_for_block(
    data,
    repo_root: Path,
    hotspot: HotspotBlock,
    pattern: str | None,
) -> list[str]:
    resolved = _resolve_coverage_path(data, repo_root, hotspot.path)
    if resolved is None:
        return []
    contexts_by_line = data.contexts_by_lineno(str(resolved))
    contexts: list[str] = []
    for line in range(hotspot.start_line, hotspot.end_line + 1):
        contexts.extend(contexts_by_line.get(line, []))
    return _filter_contexts(contexts, pattern)


def _rank_files(summaries: list[FileSummary]) -> list[dict[str, object]]:
    ranked = []
    for item in summaries:
        ratio = 0.0
        if item.lines_covered:
            ratio = item.over_threshold / item.lines_covered
        ranked.append(
            {
                "file": item.path,
                "lines_covered": item.lines_covered,
                "mean_count": item.mean_count,
                "median_count": item.median_count,
                "max_count": item.max_count,
                "over_threshold": item.over_threshold,
                "over_ratio": round(ratio, 4),
            }
        )
    ranked.sort(key=lambda entry: (entry["over_ratio"], entry["max_count"]), reverse=True)
    return ranked


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Over-Testing Triage Report")
    lines.append("")
    metadata = payload.get("metadata", {})
    lines.append("## Metadata")
    lines.append("")
    for key, value in metadata.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    lines.append("## Top over-tested files")
    lines.append("")
    lines.append("| File | Over ratio | Max count | Over-threshold lines | Lines covered |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in payload["top_files"]:
        lines.append(
            "| {file} | {over_ratio} | {max_count} | {over_threshold} | {lines_covered} |".format(
                **entry
            )
        )
    lines.append("")

    lines.append("## Top hotspot lines")
    lines.append("")
    lines.append("| File | Line | Test count |")
    lines.append("| --- | --- | --- |")
    for entry in payload["top_lines"]:
        lines.append(
            f"| {entry['file']} | {entry['line']} | {entry['test_count']} |"
        )
    lines.append("")

    lines.append("## Top hotspot blocks")
    lines.append("")
    lines.append("| File | Start | End | Test count | Length |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in payload["top_blocks"]:
        lines.append(
            f"| {entry['file']} | {entry['start_line']} | {entry['end_line']} | "
            f"{entry['test_count']} | {entry['length']} |"
        )
    lines.append("")

    lines.append("## Suggested process")
    lines.append("")
    lines.append("1. Run pytest with `--cov-context=test` to record per-test contexts.")
    lines.append("2. Run `scripts/over_testing/over_testing_report.py --require-multiple-contexts`.")
    lines.append("3. Run this script and inspect the top hotspots above.")
    lines.append("4. For each hotspot, review tests for duplicate assertions or setup-only coverage.")
    lines.append("5. Consolidate redundant tests and re-run the reports to confirm lower counts.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_contexts_markdown(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Hotspot Context Mapping")
    lines.append("")
    lines.append("## Top hotspot lines")
    lines.append("")
    for entry in payload.get("line_contexts", []):
        lines.append(
            f"### {entry['file']}:{entry['line']} (test_count={entry['test_count']})"
        )
        if entry["contexts"]:
            for context in entry["contexts"]:
                lines.append(f"- {context}")
        else:
            lines.append("- (no contexts found)")
        lines.append("")

    lines.append("## Top hotspot blocks")
    lines.append("")
    for entry in payload.get("block_contexts", []):
        lines.append(
            f"### {entry['file']}:{entry['start_line']}-{entry['end_line']} "
            f"(test_count={entry['test_count']})"
        )
        if entry["contexts"]:
            for context in entry["contexts"]:
                lines.append(f"- {context}")
        else:
            lines.append("- (no contexts found)")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()

    summary_path = args.summary
    lines_path = args.lines
    blocks_path = args.blocks
    metadata_path = args.metadata

    if not summary_path.exists():
        print(f"Missing summary file: {summary_path}")
        return 1
    if not lines_path.exists():
        print(f"Missing line coverage file: {lines_path}")
        return 1
    if not blocks_path.exists():
        print(f"Missing block coverage file: {blocks_path}")
        return 1

    summaries = _load_summary(summary_path)
    line_hotspots = _load_line_hotspots(lines_path, args.min_count)
    block_hotspots = _load_block_hotspots(blocks_path, args.min_count)

    ranked_files = _rank_files(summaries)
    ranked_lines = sorted(line_hotspots, key=lambda item: item.test_count, reverse=True)
    ranked_blocks = sorted(block_hotspots, key=lambda item: item.test_count, reverse=True)

    metadata: dict[str, object] = {
        "min_count": args.min_count,
        "summary": str(summary_path),
        "lines": str(lines_path),
        "blocks": str(blocks_path),
    }
    if metadata_path.exists():
        metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))

    payload = {
        "metadata": metadata,
        "top_files": ranked_files[: args.top_files],
        "top_lines": [
            {"file": item.path, "line": item.line, "test_count": item.test_count}
            for item in ranked_lines[: args.top_lines]
        ],
        "top_blocks": [
            {
                "file": item.path,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "test_count": item.test_count,
                "length": item.length,
            }
            for item in ranked_blocks[: args.top_blocks]
        ],
    }

    contexts_payload: dict[str, object] = {}
    if args.include_contexts:
        data = _load_contexts(args.coverage_file)
        repo_root = args.repo_root.resolve()
        contexts_payload = {
            "metadata": {
                "coverage_file": str(args.coverage_file),
                "context_regex": args.context_regex,
                "contexts_per_hotspot": args.contexts_per_hotspot,
            },
            "line_contexts": [
                {
                    "file": item.path,
                    "line": item.line,
                    "test_count": item.test_count,
                    "contexts": _contexts_for_line(
                        data, repo_root, item, args.context_regex
                    )[: args.contexts_per_hotspot],
                }
                for item in ranked_lines[: args.top_lines]
            ],
            "block_contexts": [
                {
                    "file": item.path,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "test_count": item.test_count,
                    "contexts": _contexts_for_block(
                        data, repo_root, item, args.context_regex
                    )[: args.contexts_per_hotspot],
                }
                for item in ranked_blocks[: args.top_blocks]
            ],
        }

    _write_json(args.output_json, payload)
    _write_markdown(args.output_md, payload)
    if args.include_contexts:
        _write_json(args.output_contexts_json, contexts_payload)
        _write_contexts_markdown(args.output_contexts_md, contexts_payload)

    print(f"Triage JSON written to {args.output_json}")
    print(f"Triage Markdown written to {args.output_md}")
    if args.include_contexts:
        print(f"Hotspot contexts JSON written to {args.output_contexts_json}")
        print(f"Hotspot contexts Markdown written to {args.output_contexts_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
