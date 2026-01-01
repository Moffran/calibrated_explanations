"""Summarize fast-filtering evaluation results and optionally render plots."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import statistics as stats


@dataclass(frozen=True)
class AblationRow:
    dataset: str
    task: str
    model: str
    with_filtering: float | None
    without_filtering: float | None
    ratio: float | None


@dataclass(frozen=True)
class OverlapRow:
    dataset: str
    task: str
    model: str
    jaccard_topk: float | None
    topk_inclusion: float | None
    spearman_rank: float | None
    kept_feature_count: float | None


@dataclass(frozen=True)
class TopkRow:
    dataset: str
    task: str
    model: str
    topk: int
    time_ratio: float | None
    jaccard_topk: float | None
    topk_inclusion: float | None
    spearman_rank: float | None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_to_datasets(synthetic: dict[str, Any]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for entry in synthetic.get("experiments", []):
        task = entry.get("task", "unknown")
        n_features = entry.get("n_features", "unknown")
        model = entry.get("model", "unknown")
        dataset_name = f"synthetic_f{n_features}"
        datasets.setdefault(dataset_name, {"task": task, "models": {}})
        datasets[dataset_name]["models"][model] = {
            "baseline": entry.get("baseline", {}),
            "top_k": entry.get("top_k", {}),
        }
    return datasets


def _maybe_median(value: dict[str, Any]) -> float | None:
    if not isinstance(value, dict):
        return None
    median = value.get("median")
    return float(median) if median is not None else None


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "median": float(stats.median(values)),
        "mean": float(stats.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _format_float(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _write_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _extract_ablation(ablation: dict[str, Any]) -> list[AblationRow]:
    rows: list[AblationRow] = []
    for dname, dval in ablation.get("datasets", {}).items():
        task = dval.get("task", "unknown")
        for model, mval in dval.get("models", {}).items():
            runs = mval.get("runs", {})
            with_f = runs.get("with_filtering", {}).get("explain_factual")
            without_f = runs.get("without_filtering", {}).get("explain_factual")
            ratio = None
            if with_f is not None and without_f:
                ratio = with_f / without_f
            rows.append(
                AblationRow(
                    dataset=dname,
                    task=task,
                    model=model,
                    with_filtering=with_f,
                    without_filtering=without_f,
                    ratio=ratio,
                )
            )
    return rows


def _extract_overlap(overlap: dict[str, Any]) -> list[OverlapRow]:
    rows: list[OverlapRow] = []
    for dname, dval in overlap.get("datasets", {}).items():
        task = dval.get("task", "unknown")
        for model, mval in dval.get("models", {}).items():
            summary = mval.get("explain_factual", {}).get("summary", {})
            rows.append(
                OverlapRow(
                    dataset=dname,
                    task=task,
                    model=model,
                    jaccard_topk=_maybe_median(summary.get("jaccard_topk", {})),
                    topk_inclusion=_maybe_median(summary.get("topk_inclusion", {})),
                    spearman_rank=_maybe_median(summary.get("spearman_rank", {})),
                    kept_feature_count=_maybe_median(summary.get("kept_feature_count", {})),
                )
            )
    return rows


def _extract_topk(topk: dict[str, Any]) -> list[TopkRow]:
    rows: list[TopkRow] = []
    for dname, dval in topk.get("datasets", {}).items():
        task = dval.get("task", "unknown")
        for model, mval in dval.get("models", {}).items():
            baseline = mval.get("baseline", {})
            base_time = baseline.get("explain_factual")
            for topk_value, tval in mval.get("top_k", {}).items():
                try:
                    topk_int = int(topk_value)
                except ValueError:
                    continue
                explain = tval.get("explain_factual", {})
                t_time = explain.get("time")
                time_ratio = None
                if base_time and t_time is not None:
                    time_ratio = t_time / base_time
                overlap = explain.get("overlap", {})
                rows.append(
                    TopkRow(
                        dataset=dname,
                        task=task,
                        model=model,
                        topk=topk_int,
                        time_ratio=time_ratio,
                        jaccard_topk=_maybe_median(overlap.get("jaccard_topk", {})),
                        topk_inclusion=_maybe_median(overlap.get("topk_inclusion", {})),
                        spearman_rank=_maybe_median(overlap.get("spearman_rank", {})),
                    )
                )
    return rows


def _render_ablation(rows: list[AblationRow]) -> str:
    headers = [
        "dataset",
        "task",
        "model",
        "with_filtering",
        "without_filtering",
        "ratio",
    ]
    table_rows = [
        [
            r.dataset,
            r.task,
            r.model,
            _format_float(r.with_filtering),
            _format_float(r.without_filtering),
            _format_float(r.ratio),
        ]
        for r in rows
    ]
    return _write_table(headers, table_rows)


def _render_overlap(rows: list[OverlapRow]) -> str:
    headers = [
        "dataset",
        "task",
        "model",
        "jaccard_topk",
        "topk_inclusion",
        "spearman_rank",
        "kept_feature_count",
    ]
    table_rows = [
        [
            r.dataset,
            r.task,
            r.model,
            _format_float(r.jaccard_topk),
            _format_float(r.topk_inclusion),
            _format_float(r.spearman_rank),
            _format_float(r.kept_feature_count),
        ]
        for r in rows
    ]
    return _write_table(headers, table_rows)


def _render_topk(rows: list[TopkRow]) -> str:
    headers = [
        "dataset",
        "task",
        "model",
        "topk",
        "time_ratio",
        "jaccard_topk",
        "topk_inclusion",
        "spearman_rank",
    ]
    table_rows = [
        [
            r.dataset,
            r.task,
            r.model,
            str(r.topk),
            _format_float(r.time_ratio),
            _format_float(r.jaccard_topk),
            _format_float(r.topk_inclusion),
            _format_float(r.spearman_rank),
        ]
        for r in rows
    ]
    return _write_table(headers, table_rows)


def _render_aggregate(rows: list[TopkRow]) -> str:
    headers = ["topk", "time_ratio", "jaccard_topk", "topk_inclusion", "spearman_rank"]
    by_topk: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        entry = by_topk.setdefault(
            row.topk,
            {"time_ratio": [], "jaccard_topk": [], "topk_inclusion": [], "spearman_rank": []},
        )
        if row.time_ratio is not None:
            entry["time_ratio"].append(row.time_ratio)
        if row.jaccard_topk is not None:
            entry["jaccard_topk"].append(row.jaccard_topk)
        if row.topk_inclusion is not None:
            entry["topk_inclusion"].append(row.topk_inclusion)
        if row.spearman_rank is not None:
            entry["spearman_rank"].append(row.spearman_rank)

    table_rows: list[list[str]] = []
    for topk in sorted(by_topk):
        entry = by_topk[topk]
        table_rows.append(
            [
                str(topk),
                _format_float(_summarize(entry["time_ratio"]).get("median")),
                _format_float(_summarize(entry["jaccard_topk"]).get("median")),
                _format_float(_summarize(entry["topk_inclusion"]).get("median")),
                _format_float(_summarize(entry["spearman_rank"]).get("median")),
            ]
        )
    return _write_table(headers, table_rows)


def _render_synthetic_speed(rows: list[TopkRow]) -> str:
    headers = ["n_features", "topk", "time_ratio"]
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        if not row.dataset.startswith("synthetic_f"):
            continue
        try:
            n_features = int(row.dataset.replace("synthetic_f", ""))
        except ValueError:
            continue

        if row.time_ratio is not None:
            grouped.setdefault((n_features, row.topk), []).append(row.time_ratio)

    table_rows: list[list[str]] = []
    for n_feat, topk in sorted(grouped.keys()):
        ratios = grouped[(n_feat, topk)]
        table_rows.append(
            [
                str(n_feat),
                str(topk),
                _format_float(stats.median(ratios)) if ratios else "NA",
            ]
        )
    return _write_table(headers, table_rows)


def _maybe_plot(out_dir: Path, topk_rows: list[TopkRow]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Plotting skipped (matplotlib unavailable): {exc}")
        return

    by_topk: dict[int, dict[str, list[float]]] = {}
    for row in topk_rows:
        entry = by_topk.setdefault(
            row.topk,
            {"time_ratio": [], "jaccard_topk": [], "topk_inclusion": []},
        )
        if row.time_ratio is not None:
            entry["time_ratio"].append(row.time_ratio)
        if row.jaccard_topk is not None:
            entry["jaccard_topk"].append(row.jaccard_topk)
        if row.topk_inclusion is not None:
            entry["topk_inclusion"].append(row.topk_inclusion)

    topks = sorted(by_topk)
    if not topks:
        print("No top-k rows found for plotting.")
        return

    def _med(values: list[float]) -> float | None:
        return float(stats.median(values)) if values else None

    time_ratio = [_med(by_topk[k]["time_ratio"]) for k in topks]
    jaccard = [_med(by_topk[k]["jaccard_topk"]) for k in topks]
    inclusion = [_med(by_topk[k]["topk_inclusion"]) for k in topks]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(topks, time_ratio, marker="o", label="time_ratio")
    ax.set_xlabel("top_k")
    ax.set_ylabel("median time ratio")
    ax.set_title("Fast filtering runtime ratio vs top_k")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "topk_time_ratio.png", dpi=160)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(topks, jaccard, marker="o", label="jaccard_topk")
    ax.plot(topks, inclusion, marker="o", label="topk_inclusion")
    ax.set_xlabel("top_k")
    ax.set_ylabel("median overlap")
    ax.set_title("Overlap vs top_k")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "topk_overlap.png", dpi=160)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fast-filtering results")
    parser.add_argument(
        "--ablation",
        type=str,
        default="evaluation/fast_filtering/fast_filtering_ablation_multi_results.json",
    )
    parser.add_argument(
        "--overlap",
        type=str,
        default="evaluation/fast_filtering/fast_filtering_feature_overlap_results.json",
    )
    parser.add_argument(
        "--topk",
        type=str,
        default="evaluation/fast_filtering/fast_filtering_topk_sweep_results.json",
    )
    parser.add_argument(
        "--synthetic",
        type=str,
        default="evaluation/fast_filtering/synthetic_feature_sweep_results.json",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--plots", action="store_true", help="Write summary plots to out-dir.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ablation = _read_json(Path(args.ablation))
    overlap = _read_json(Path(args.overlap))
    topk = _read_json(Path(args.topk))
    synthetic_path = Path(args.synthetic)
    synthetic = _read_json(synthetic_path) if synthetic_path.exists() else None

    ablation_rows = _extract_ablation(ablation)
    overlap_rows = _extract_overlap(overlap)
    topk_rows = _extract_topk(topk)

    sections = {
        "Ablation (per dataset/model)": _render_ablation(ablation_rows),
        "Overlap (per dataset/model)": _render_overlap(overlap_rows),
        "Top-k sweep (per dataset/model/top_k)": _render_topk(topk_rows),
        "Top-k sweep (aggregate medians)": _render_aggregate(topk_rows),
    }

    if synthetic is not None:
        synthetic_rows = _extract_topk({"datasets": _synthetic_to_datasets(synthetic)})
        sections["Synthetic sweep (per feature count/top_k)"] = _render_topk(synthetic_rows)
        sections["Synthetic sweep (speed summary)"] = _render_synthetic_speed(synthetic_rows)
        sections["Synthetic sweep (aggregate medians)"] = _render_aggregate(synthetic_rows)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for title, text in sections.items():
            filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
            out_path = out_dir / f"{filename}.md"
            out_path.write_text(text, encoding="utf-8")
        if args.plots:
            _maybe_plot(out_dir, topk_rows)
    else:
        for title, text in sections.items():
            print(f"\n## {title}\n")
            print(text)


if __name__ == "__main__":
    main()
