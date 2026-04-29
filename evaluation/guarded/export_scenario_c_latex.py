"""Export Scenario C guard-retention results to LaTeX tables.

Generates three tables:
  1. Factual task-level summary — three guard metric groups per significance level
     (eps=0.05, 0.10, 0.20), one row per (task, regression-mode) combination.
  2. Alternative task-level summary — same layout for alternative explanations.
  3. Dataset list — one row per non-skipped dataset with task, sample count,
     feature count, and class count where applicable.

Usage::

    python export_scenario_c_latex.py [--results-dir PATH] [--out-dir PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNIFICANCE_LEVELS = (0.05, 0.10, 0.20)
GUARD_METRICS = ("mean_standard_rules_per_instance",
                 "mean_guarded_rules_per_instance", "guard_retention_rate",
                 "fraction_instances_fully_filtered")

_TASK_LABEL = {"binary": "Binary", "multiclass": "Multiclass", "regression": "Regression"}
_MODE_LABEL = {
    "cls": "---", "plain": "plain", "p25": "p25", "p50": "p50", "p75": "p75"
}

_ROW_ORDER = [
    ("binary", "cls"),
    ("multiclass", "cls"),
    ("regression", "plain"),
    ("regression", "p25"),
    ("regression", "p50"),
    ("regression", "p75"),
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_rules(v: Any) -> str:
    """Format rule count (Standard / Guarded): 1 decimal place."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    return f"{v:.1f}"


def _fmt_rate(v: Any) -> str:
    """Format fraction/rate (Retention / Filtered): 2 decimal places."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    return f"{v:.2f}"


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _write_tex(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _booktabs_table(
    caption: str,
    label: str,
    col_format: str,
    header_lines: list[str],
    body_lines: list[str],
) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_format}}}",
        r"\toprule",
    ]
    lines.extend(header_lines)
    lines.append(r"\midrule")
    lines.extend(body_lines)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task-level table
# ---------------------------------------------------------------------------

def _build_task_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to (task, regression_mode, explanation_type, significance) level."""
    cls_mask = (
        df["task"].isin(["binary", "multiclass"])
        & (df["task_skipped"] == False)  # noqa: E712
        & df["regression_mode"].isna()
    )
    df = df.copy()
    df.loc[cls_mask, "regression_mode"] = "cls"

    sub = df[
        (df["task_skipped"] == False)  # noqa: E712
        & df["error"].isna()
        & df["explanation_type"].notna()
    ].copy()

    rows: dict[tuple[str, str, str], dict[str, float]] = {}
    for sig in SIGNIFICANCE_LEVELS:
        sig_sub = sub[np.isclose(sub["significance"], sig)]
        agg = (
            sig_sub
            .groupby(["task", "regression_mode", "explanation_type"])[list(GUARD_METRICS)]
            .mean()
            .reset_index()
        )
        for _, row in agg.iterrows():
            key = (str(row["task"]), str(row["regression_mode"]), str(row["explanation_type"]))
            if key not in rows:
                rows[key] = {}
            for metric in GUARD_METRICS:
                rows[key][(sig, metric)] = float(row[metric])

    if not rows:
        return pd.DataFrame()

    index = pd.MultiIndex.from_tuples(
        list(rows.keys()), names=["task", "regression_mode", "explanation_type"]
    )
    col_index = pd.MultiIndex.from_tuples(
        [(sig, m) for sig in SIGNIFICANCE_LEVELS for m in GUARD_METRICS],
        names=["significance", "metric"],
    )
    data = [[rows.get(k, {}).get((sig, m), float("nan"))
             for sig in SIGNIFICANCE_LEVELS for m in GUARD_METRICS]
            for k in rows]
    return pd.DataFrame(data, index=index, columns=col_index)


def _build_task_level_body(
    table: pd.DataFrame,
    explanation_type: str,
) -> list[str]:
    """Build LaTeX body rows for one explanation type (factual or alternative)."""
    _std_metric = "mean_standard_rules_per_instance"
    body_lines: list[str] = []
    prev_task = None

    for task, mode in _ROW_ORDER:
        key = (task, mode, explanation_type)
        if key not in table.index:
            continue

        task_label = _TASK_LABEL.get(task, task)
        mode_label = _MODE_LABEL.get(mode, mode)
        row_vals = table.loc[key]

        std_val = _fmt_rules(row_vals.get((0.10, _std_metric), float("nan")))
        guard_cells = [
            _fmt_rules(row_vals.get((sig, "mean_guarded_rules_per_instance"), float("nan")))
            for sig in SIGNIFICANCE_LEVELS
        ]
        retention_cells = [
            _fmt_rate(row_vals.get((sig, "guard_retention_rate"), float("nan")))
            for sig in SIGNIFICANCE_LEVELS
        ]
        filtered_cells = [
            _fmt_rate(row_vals.get((sig, "fraction_instances_fully_filtered"), float("nan")))
            for sig in SIGNIFICANCE_LEVELS
        ]

        if task == "regression" and prev_task != "regression":
            body_lines.append(r"\midrule")

        cells = [task_label, mode_label, std_val] + guard_cells + retention_cells + filtered_cells
        body_lines.append(" & ".join(cells) + r" \\")
        prev_task = task

    return body_lines


def _task_level_header() -> tuple[str, list[str]]:
    """Return (col_format, header_lines) for the task-level table."""
    n_sig = len(SIGNIFICANCE_LEVELS)
    metric_names = ("Guarded", "Retention", "Filtered")
    n_display_groups = len(metric_names)

    col_format = "ll r|" + "rrr|" * (n_display_groups - 1) + "rrr"

    header1_parts = [r"\multicolumn{2}{l}{}", r"\multicolumn{1}{c}{Standard}"]
    for i, name in enumerate(metric_names):
        sep = "|" if i < n_display_groups - 1 else ""
        header1_parts.append(
            rf"\multicolumn{{{n_sig}}}{{c{sep}}}{{\textit{{{name}}}}}"
        )
    header1 = " & ".join(header1_parts) + r" \\"

    header2_parts = ["Task", "Mode", ""]
    for _ in metric_names:
        for sig in SIGNIFICANCE_LEVELS:
            header2_parts.append(rf"$\varepsilon\!=\!{sig}$")
    header2 = " & ".join(header2_parts) + r" \\"

    return col_format, [header1, header2]


def write_factual_task_level_latex(table: pd.DataFrame, out_path: Path) -> None:
    """Write the factual task-level guard-retention table."""
    col_format, header_lines = _task_level_header()
    body_lines = _build_task_level_body(table, "factual")

    content = _booktabs_table(
        caption=(
            "Guard retention benchmark for \\textit{factual} explanations across all tasks and "
            "regression modes (averaged over datasets, seeds, and calibration sizes). "
            "\\textit{Standard}: mean CE rules per instance (no guard, at $\\varepsilon=0.10$); "
            "\\textit{Guarded}: mean guarded rules per instance; "
            "\\textit{Retention}: emitted / (emitted + guard-removed) over factual bins "
            "(expected floor: $1-\\varepsilon$); "
            "\\textit{Filtered}: fraction of instances with zero guarded rules."
        ),
        label="tab:guarded_scenario_c_factual",
        col_format=col_format,
        header_lines=header_lines,
        body_lines=body_lines,
    )
    _write_tex(out_path, content)
    print(f"  Wrote factual table: {out_path}")


def write_alternative_task_level_latex(table: pd.DataFrame, out_path: Path) -> None:
    """Write the alternative task-level guard-retention table."""
    col_format, header_lines = _task_level_header()
    body_lines = _build_task_level_body(table, "alternative")

    content = _booktabs_table(
        caption=(
            "Guard retention benchmark for \\textit{alternative} explanations across all tasks "
            "and regression modes (averaged over datasets, seeds, and calibration sizes). "
            "\\textit{Standard}: mean CE rules per instance (no guard, at $\\varepsilon=0.10$); "
            "\\textit{Guarded}: mean guarded rules per instance; "
            "\\textit{Retention}: emitted / (emitted + guard-removed) over alternative bins "
            "(expected floor: $1-\\varepsilon$); "
            "\\textit{Filtered}: fraction of instances with zero guarded rules."
        ),
        label="tab:guarded_scenario_c_alternative",
        col_format=col_format,
        header_lines=header_lines,
        body_lines=body_lines,
    )
    _write_tex(out_path, content)
    print(f"  Wrote alternative table: {out_path}")


# ---------------------------------------------------------------------------
# Dataset list table
# ---------------------------------------------------------------------------

def build_dataset_list_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build a dataset-listing table from the scenario C raw results CSV."""
    from evaluation.ensure.datasets_ensure import (
        load_binary_dataset,
        load_multiclass_dataset,
        load_regression_dataset_from_txt,
    )

    loaders = {
        "binary": load_binary_dataset,
        "multiclass": load_multiclass_dataset,
        "regression": load_regression_dataset_from_txt,
    }

    meta = (
        raw_df[raw_df["task_skipped"] == False]  # noqa: E712
        .drop_duplicates(subset=["task", "dataset"])
        [["task", "dataset", "n_features", "n_classes"]]
        .sort_values(["task", "dataset"])
    )

    rows: list[dict] = []
    for _, r in meta.iterrows():
        task, name = str(r["task"]), str(r["dataset"])
        try:
            ds = loaders[task](name)
            n_samples = int(ds.X.shape[0])
        except Exception:
            n_samples = None
        n_classes = None if pd.isna(r["n_classes"]) else int(r["n_classes"])
        rows.append({
            "Task": _TASK_LABEL.get(task, task),
            "Dataset": name,
            "Samples": n_samples if n_samples is not None else "---",
            "Features": int(r["n_features"]),
            "Classes": n_classes if n_classes is not None else "---",
        })

    if not rows:
        return pd.DataFrame(columns=["Task", "Dataset", "Samples", "Features", "Classes"])
    return pd.DataFrame(rows).set_index("Dataset")


def write_dataset_list_latex(ds_table: pd.DataFrame, out_path: Path) -> None:
    """Write the dataset-listing table as a booktabs LaTeX table."""
    col_format = "llrrr"
    header_lines = [r"Dataset & Task & Samples & Features & Classes \\"]

    body_lines: list[str] = []
    prev_task = None
    for ds_name, row in ds_table.iterrows():
        task = str(row["Task"])
        if prev_task is not None and task != prev_task:
            body_lines.append(r"\midrule")
        body_lines.append(
            f"{ds_name} & {task} & {row['Samples']} & {row['Features']} & {row['Classes']} \\\\"
        )
        prev_task = task

    content = _booktabs_table(
        caption=(
            "Datasets used in the Scenario~C guard-retention benchmark. "
            "\\textit{Classes}: number of target classes for classification tasks; "
            "\\textit{---} for regression."
        ),
        label="tab:guarded_scenario_c_datasets",
        col_format=col_format,
        header_lines=header_lines,
        body_lines=body_lines,
    )
    _write_tex(out_path, content)
    print(f"  Wrote dataset list table: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "guarded" / "scenario_c",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "guarded" / "scenario_c" / "latex_tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_csv = args.results_dir / "scenario_c_raw.csv"

    if not raw_csv.exists():
        print(f"[ERROR] Raw results not found: {raw_csv}")
        sys.exit(1)

    df = pd.read_csv(raw_csv)

    print("Building task-level tables...")
    task_table = _build_task_level_table(df)
    write_factual_task_level_latex(task_table, args.out_dir / "scenario_c_factual.tex")
    write_alternative_task_level_latex(task_table, args.out_dir / "scenario_c_alternative.tex")

    print("Building dataset list table...")
    ds_table = build_dataset_list_table(df)
    write_dataset_list_latex(ds_table, args.out_dir / "scenario_c_datasets.tex")

    print(f"\nDone. Tables written under {args.out_dir}")


if __name__ == "__main__":
    main()
