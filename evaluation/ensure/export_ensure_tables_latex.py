"""Export ensured evaluation results to LaTeX tables.

Generates per-dataset tables, aggregated tables, a master table combining all
tasks, and per-task dataset listing tables.

Evaluation-only code (ADR-010).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


# NOTE: VS Code's Python execution wrapper may not include the repo root on
# sys.path when running scripts. Add it for consistency.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


CANON_COLUMNS = [
    "Total",
    "CoFa",
    "CoPo",
    "SeFa",
    "SePo",
    "SuFa",
    "SuPo",
    "Ens",
    "Pareto",
]


RANKVAL_COLUMNS = [
    "Spearman(S,U)",
    "Spearman(S,P)",
]


def _counts_to_row(counts_mean: dict[str, float]) -> dict[str, float]:
    return {
        "Total": float(counts_mean.get("total", np.nan)),
        "CoFa": float(counts_mean.get("counterfactual", np.nan)),
        "CoPo": float(counts_mean.get("counterpotential", np.nan)),
        "SeFa": float(counts_mean.get("semifactual", np.nan)),
        "SePo": float(counts_mean.get("semipotential", np.nan)),
        "SuFa": float(counts_mean.get("superfactual", np.nan)),
        "SuPo": float(counts_mean.get("superpotential", np.nan)),
        "Ens": float(counts_mean.get("ensured", np.nan)),
        "Pareto": float(counts_mean.get("pareto", np.nan)),
    }


def _is_skipped(ds_result: dict[str, Any]) -> bool:
    return ds_result.get("meta", {}).get("skipped", False)


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def build_table_for_dataset(task_result: dict[str, Any]) -> pd.DataFrame:
    """Build a table for a single dataset (binary/multiclass)."""

    by_cal = task_result["by_calibration_size"]
    rows: dict[str, dict[str, float]] = {}

    for cal_size in sorted(by_cal.keys(), reverse=False):
        explore = by_cal[cal_size]["explore"]
        conj = by_cal[cal_size]["conjugate"]
        rows[f"{cal_size} (s)"] = _counts_to_row(explore["counts_mean"])
        rows[f"{cal_size} (c)"] = _counts_to_row(conj["counts_mean"])

    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.reindex(columns=CANON_COLUMNS)


def build_table_for_regression_setting(
    setting: dict[str, Any],
) -> pd.DataFrame:
    """Build a table for one regression setting (plain or one threshold)."""

    by_cal = setting["by_calibration_size"]
    rows: dict[str, dict[str, float]] = {}

    for cal_size in sorted(by_cal.keys(), reverse=False):
        explore = by_cal[cal_size]["explore"]
        conj = by_cal[cal_size]["conjugate"]
        rows[f"{cal_size} (s)"] = _counts_to_row(explore["counts_mean"])
        rows[f"{cal_size} (c)"] = _counts_to_row(conj["counts_mean"])

    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.reindex(columns=CANON_COLUMNS)


def _mean_std_cell(
    values: Iterable[float], *, show_std: bool = False
) -> str:
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return ""
    if show_std:
        return f"{np.mean(v):.2f} ({np.std(v):.2f})"
    return f"{np.mean(v):.2f}"


def build_aggregated_table(
    tables: dict[str, pd.DataFrame],
    *,
    show_std: bool = False,
) -> pd.DataFrame:
    """Aggregate multiple per-dataset tables into a summary."""

    if not tables:
        return pd.DataFrame(columns=CANON_COLUMNS)

    all_index = sorted(
        {idx for t in tables.values() for idx in t.index}
    )
    agg_rows: dict[str, dict[str, str]] = {}

    for idx in all_index:
        agg_rows[idx] = {}
        for col in CANON_COLUMNS:
            vals = []
            for t in tables.values():
                if idx in t.index:
                    vals.append(t.loc[idx, col])
                else:
                    vals.append(np.nan)
            agg_rows[idx][col] = _mean_std_cell(
                vals, show_std=show_std
            )

    return pd.DataFrame.from_dict(agg_rows, orient="index")


def write_table(
    df: pd.DataFrame,
    out_path: Path,
    *,
    caption: str,
    label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_numeric = all(
        is_numeric_dtype(df[col]) for col in df.columns
    )
    to_latex_kwargs = dict(
        index=True,
        escape=True,
        caption=caption,
        label=label,
        bold_rows=False,
        column_format="l" + "c" * len(df.columns),
        longtable=False,
    )
    if all_numeric:
        to_latex_kwargs["float_format"] = lambda x: f"{x:.2f}"

    latex = df.to_latex(**to_latex_kwargs)
    out_path.write_text(latex, encoding="utf-8")


def _format_mean_std(values: list[float | None], *, show_std: bool) -> str:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return ""
    arr = np.asarray(vals, dtype=float)
    if show_std:
        return f"{float(np.mean(arr)):.3f} ({float(np.std(arr)):.3f})"
    return f"{float(np.mean(arr)):.3f}"


def build_ranking_validation_table(
    results: dict[str, Any],
    *,
    cal_size: int,
    mode: str,
    show_std: bool,
) -> pd.DataFrame:
    """Aggregate ranking-validation summaries across datasets.

    Parameters
    ----------
    results:
        The task-level results pickle content.
    cal_size:
        Calibration size row to extract (e.g., 100).
    mode:
        Either "explore" or "conjugate".
    show_std:
        Whether to format cells as mean (std).
    """

    cfg = results.get("config", {})
    weights = [float(w) for w in cfg.get("ranking_weights", [-1.0, -0.5, 0.0, 0.5, 1.0])]

    # Collect dataset-level per-weight metrics.
    per_weight: dict[str, dict[str, list[float | None]]] = {
        str(float(w)): {c: [] for c in RANKVAL_COLUMNS} for w in weights
    }

    for _, ds_result in results.get("results", {}).items():
        if _is_skipped(ds_result):
            continue

        by_cal = ds_result.get("by_calibration_size")
        if not by_cal or int(cal_size) not in by_cal:
            continue

        rv = by_cal[int(cal_size)].get(mode, {}).get("ranking_validation", {})
        per_w = rv.get("per_w", {})

        for w in weights:
            w_key = str(float(w))
            row = per_w.get(w_key)
            if row is None:
                continue
            per_weight[w_key]["Spearman(S,U)"].append(row.get("spearman_score_uncertainty"))
            per_weight[w_key]["Spearman(S,P)"].append(row.get("spearman_score_prediction"))

    rows: dict[str, dict[str, str]] = {}
    for w in weights:
        w_key = str(float(w))
        rows[w_key] = {
            "Spearman(S,U)": _format_mean_std(per_weight[w_key]["Spearman(S,U)"], show_std=show_std),
            "Spearman(S,P)": _format_mean_std(per_weight[w_key]["Spearman(S,P)"], show_std=show_std),
        }

    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=RANKVAL_COLUMNS)


def build_pareto_consistency_table(
    results: dict[str, Any],
    *,
    cal_size: int,
    mode: str,
    show_std: bool,
) -> pd.DataFrame:
    """Aggregate Pareto-consistency (top-ranked on Pareto frontier) across datasets.

    Only includes representative weights with |w| < 1 (e.g., -0.5, 0.0, 0.5).
    Returns a DataFrame with one column 'Pareto-cons. (%)' showing mean percent.
    """

    cfg = results.get("config", {})
    weights = [float(w) for w in cfg.get("ranking_weights", [-1.0, -0.5, 0.0, 0.5, 1.0])]
    # Keep only representative interior weights (exclude pure +/-1.0)
    rep_weights = [w for w in weights if abs(w) < 1.0]

    per_weight: dict[str, list[float | None]] = {str(float(w)): [] for w in rep_weights}

    for _, ds_result in results.get("results", {}).items():
        if _is_skipped(ds_result):
            continue

        by_cal = ds_result.get("by_calibration_size")
        if not by_cal or int(cal_size) not in by_cal:
            continue

        rv = by_cal[int(cal_size)].get(mode, {}).get("ranking_validation", {})
        per_w = rv.get("per_w", {})

        for w in rep_weights:
            w_key = str(float(w))
            row = per_w.get(w_key)
            if row is None:
                per_weight[w_key].append(None)
                continue
            # Expect a value in [0,1]; convert to percent for presentation
            val = row.get("pareto_consistency_pct")
            per_weight[w_key].append(float(val) * 100.0 if val is not None and not (isinstance(val, float) and np.isnan(val)) else None)

    rows: dict[str, dict[str, str]] = {}
    for w in rep_weights:
        w_key = str(float(w))
        rows[w_key] = {
            "Pareto-cons. (%)": _format_mean_std(per_weight[w_key], show_std=show_std),
        }

    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=["Pareto-cons. (%)"])


def _build_dataset_list_table(
    results: dict[str, Any],
    *,
    include_classes: bool = False,
) -> pd.DataFrame:
    """Build a table listing non-skipped datasets with basic metadata."""
    rows: list[dict[str, Any]] = []
    for ds_name, ds_result in results.items():
        if _is_skipped(ds_result):
            continue
        meta = ds_result.get("meta", {})
        row: dict[str, Any] = {
            "Dataset": meta.get("name", ds_name),
            "Samples": meta.get("n_samples", ""),
            "Features": meta.get("n_features", ""),
        }
        if include_classes:
            row["Classes"] = meta.get("n_classes", "")
        rows.append(row)
    if not rows:
        cols = ["Dataset", "Samples", "Features"]
        if include_classes:
            cols.append("Classes")
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).set_index("Dataset")


def _extract_agg_row(
    agg_table: pd.DataFrame,
    row_key: str,
) -> dict[str, str]:
    """Extract a single row from an aggregated table as a dict."""
    if row_key in agg_table.index:
        return agg_table.loc[row_key].to_dict()
    return {col: "" for col in CANON_COLUMNS}


def _build_master_table(
    *,
    binary_agg: pd.DataFrame | None,
    multiclass_agg: pd.DataFrame | None,
    reg_plain_agg: pd.DataFrame | None,
    reg_p25_agg: pd.DataFrame | None,
    reg_p50_agg: pd.DataFrame | None,
    reg_p75_agg: pd.DataFrame | None,
    cal_size: int = 100,
) -> pd.DataFrame:
    """Build the master aggregated table across all tasks."""
    empty = pd.DataFrame(columns=CANON_COLUMNS)
    s_key = f"{cal_size} (s)"
    c_key = f"{cal_size} (c)"

    binary_src = binary_agg if binary_agg is not None else empty
    multiclass_src = multiclass_agg if multiclass_agg is not None else empty
    reg_plain_src = reg_plain_agg if reg_plain_agg is not None else empty
    reg_p25_src = reg_p25_agg if reg_p25_agg is not None else empty
    reg_p50_src = reg_p50_agg if reg_p50_agg is not None else empty
    reg_p75_src = reg_p75_agg if reg_p75_agg is not None else empty

    sources = [
        ("s binary", binary_src, s_key),
        ("c binary", binary_src, c_key),
        ("s multi.", multiclass_src, s_key),
        ("c multi.", multiclass_src, c_key),
        ("s 25 regr.", reg_p25_src, s_key),
        ("c 25 regr.", reg_p25_src, c_key),
        ("s 50 regr.", reg_p50_src, s_key),
        ("c 50 regr.", reg_p50_src, c_key),
        ("s 75 regr.", reg_p75_src, s_key),
        ("c 75 regr.", reg_p75_src, c_key),
        ("s regr.", reg_plain_src, s_key),
        ("c regr.", reg_plain_src, c_key),
    ]

    rows: dict[str, dict[str, str]] = {}
    for label, agg, key in sources:
        rows[label] = _extract_agg_row(agg, key)

    return pd.DataFrame.from_dict(rows, orient="index").reindex(
        columns=CANON_COLUMNS
    )


SENSITIVITY_COLUMNS = [
    "Total(100)", "Total(300)", "Total(500)",
    "Ens(100)", "Ens(300)", "Ens(500)",
]


def _build_calibration_sensitivity_table(
    *,
    binary_agg: pd.DataFrame | None,
    multiclass_agg: pd.DataFrame | None,
    reg_plain_agg: pd.DataFrame | None,
    reg_p25_agg: pd.DataFrame | None,
    reg_p50_agg: pd.DataFrame | None,
    reg_p75_agg: pd.DataFrame | None,
    cal_sizes: tuple[int, ...] = (100, 300, 500),
) -> pd.DataFrame:
    """Build a calibration-size sensitivity table (RQ3).

    Same rows as the master table, but columns show Total and Ens at each
    calibration size.
    """
    empty = pd.DataFrame(columns=CANON_COLUMNS)

    task_sources = [
        ("s binary", binary_agg, "s"),
        ("c binary", binary_agg, "c"),
        ("s multi.", multiclass_agg, "s"),
        ("c multi.", multiclass_agg, "c"),
        ("s 25 regr.", reg_p25_agg, "s"),
        ("c 25 regr.", reg_p25_agg, "c"),
        ("s 50 regr.", reg_p50_agg, "s"),
        ("c 50 regr.", reg_p50_agg, "c"),
        ("s 75 regr.", reg_p75_agg, "s"),
        ("c 75 regr.", reg_p75_agg, "c"),
        ("s regr.", reg_plain_agg, "s"),
        ("c regr.", reg_plain_agg, "c"),
    ]

    rows: dict[str, dict[str, str]] = {}
    for label, agg, mode in task_sources:
        src = agg if agg is not None else empty
        row: dict[str, str] = {}
        for cs in cal_sizes:
            key = f"{cs} ({mode})"
            if key in src.index:
                row[f"Total({cs})"] = str(src.loc[key, "Total"])
                row[f"Ens({cs})"] = str(src.loc[key, "Ens"])
            else:
                row[f"Total({cs})"] = ""
                row[f"Ens({cs})"] = ""
        rows[label] = row

    return pd.DataFrame.from_dict(rows, orient="index").reindex(
        columns=SENSITIVITY_COLUMNS
    )


def _rv_from_setting(
    reg: dict[str, Any],
    *,
    out_dir: Path,
    cal_size: int,
    show_std: bool,
    key: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Export ranking-validation and Pareto-consistency tables for one
    regression setting and return the two Pareto-consistency DataFrames.

    This mirrors the previous nested helper but is top-level for clarity.
    """
    if not reg:
        return None, None

    pseudo_results = {"config": reg.get("config", {}), "results": {}}
    for ds_name, ds_result in reg.get("results", {}).items():
        if _is_skipped(ds_result):
            continue
        if key == "plain":
            setting = ds_result.get("plain")
        else:
            setting = ds_result.get("probabilistic", {}).get(key)
        if not setting or "by_calibration_size" not in setting:
            continue
        pseudo_results["results"][ds_name] = {
            "meta": ds_result.get("meta", {}),
            "by_calibration_size": setting["by_calibration_size"],
        }

    # Ranking validation (single + conjunctive) and Pareto-consistency
    rv_single = build_ranking_validation_table(
        pseudo_results, cal_size=cal_size, mode="explore", show_std=show_std
    )
    write_table(
        rv_single,
        out_dir / "regression" / f"ensure_regression_{key}_ranking_validation_single.tex",
        caption=f"Regression ({key}) - Ranking validation (single-feature alternatives).",
        label=f"tab:ensure_reg_{key}_rankval_single",
    )
    pc_single = build_pareto_consistency_table(pseudo_results, cal_size=cal_size, mode="explore", show_std=show_std)
    write_table(
        pc_single,
        out_dir / "regression" / f"ensure_regression_{key}_pareto_consistency_single.tex",
        caption=f"Regression ({key}) - Pareto-consistency of top-ranked candidate (single-feature).",
        label=f"tab:ensure_reg_{key}_pareto_single",
    )

    rv_conj = build_ranking_validation_table(
        pseudo_results, cal_size=cal_size, mode="conjugate", show_std=show_std
    )
    write_table(
        rv_conj,
        out_dir / "regression" / f"ensure_regression_{key}_ranking_validation_conjunctive.tex",
        caption=f"Regression ({key}) - Ranking validation (conjunctive alternatives).",
        label=f"tab:ensure_reg_{key}_rankval_conj",
    )
    pc_conj = build_pareto_consistency_table(pseudo_results, cal_size=cal_size, mode="conjugate", show_std=show_std)
    write_table(
        pc_conj,
        out_dir / "regression" / f"ensure_regression_{key}_pareto_consistency_conjunctive.tex",
        caption=f"Regression ({key}) - Pareto-consistency of top-ranked candidate (conjunctive).",
        label=f"tab:ensure_reg_{key}_pareto_conj",
    )

    return pc_single, pc_conj


def _build_master_pareto_table(
    *,
    binary_pc_single: pd.DataFrame | None,
    binary_pc_conj: pd.DataFrame | None,
    multiclass_pc_single: pd.DataFrame | None,
    multiclass_pc_conj: pd.DataFrame | None,
    reg_plain_pc_single: pd.DataFrame | None,
    reg_plain_pc_conj: pd.DataFrame | None,
    reg_p25_pc_single: pd.DataFrame | None,
    reg_p25_pc_conj: pd.DataFrame | None,
    reg_p50_pc_single: pd.DataFrame | None,
    reg_p50_pc_conj: pd.DataFrame | None,
    reg_p75_pc_single: pd.DataFrame | None,
    reg_p75_pc_conj: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compose a master Pareto-consistency table with rows for each
    task/mode and columns for representative weights (e.g., -0.5, 0.0, 0.5).
    """

    sources = [
        ("s binary", binary_pc_single),
        ("c binary", binary_pc_conj),
        ("s multi.", multiclass_pc_single),
        ("c multi.", multiclass_pc_conj),
        ("s 25 regr.", reg_p25_pc_single),
        ("c 25 regr.", reg_p25_pc_conj),
        ("s 50 regr.", reg_p50_pc_single),
        ("c 50 regr.", reg_p50_pc_conj),
        ("s 75 regr.", reg_p75_pc_single),
        ("c 75 regr.", reg_p75_pc_conj),
        ("s regr.", reg_plain_pc_single),
        ("c regr.", reg_plain_pc_conj),
    ]

    # Use standard representative weight keys
    weight_keys = ["-0.5", "0.0", "0.5"]
    rows: dict[str, dict[str, str]] = {}
    for label, df in sources:
        row: dict[str, str] = {}
        if df is None or df.empty:
            for wk in weight_keys:
                row[wk] = ""
        else:
            for wk in weight_keys:
                if wk in df.index and "Pareto-cons. (%)" in df.columns:
                    row[wk] = str(df.loc[wk, "Pareto-cons. (%)"]) if df.loc[wk, "Pareto-cons. (%)"] != "" else ""
                else:
                    row[wk] = ""
        rows[label] = row

    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=weight_keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "results"
            / "results_ensure_binary.pkl"
        ),
    )
    parser.add_argument(
        "--multiclass",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "results"
            / "results_ensure_multiclass.pkl"
        ),
    )
    parser.add_argument(
        "--regression",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "results"
            / "results_ensure_regression.pkl"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "latex_tables"),
    )
    parser.add_argument(
        "--show-std",
        action="store_true",
        default=False,
        help="Include standard deviation in aggregated tables",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    show_std: bool = args.show_std

    binary_agg: pd.DataFrame | None = None
    multiclass_agg: pd.DataFrame | None = None
    reg_plain_agg: pd.DataFrame | None = None
    reg_p25_agg: pd.DataFrame | None = None
    reg_p50_agg: pd.DataFrame | None = None
    reg_p75_agg: pd.DataFrame | None = None
    # Pareto-consistency aggregated tables (per-task)
    binary_pc_single: pd.DataFrame | None = None
    binary_pc_conj: pd.DataFrame | None = None
    multiclass_pc_single: pd.DataFrame | None = None
    multiclass_pc_conj: pd.DataFrame | None = None
    reg_plain_pc_single: pd.DataFrame | None = None
    reg_plain_pc_conj: pd.DataFrame | None = None
    reg_p25_pc_single: pd.DataFrame | None = None
    reg_p25_pc_conj: pd.DataFrame | None = None
    reg_p50_pc_single: pd.DataFrame | None = None
    reg_p50_pc_conj: pd.DataFrame | None = None
    reg_p75_pc_single: pd.DataFrame | None = None
    reg_p75_pc_conj: pd.DataFrame | None = None

    # Binary
    binary_path = Path(args.binary)
    if binary_path.exists():
        binary = load_pickle(binary_path)
        per_ds_tables: dict[str, pd.DataFrame] = {}
        for ds_name, ds_result in binary["results"].items():
            if _is_skipped(ds_result):
                continue
            table = build_table_for_dataset(ds_result)
            per_ds_tables[ds_name] = table
            write_table(
                table,
                out_dir / "binary" / f"ensure_binary_{ds_name}.tex",
                caption=(
                    f"{ds_name} - Number of alternative "
                    f"explanations (binary)."
                ),
                label=f"tab:ensure_binary_{ds_name}",
            )
        binary_agg = build_aggregated_table(
            per_ds_tables, show_std=show_std
        )
        write_table(
            binary_agg,
            out_dir / "binary" / "ensure_binary_aggregated.tex",
            caption=(
                "Binary classification - Number of alternative "
                "explanations (mean across datasets)."
            ),
            label="tab:ensure_binary_agg",
        )
        ds_list = _build_dataset_list_table(
            binary["results"], include_classes=True
        )
        write_table(
            ds_list,
            out_dir / "binary" / "ensure_binary_datasets.tex",
            caption="Binary classification datasets used.",
            label="tab:ensure_binary_datasets",
        )

        # Ranking validation (calibration size 100 by default)
        cal_size = int(binary.get("config", {}).get("calibration_sizes", [100])[0])
        rv_single = build_ranking_validation_table(
            binary, cal_size=cal_size, mode="explore", show_std=show_std
        )
        write_table(
            rv_single,
            out_dir / "binary" / "ensure_binary_ranking_validation_single.tex",
            caption="Binary classification - Ranking validation (single-feature alternatives).",
            label="tab:ensure_binary_rankval_single",
        )
        # Pareto-consistency table (representative weights |w|<1)
        pc_single = build_pareto_consistency_table(binary, cal_size=cal_size, mode="explore", show_std=show_std)
        write_table(
            pc_single,
            out_dir / "binary" / "ensure_binary_pareto_consistency_single.tex",
            caption="Binary classification - Pareto-consistency of top-ranked candidate (single-feature).",
            label="tab:ensure_binary_pareto_single",
        )
        binary_pc_single = pc_single
        rv_conj = build_ranking_validation_table(
            binary, cal_size=cal_size, mode="conjugate", show_std=show_std
        )
        write_table(
            rv_conj,
            out_dir / "binary" / "ensure_binary_ranking_validation_conjunctive.tex",
            caption="Binary classification - Ranking validation (conjunctive alternatives).",
            label="tab:ensure_binary_rankval_conj",
        )
        pc_conj = build_pareto_consistency_table(binary, cal_size=cal_size, mode="conjugate", show_std=show_std)
        write_table(
            pc_conj,
            out_dir / "binary" / "ensure_binary_pareto_consistency_conjunctive.tex",
            caption="Binary classification - Pareto-consistency of top-ranked candidate (conjunctive).",
            label="tab:ensure_binary_pareto_conj",
        )
        binary_pc_conj = pc_conj

    # Multiclass
    multi_path = Path(args.multiclass)
    if multi_path.exists():
        multi = load_pickle(multi_path)
        per_ds_tables = {}
        for ds_name, ds_result in multi["results"].items():
            if _is_skipped(ds_result):
                continue
            table = build_table_for_dataset(ds_result)
            per_ds_tables[ds_name] = table
            write_table(
                table,
                out_dir
                / "multiclass"
                / f"ensure_multiclass_{ds_name}.tex",
                caption=(
                    f"{ds_name} - Number of alternative "
                    f"explanations (multiclass)."
                ),
                label=f"tab:ensure_multiclass_{ds_name}",
            )
        multiclass_agg = build_aggregated_table(
            per_ds_tables, show_std=show_std
        )
        write_table(
            multiclass_agg,
            out_dir
            / "multiclass"
            / "ensure_multiclass_aggregated.tex",
            caption=(
                "Multiclass classification - Number of alternative "
                "explanations (mean across datasets)."
            ),
            label="tab:ensure_multiclass_agg",
        )
        ds_list = _build_dataset_list_table(
            multi["results"], include_classes=True
        )
        write_table(
            ds_list,
            out_dir
            / "multiclass"
            / "ensure_multiclass_datasets.tex",
            caption="Multiclass classification datasets used.",
            label="tab:ensure_multiclass_datasets",
        )

        cal_size = int(multi.get("config", {}).get("calibration_sizes", [100])[0])
        rv_single = build_ranking_validation_table(
            multi, cal_size=cal_size, mode="explore", show_std=show_std
        )
        write_table(
            rv_single,
            out_dir / "multiclass" / "ensure_multiclass_ranking_validation_single.tex",
            caption="Multiclass classification - Ranking validation (single-feature alternatives).",
            label="tab:ensure_multiclass_rankval_single",
        )
        pc_single = build_pareto_consistency_table(multi, cal_size=cal_size, mode="explore", show_std=show_std)
        write_table(
            pc_single,
            out_dir / "multiclass" / "ensure_multiclass_pareto_consistency_single.tex",
            caption="Multiclass classification - Pareto-consistency of top-ranked candidate (single-feature).",
            label="tab:ensure_multiclass_pareto_single",
        )
        multiclass_pc_single = pc_single
        rv_conj = build_ranking_validation_table(
            multi, cal_size=cal_size, mode="conjugate", show_std=show_std
        )
        write_table(
            rv_conj,
            out_dir / "multiclass" / "ensure_multiclass_ranking_validation_conjunctive.tex",
            caption="Multiclass classification - Ranking validation (conjunctive alternatives).",
            label="tab:ensure_multiclass_rankval_conj",
        )
        pc_conj = build_pareto_consistency_table(multi, cal_size=cal_size, mode="conjugate", show_std=show_std)
        write_table(
            pc_conj,
            out_dir / "multiclass" / "ensure_multiclass_pareto_consistency_conjunctive.tex",
            caption="Multiclass classification - Pareto-consistency of top-ranked candidate (conjunctive).",
            label="tab:ensure_multiclass_pareto_conj",
        )
        multiclass_pc_conj = pc_conj

    # Regression
    reg_path = Path(args.regression)
    if reg_path.exists():
        reg = load_pickle(reg_path)

        # Plain regression
        per_ds_tables_plain: dict[str, pd.DataFrame] = {}
        for ds_name, ds_result in reg["results"].items():
            if _is_skipped(ds_result):
                continue
            table = build_table_for_regression_setting(
                ds_result["plain"]
            )
            per_ds_tables_plain[ds_name] = table
            write_table(
                table,
                out_dir
                / "regression"
                / f"ensure_regression_plain_{ds_name}.tex",
                caption=(
                    f"{ds_name} - Number of alternative "
                    f"explanations (plain regression)."
                ),
                label=f"tab:ensure_reg_plain_{ds_name}",
            )
        reg_plain_agg = build_aggregated_table(
            per_ds_tables_plain, show_std=show_std
        )
        write_table(
            reg_plain_agg,
            out_dir
            / "regression"
            / "ensure_regression_plain_aggregated.tex",
            caption=(
                "Plain regression - Number of alternative "
                "explanations (mean across datasets)."
            ),
            label="tab:ensure_reg_plain_agg",
        )

        # Probabilistic regression (per percentile key)
        threshold_keys = ["p25", "p50", "p75"]
        reg_prob_aggs: dict[str, pd.DataFrame] = {}
        for key in threshold_keys:
            per_ds_tables_thr: dict[str, pd.DataFrame] = {}
            for ds_name, ds_result in reg["results"].items():
                if _is_skipped(ds_result):
                    continue
                prob = ds_result["probabilistic"].get(key)
                if prob is None:
                    continue
                table = build_table_for_regression_setting(prob)
                per_ds_tables_thr[ds_name] = table
                write_table(
                    table,
                    out_dir
                    / "regression"
                    / f"ensure_regression_{key}_{ds_name}.tex",
                    caption=(
                        f"{ds_name} - Number of alternative "
                        f"explanations (probabilistic, {key})."
                    ),
                    label=f"tab:ensure_reg_{key}_{ds_name}",
                )
            agg_thr = build_aggregated_table(
                per_ds_tables_thr, show_std=show_std
            )
            reg_prob_aggs[key] = agg_thr
            write_table(
                agg_thr,
                out_dir
                / "regression"
                / f"ensure_regression_{key}_aggregated.tex",
                caption=(
                    f"Probabilistic regression ({key}) - Number "
                    f"of alternative explanations "
                    f"(mean across datasets)."
                ),
                label=f"tab:ensure_reg_{key}_agg",
            )
        reg_p25_agg = reg_prob_aggs.get("p25")
        reg_p50_agg = reg_prob_aggs.get("p50")
        reg_p75_agg = reg_prob_aggs.get("p75")

        # Regression dataset listing
        ds_list = _build_dataset_list_table(
            reg["results"], include_classes=False
        )
        write_table(
            ds_list,
            out_dir
            / "regression"
            / "ensure_regression_datasets.tex",
            caption="Regression datasets used.",
            label="tab:ensure_regression_datasets",
        )

        # Ranking validation tables for regression are exported separately per setting.
        cal_size = int(reg.get("config", {}).get("calibration_sizes", [100])[0])

        reg_plain_pc_single, reg_plain_pc_conj = _rv_from_setting(
            reg, out_dir=out_dir, cal_size=cal_size, show_std=show_std, key="plain"
        )
        reg_p25_pc_single, reg_p25_pc_conj = _rv_from_setting(
            reg, out_dir=out_dir, cal_size=cal_size, show_std=show_std, key="p25"
        )
        reg_p50_pc_single, reg_p50_pc_conj = _rv_from_setting(
            reg, out_dir=out_dir, cal_size=cal_size, show_std=show_std, key="p50"
        )
        reg_p75_pc_single, reg_p75_pc_conj = _rv_from_setting(
            reg, out_dir=out_dir, cal_size=cal_size, show_std=show_std, key="p75"
        )

    # Master aggregated table
    master = _build_master_table(
        binary_agg=binary_agg,
        multiclass_agg=multiclass_agg,
        reg_plain_agg=reg_plain_agg,
        reg_p25_agg=reg_p25_agg,
        reg_p50_agg=reg_p50_agg,
        reg_p75_agg=reg_p75_agg,
    )
    write_table(
        master,
        out_dir / "ensure_master_aggregated.tex",
        caption=(
            "Aggregated number of alternative explanations "
            "across all tasks."
        ),
        label="tab:ensure_master_agg",
    )

    # Calibration-size sensitivity table (RQ3)
    sensitivity = _build_calibration_sensitivity_table(
        binary_agg=binary_agg,
        multiclass_agg=multiclass_agg,
        reg_plain_agg=reg_plain_agg,
        reg_p25_agg=reg_p25_agg,
        reg_p50_agg=reg_p50_agg,
        reg_p75_agg=reg_p75_agg,
    )
    write_table(
        sensitivity,
        out_dir / "ensure_calibration_sensitivity.tex",
        caption=(
            "Total and ensured alternative explanations as a function "
            "of calibration-set size (mean across datasets)."
        ),
        label="tab:ensure_cal_sensitivity",
    )

    # Master Pareto-consistency table across tasks/modes
    master_pareto = _build_master_pareto_table(
        binary_pc_single=binary_pc_single,
        binary_pc_conj=binary_pc_conj,
        multiclass_pc_single=multiclass_pc_single,
        multiclass_pc_conj=multiclass_pc_conj,
        reg_plain_pc_single=reg_plain_pc_single,
        reg_plain_pc_conj=reg_plain_pc_conj,
        reg_p25_pc_single=reg_p25_pc_single,
        reg_p25_pc_conj=reg_p25_pc_conj,
        reg_p50_pc_single=reg_p50_pc_single,
        reg_p50_pc_conj=reg_p50_pc_conj,
        reg_p75_pc_single=reg_p75_pc_single,
        reg_p75_pc_conj=reg_p75_pc_conj,
    )
    write_table(
        master_pareto,
        out_dir / "ensure_master_pareto_consistency.tex",
        caption=(
            "Aggregated Pareto-consistency (\%) of top-ranked candidates "
            "across tasks and modes (weights -0.5, 0.0, 0.5)."
        ),
        label="tab:ensure_master_pareto",
    )

    print(f"Wrote LaTeX tables under {out_dir}")


if __name__ == "__main__":
    main()
