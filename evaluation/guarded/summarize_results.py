"""Consolidated results summary for the guarded explanation evaluation suite.

Reads all scenario artifacts from previously-run scenarios, prints a
dashboard-style overview to stdout, and writes consolidated_summary.md to
the artifacts directory.

Missing scenario artifacts are silently skipped — you can run this after
any subset of scenarios.

Usage:
    python summarize_results.py
    python summarize_results.py --artifacts-dir path/to/artifacts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ANSI helpers (auto-disabled when not a TTY)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def red(t: str) -> str:   return _c(t, "31")
def green(t: str) -> str: return _c(t, "32")
def yellow(t: str) -> str: return _c(t, "33")
def bold(t: str) -> str:  return _c(t, "1")
def dim(t: str) -> str:   return _c(t, "2")


def status_marker(ok: bool, warn: bool = False) -> str:
    if ok:
        return green("✓")
    if warn:
        return yellow("!")
    return red("✗")


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

def _artifact_dirs(base: Path) -> dict:
    return {
        "a": base.parent / "guarded_vs_standard" / "scenario_a",
        "b": base / "scenario_b",
        "c": base / "scenario_c",
        "d": base / "scenario_d",
        "e": base / "scenario_e",
    }


def _load(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except (OSError, pd.errors.ParserError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Parameter extraction helpers
# ---------------------------------------------------------------------------

def _param_grid(df: pd.DataFrame, columns: List[str]) -> str:
    """Return 'col=values, col=values' for each column present in df."""
    parts = []
    for col in columns:
        if col in df.columns:
            vals = sorted(df[col].dropna().unique().tolist())
            vals_str = ", ".join(str(v) for v in vals)
            parts.append(f"{col}=[{vals_str}]")
    return "  ".join(parts) if parts else "—"


# ---------------------------------------------------------------------------
# Failure collection
# ---------------------------------------------------------------------------

class Failure:
    def __init__(self, scenario: str, description: str, detail: str = ""):
        self.scenario = scenario
        self.description = description
        self.detail = detail

    def __str__(self) -> str:
        d = f" → {self.detail}" if self.detail else ""
        return f"[{self.scenario}] {self.description}{d}"


# ---------------------------------------------------------------------------
# Scenario-specific summary builders
# ---------------------------------------------------------------------------

def _summarize_scenario_a(dirs: dict, failures: List[Failure]) -> Tuple[List[str], List[str]]:
    """Returns (terminal_lines, markdown_lines)."""
    d = dirs["a"]
    summary = _load(d / "summary_metrics.csv")
    metrics = _load(d / "metrics_records.csv")

    term: List[str] = [bold("  Scenario A — Domain plausibility")]
    md: List[str] = ["### Scenario A — Domain plausibility", ""]

    if summary is None:
        msg = dim("  Not run yet — no artifacts found.")
        term.append(msg)
        md.append("*Not run yet — no artifacts found.*")
        return term, md

    # Parameters used
    if metrics is not None:
        params = _param_grid(metrics, ["significance", "n_neighbors", "merge_adjacent"])
        seeds = metrics["seed"].nunique() if "seed" in metrics.columns else "?"
        models = list(metrics["model"].unique()) if "model" in metrics.columns else []
        term.append(f"  Params: seeds={seeds}  models={models}  {params}")
        md.append(f"**Params:** seeds={seeds}, models={models}, {params}")
        md.append("")

    # Key metric: violation_rate guarded vs standard
    vr = summary[summary["metric"] == "violation_rate"] if "metric" in summary.columns else pd.DataFrame()
    if not vr.empty:
        term.append(f"  {'sig':>6}  {'mode':<12}  {'guarded':>9}  {'standard':>9}  {'Δ':>8}  {'p-val':>8}  {'ok?':>4}")
        term.append("  " + "-" * 66)
        md_rows = ["| sig | mode | guarded | standard | Δ (g−s) | wilcoxon_p | ok? |",
                   "|---|---|---|---|---|---|---|"]
        for _, row in vr.sort_values(["mode", "significance"]).iterrows():
            g = row.get("guarded_mean", float("nan"))
            s = row.get("standard_mean", float("nan"))
            delta = row.get("median_difference_guarded_minus_standard", float("nan"))
            p = row.get("wilcoxon_p_value", float("nan"))
            sig = row.get("significance", "?")
            mode = row.get("mode", "?")
            # Use means for pass/fail: median diff is 0 for sparse metrics like
            # violation_rate where most instances have 0 for both methods, so
            # median(guarded - standard) = 0 even when means differ significantly.
            # Require both: guarded mean is lower AND the difference is statistically
            # significant (Wilcoxon p < 0.05). Ties (g == s, both 0) are a pass.
            if np.isnan(g) or np.isnan(s):
                ok = None
            elif np.isclose(g, s):
                ok = True   # both zero — guard not needed, not a failure
            else:
                sig_stat = (not np.isnan(p)) and (p < 0.05)
                ok = (g < s) if sig_stat else None  # warn if improvement not significant
            marker = status_marker(ok if ok is not None else False) if ok is not None else dim("?")
            md_ok = "✓" if ok else ("?" if ok is None else "✗")
            term.append(f"  {sig:>6.2f}  {mode:<12}  {g:>9.4f}  {s:>9.4f}  {delta:>+8.4f}  {p:>8.3e}  {marker}")
            md_rows.append(f"| {sig:.2f} | {mode} | {g:.4f} | {s:.4f} | {delta:+.4f} | {p:.3e} | {md_ok} |")
            if ok is False:
                failures.append(Failure("A", "violation_rate: guarded ≥ standard",
                                        f"sig={sig}, mode={mode}, guarded={g:.4f}, standard={s:.4f}"))
        md += md_rows
    else:
        term.append(dim("  violation_rate data not found in summary_metrics.csv"))
        md.append("*violation_rate data not found.*")

    return term, md


def _summarize_scenario_b(dirs: dict, failures: List[Failure]) -> Tuple[List[str], List[str]]:
    d = dirs["b"]
    df = _load(d / "ood_metrics.csv")

    term: List[str] = [bold("  Scenario B — OOD detection quality")]
    md: List[str] = ["### Scenario B — OOD detection quality", ""]

    if df is None:
        term.append(dim("  Not run yet — no artifacts found."))
        md.append("*Not run yet — no artifacts found.*")
        return term, md

    params = _param_grid(df, ["significance", "n_neighbors", "n_dim", "normalize_guard"])
    seeds = df["seed"].nunique() if "seed" in df.columns else "?"
    term.append(f"  Params: seeds={seeds}  {params}")
    md.append(f"**Params:** seeds={seeds}, {params}")
    md.append("")

    # AUROC by shift_level × n_dim (normalized only), averaged over seeds and n_neighbors
    norm_df = df[df["normalize_guard"] == True] if "normalize_guard" in df.columns else df  # noqa: E712
    auroc_pivot = (
        norm_df.groupby(["n_dim", "shift_level"])["auroc"].mean().unstack("shift_level")
        if "n_dim" in norm_df.columns and "shift_level" in norm_df.columns
        else pd.DataFrame()
    )

    if not auroc_pivot.empty:
        shift_cols = list(auroc_pivot.columns)
        header = f"  {'n_dim':>6}" + "".join(f"  {c:>12}" for c in shift_cols) + "  attention"
        term.append("  AUROC (normalize_guard=True, mean over seeds × n_neighbors):")
        term.append(header)
        term.append("  " + "-" * (8 + 14 * len(shift_cols) + 10))

        md_hdr = "| n_dim |" + "|".join(f" auroc ({c}) " for c in shift_cols) + "| attention |"
        md_sep = "|---|" + "|".join("---|" for _ in shift_cols) + "---|"
        md_rows_auroc = [md_hdr, md_sep]

        for n_dim, row in auroc_pivot.iterrows():
            attentions = []
            md_vals = []
            for col in shift_cols:
                val = row.get(col, float("nan"))
                md_vals.append(f" {val:.3f} " if not np.isnan(val) else " — ")
                # AUROC < 0.60 for extreme shift is a red flag
                if col == "extreme" and not np.isnan(val) and val < 0.60:
                    attentions.append(red("AUROC<0.60 @extreme"))
                    failures.append(Failure("B", "AUROC < 0.60 for extreme shift",
                                            f"n_dim={n_dim}, auroc={val:.3f}"))
                elif col == "moderate" and not np.isnan(val) and val < 0.60:
                    attentions.append(yellow("AUROC<0.60 @moderate"))
            attn_str = ", ".join(attentions) if attentions else green("ok")
            row_str = f"  {n_dim:>6}" + "".join(
                f"  {row.get(c, float('nan')):>12.3f}" for c in shift_cols
            ) + f"  {attn_str}"
            term.append(row_str)
            md_rows_auroc.append(
                f"| {n_dim} |" + "|".join(md_vals) + f"| {'⚠' if attentions else '✓'} |"
            )
        md += md_rows_auroc
    else:
        term.append(dim("  AUROC data unavailable."))

    # FPR check
    if "fpr_at_significance" in df.columns and "significance" in df.columns:
        fpr_by_sig = df.groupby(["n_dim", "shift_level"])[["fpr_at_significance", "significance"]].mean()
        worst_violations = []
        for (n_dim, sl), row in fpr_by_sig.iterrows():
            fpr = row["fpr_at_significance"]
            sig = row["significance"]
            if not np.isnan(fpr) and not np.isnan(sig) and fpr > sig * 1.5:
                worst_violations.append((n_dim, sl, fpr, sig))
                failures.append(Failure("B", "FPR >> significance",
                                        f"n_dim={n_dim}, shift={sl}, fpr={fpr:.3f} >> sig={sig:.3f}"))
        if worst_violations:
            term.append(red(f"  FPR violations (fpr > 1.5×significance): {len(worst_violations)} config(s)"))
            md.append(f"\n**FPR violations:** {len(worst_violations)} config(s) where FPR > 1.5×significance.")
        else:
            term.append(green("  FPR: within conformal validity bounds across all configs."))
            md.append("\n**FPR:** within conformal validity bounds across all configs.")

    # Effect of normalize_guard
    if "normalize_guard" in df.columns:
        norm_effect = df.groupby("normalize_guard")["auroc"].mean()
        if True in norm_effect.index and False in norm_effect.index:
            delta_norm = norm_effect[True] - norm_effect[False]
            flag = yellow(f"Δ={delta_norm:+.3f}") if abs(delta_norm) > 0.05 else green(f"Δ={delta_norm:+.3f}")
            term.append(f"  normalize_guard effect on AUROC: {flag} (True={norm_effect[True]:.3f}, False={norm_effect[False]:.3f})")
            md.append(f"\n**normalize_guard effect:** Δ={delta_norm:+.3f} (True={norm_effect[True]:.3f}, False={norm_effect[False]:.3f})")

    return term, md


def _summarize_scenario_c(dirs: dict, failures: List[Failure]) -> Tuple[List[str], List[str]]:
    d = dirs["c"]
    violations = _load(d / "invariant_violations.csv")
    metrics = _load(d / "regression_metrics.csv")

    term: List[str] = [bold("  Scenario C — Regression invariants")]
    md: List[str] = ["### Scenario C — Regression invariants", ""]

    if violations is None and metrics is None:
        term.append(dim("  Not run yet — no artifacts found."))
        md.append("*Not run yet — no artifacts found.*")
        return term, md

    if metrics is not None:
        params = _param_grid(metrics, ["significance", "n_neighbors"])
        seeds = metrics["seed"].nunique() if "seed" in metrics.columns else "?"
        datasets = list(metrics["dataset"].unique()) if "dataset" in metrics.columns else []
        models = list(metrics["model"].unique()) if "model" in metrics.columns else []
        term.append(f"  Params: seeds={seeds}  datasets={datasets}  models={models}  {params}")
        md.append(f"**Params:** seeds={seeds}, datasets={datasets}, models={models}, {params}")
        md.append("")

    n_viol = 0 if violations is None or violations.empty else len(violations)
    if n_viol == 0:
        term.append(green("  n_invariant_violations = 0  ✓  Interval invariant holds everywhere."))
        md.append("**n_invariant_violations = 0** — interval invariant holds everywhere. ✓")
    else:
        term.append(red(f"  n_invariant_violations = {n_viol}  ✗  BUG: see invariant_violations.csv"))
        md.append(f"**n_invariant_violations = {n_viol}** — BUG detected. See `invariant_violations.csv`. ✗")
        failures.append(Failure("C", f"Interval invariant violated {n_viol} time(s)",
                                "See artifacts/guarded/scenario_c/invariant_violations.csv"))

    # Secondary diagnostic: fraction_removed by is_ood if available
    if metrics is not None and "is_ood" in metrics.columns and "intervals_removed_guard" in metrics.columns:
        diag = metrics.groupby("is_ood")["intervals_removed_guard"].mean()
        frac_id = diag.get(False, float("nan"))
        frac_ood = diag.get(True, float("nan"))
        responsive = frac_ood > frac_id if not (np.isnan(frac_id) or np.isnan(frac_ood)) else None
        marker = status_marker(responsive) if responsive is not None else dim("?")
        term.append(f"  Guard responsiveness (ID removed={frac_id:.2f} vs OOD removed={frac_ood:.2f}) {marker}")
        md.append(f"\n**Guard responsiveness:** ID intervals removed={frac_id:.2f}, OOD intervals removed={frac_ood:.2f} — {'OOD filtered more ✓' if responsive else ('similar ?' if responsive is None else 'OOD not filtered more ✗')}")

    return term, md


def _summarize_scenario_d(dirs: dict, failures: List[Failure]) -> Tuple[List[str], List[str]]:
    d = dirs["d"]
    metrics = _load(d / "real_dataset_metrics.csv")
    completeness = _load(d / "audit_completeness_details.csv")

    term: List[str] = [bold("  Scenario D — Real dataset correctness")]
    md: List[str] = ["### Scenario D — Real dataset correctness", ""]

    if metrics is None:
        term.append(dim("  Not run yet — no artifacts found."))
        md.append("*Not run yet — no artifacts found.*")
        return term, md

    params = _param_grid(metrics, ["significance", "n_neighbors"])
    seeds = metrics["seed"].nunique() if "seed" in metrics.columns else "?"
    datasets = list(metrics["dataset"].unique()) if "dataset" in metrics.columns else []
    term.append(f"  Params: seeds={seeds}  datasets={datasets}  {params}")
    md.append(f"**Params:** seeds={seeds}, datasets={datasets}, {params}")
    md.append("")

    # Field completeness
    n_missing = 0 if completeness is None or completeness.empty else len(completeness)
    complete_ok = n_missing == 0
    marker = status_marker(complete_ok)
    term.append(f"  audit_field_completeness: {marker}  missing-field records = {n_missing}")
    md.append(f"**audit_field_completeness:** {n_missing} missing-field records. {'✓ All fields present.' if complete_ok else '✗ See audit_completeness_details.csv.'}")
    if not complete_ok:
        failures.append(Failure("D", f"audit_field_completeness: {n_missing} missing-field record(s)",
                                "See artifacts/guarded/scenario_d/audit_completeness_details.csv"))

    # Fraction fully filtered per dataset at sig=0.10
    sub = metrics[
        np.isclose(metrics["significance"], 0.10)
    ] if "significance" in metrics.columns else metrics

    if not sub.empty and "fraction_instances_fully_filtered" in sub.columns:
        by_ds = sub.groupby("dataset")[["fraction_instances_fully_filtered",
                                        "mean_intervals_removed_per_instance",
                                        "n_features", "n_classes", "n_cal"]].mean()
        term.append("  fraction_instances_fully_filtered (sig=0.10):")
        term.append(f"    {'dataset':<18}  {'n_feat':>6}  {'n_cls':>5}  {'n_cal':>5}  {'frac_filtered':>14}  {'mean_removed':>12}  status")
        term.append("    " + "-" * 80)

        md_hdr = "| dataset | n_feat | n_cls | n_cal | frac_fully_filtered | mean_removed | status |"
        md_sep = "|---|---|---|---|---|---|---|"
        md_tbl = [md_hdr, md_sep]

        for ds, row in by_ds.iterrows():
            frac = row["fraction_instances_fully_filtered"]
            removed = row.get("mean_intervals_removed_per_instance", float("nan"))
            nf = int(row.get("n_features", 0))
            nc = int(row.get("n_classes", 0))
            ncal = int(row.get("n_cal", 0))
            if np.isnan(frac):
                st = dim("?")
                md_st = "?"
            elif frac > 0.10:
                st = red(f"✗ {frac:.1%} > 10%")
                md_st = f"✗ {frac:.1%}"
                failures.append(Failure("D", "fraction_fully_filtered > 10%",
                                        f"dataset={ds}, frac={frac:.1%}"))
            elif frac > 0.05:
                st = yellow(f"! {frac:.1%}")
                md_st = f"! {frac:.1%}"
            else:
                st = green(f"✓ {frac:.1%}")
                md_st = f"✓ {frac:.1%}"
            term.append(f"    {ds:<18}  {nf:>6}  {nc:>5}  {ncal:>5}  {frac:>14.1%}  {removed:>12.2f}  {st}")
            md_tbl.append(f"| {ds} | {nf} | {nc} | {ncal} | {frac:.1%} | {removed:.2f} | {md_st} |")

        md += md_tbl

        # API crash check
        if "error" in metrics.columns:
            n_errors = metrics["error"].notna().sum()
            if n_errors > 0:
                term.append(red(f"  API crashes: {n_errors} exception(s) recorded"))
                failures.append(Failure("D", f"API exception(s): {n_errors}",
                                        "Check 'error' column in real_dataset_metrics.csv"))
            else:
                term.append(green("  API crashes: 0"))

    return term, md


def _summarize_scenario_e(dirs: dict, failures: List[Failure]) -> Tuple[List[str], List[str]]:
    d = dirs["e"]
    df = _load(d / "edge_case_results.csv")

    term: List[str] = [bold("  Scenario E — Edge case behavior")]
    md: List[str] = ["### Scenario E — Edge case behavior", ""]

    if df is None:
        term.append(dim("  Not run yet — no artifacts found."))
        md.append("*Not run yet — no artifacts found.*")
        return term, md

    n_pass = int((df["status"] == "PASS").sum())
    n_fail = int((df["status"] == "FAIL").sum())
    term.append(f"  {n_pass} PASS / {n_fail} FAIL across {len(df)} cases")
    md.append(f"**{n_pass} PASS / {n_fail} FAIL** across {len(df)} cases.")
    md.append("")
    md.append("| Case | Status | Details |")
    md.append("|---|---|---|")

    # Cases where FAIL is expected (design boundaries) vs unexpected
    EXPECTED_FAIL_REASONS = {"E2"}  # E2 failure reveals implementation detail, not a bug

    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        status = str(row["status"])
        details = str(row.get("details", ""))
        is_fail = status == "FAIL"
        is_expected_boundary = case_id in EXPECTED_FAIL_REASONS

        if is_fail and is_expected_boundary:
            marker = yellow("! FAIL")
            md_marker = "⚠ FAIL (design boundary)"
        elif is_fail:
            marker = red("✗ FAIL")
            md_marker = "✗ FAIL"
            failures.append(Failure("E", f"{case_id}: {details[:80]}"))
        else:
            marker = green("✓ PASS")
            md_marker = "✓ PASS"

        details_short = details[:90] + "…" if len(details) > 90 else details
        term.append(f"  {case_id:>3}  {marker:<12}  {details_short}")
        md.append(f"| {case_id} | {md_marker} | {details[:120]} |")

    if n_fail > 0:
        md.append("")
        md.append("**Failure details:**")
        for _, row in df[df["status"] == "FAIL"].iterrows():
            case_id = str(row["case_id"])
            details = str(row.get("details", ""))
            expected = str(row.get("expected", ""))
            md.append(f"\n*{case_id}* — {details}")
            md.append(f"> Expected: {expected}")

    return term, md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "guarded",
        help="Root directory containing guarded artifact subfolders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    guarded_dir = args.artifacts_dir
    dirs = _artifact_dirs(guarded_dir)

    failures: List[Failure] = []
    all_md: List[str] = [
        "# Guarded Evaluation — Consolidated Summary",
        "",
        "Auto-generated by `summarize_results.py`. "
        "Re-run after adding new scenario artifacts.",
        "",
    ]

    print()
    print(bold("=" * 72))
    print(bold("  Guarded Explanation Evaluation — Consolidated Summary"))
    print(bold("=" * 72))

    for _, summarizer in [
        ("a", _summarize_scenario_a),
        ("b", _summarize_scenario_b),
        ("c", _summarize_scenario_c),
        ("d", _summarize_scenario_d),
        ("e", _summarize_scenario_e),
    ]:
        print()
        term_lines, md_lines = summarizer(dirs, failures)
        for line in term_lines:
            print(line)
        all_md += md_lines + [""]

    # Failure summary
    print()
    print(bold("=" * 72))
    if failures:
        print(bold(red(f"  ATTENTION — {len(failures)} issue(s) require review:")))
        print()
        all_md += ["---", "", f"## Attention — {len(failures)} issue(s)", ""]
        for i, f in enumerate(failures, 1):
            print(f"  {red(str(i) + '.')} {str(f)}")
            all_md.append(f"{i}. **[{f.scenario}]** {f.description}" + (f": {f.detail}" if f.detail else ""))
    else:
        print(bold(green("  All checks passed — no issues found.")))
        all_md += ["---", "", "## Status", "", "All checks passed — no issues found. ✓"]

    print()
    print(bold("=" * 72))
    print()

    # Write markdown
    out_path = guarded_dir / "consolidated_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(all_md), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
