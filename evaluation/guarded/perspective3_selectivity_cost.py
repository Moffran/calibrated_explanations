"""Perspective 3: Selectivity cost and subgroup impact.

Question: how much explanatory coverage is lost as filtering strength
increases, and does the guard disproportionately suppress rare instances?

Analysis 1 — Selective explanation curves:
  For each significance threshold epsilon, report the fraction of instances
  with non-empty explanations, fraction of retained features/rules, and
  the plausibility gain (violation rate).

Analysis 2 — Stratified retention by local density:
  Assign each test instance to a density quartile (based on distance to
  k-th nearest calibration neighbour) and report retention and violations
  per stratum at a fixed epsilon.

Analysis 3 — Calibration-size sensitivity:
  Repeat key analyses for multiple calibration-set sizes at fixed epsilon
  to quantify over-conservatism risk in small calibration regimes.

Metrics:
  nonempty_rate — fraction of instances with >= 1 emitted rule.
  feature_retention — fraction of features retaining >= 1 conforming rule.
  rule_retention — fraction of candidate rules passing the guard.
  violation_rate — fraction of emitted rules violating the known constraint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from calibrated_explanations.ce_agent_utils import (
    ensure_ce_first_wrapper,
    fit_and_calibrate,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SELECTIVITY_EPSILONS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)
CALSIZE_EPSILON = 0.10
CALSIZE_GRID = (250, 500, 1000, 2000, 5000)
DENSITY_EPSILON = 0.10
K_NEIGHBORS = 5


# ---------------------------------------------------------------------------
# Data generation (4D Scenario A: 2 informative + 2 noise)
# ---------------------------------------------------------------------------

def scenario_a_constraint(x: np.ndarray) -> np.ndarray:
    return x[:, 1] <= (2.0 * x[:, 0] + 3.0)


def generate_scenario_a(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    accepted: List[np.ndarray] = []
    cov = np.array([[1.0, 0.9], [0.9, 1.5]])
    while sum(c.shape[0] for c in accepted) < n:
        candidate = rng.multivariate_normal([0.0, 0.0], cov, size=max(2048, n))
        mask = scenario_a_constraint(candidate)
        if np.any(mask):
            accepted.append(candidate[mask])
    x_info = np.vstack(accepted)[:n]
    x_noise = rng.standard_normal((n, 2))
    x = np.hstack([x_info, x_noise])
    logits = 0.9 * x[:, 0] + 0.7 * x[:, 1] + rng.normal(0.0, 0.75, size=n)
    y = (logits > np.median(logits)).astype(int)
    return x, y


def get_models(seed: int) -> Dict[str, Any]:
    return {
        "logreg": LogisticRegression(max_iter=1200, random_state=seed),
        "rf": RandomForestClassifier(
            n_estimators=240, max_depth=8, random_state=seed, n_jobs=1,
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_representative(sampled_values: Any) -> Optional[float]:
    if sampled_values is None:
        return None
    if isinstance(sampled_values, (int, float, np.number)):
        return float(sampled_values)
    arr = np.asarray(sampled_values).reshape(-1)
    return float(arr[-1]) if arr.size > 0 else None


def check_rule_violation(
    x_instance: np.ndarray, feature: int, representative: Optional[float],
) -> bool:
    """Return True if perturbing feature to representative violates the constraint."""
    if representative is None:
        return False
    candidate = np.array(x_instance, copy=True)
    candidate[feature] = representative
    return not bool(scenario_a_constraint(candidate.reshape(1, -1))[0])


def compute_instance_density(
    x_test: np.ndarray, x_cal: np.ndarray, k: int = K_NEIGHBORS,
) -> np.ndarray:
    """Mean distance to k-th nearest calibration neighbour per test instance."""
    nn = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2)
    nn.fit(x_cal)
    distances, _ = nn.kneighbors(x_test)
    return distances.mean(axis=1)


# ---------------------------------------------------------------------------
# Analysis 1: Selective explanation curves over epsilon
# ---------------------------------------------------------------------------

def run_selectivity_curves(args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in range(args.num_seeds):
        x_tr, y_tr = generate_scenario_a(args.n_train, seed * 100 + 11)
        x_cal, y_cal = generate_scenario_a(args.n_cal, seed * 100 + 17)
        x_te, _ = generate_scenario_a(args.n_test, seed * 100 + 23)

        rng = np.random.default_rng(seed + 771)
        sampled = np.sort(
            rng.choice(args.n_test, size=min(args.sample_test, args.n_test), replace=False)
        )

        for model_name, model in get_models(seed).items():
            wrapper = ensure_ce_first_wrapper(model)
            fit_and_calibrate(wrapper, x_tr, y_tr, x_cal, y_cal)
            n_features = x_te.shape[1]

            for eps in SELECTIVITY_EPSILONS:
                for inst_id in sampled:
                    x_inst = x_te[inst_id : inst_id + 1]
                    g_expl = wrapper.explain_guarded_factual(
                        x_inst,
                        significance=eps,
                        n_neighbors=K_NEIGHBORS,
                        merge_adjacent=False,
                        normalize_guard=True,
                    )
                    audit = g_expl.get_guarded_audit()["instances"][0]
                    summary = audit["summary"]

                    tested = max(summary["intervals_tested"], 1)
                    emitted = summary["intervals_emitted"]
                    emitted_features = {
                        int(rec["feature"])
                        for rec in audit["intervals"]
                        if rec["emitted"]
                    }

                    # Violation check on emitted rules
                    rules = g_expl[0].get_rules()
                    n_rules = len(rules.get("rule", []))
                    violations = 0
                    for idx in range(n_rules):
                        feat = int(rules["feature"][idx])
                        rep = safe_representative(rules["sampled_values"][idx])
                        if check_rule_violation(x_inst[0], feat, rep):
                            violations += 1

                    rows.append({
                        "seed": seed,
                        "model": model_name,
                        "instance_id": int(inst_id),
                        "significance": eps,
                        "nonempty": int(n_rules > 0),
                        "feature_retention": len(emitted_features) / max(n_features, 1),
                        "rule_retention": emitted / tested,
                        "violation_rate": violations / max(n_rules, 1),
                        "rule_count": n_rules,
                    })
        print(f"  Selectivity curves: seed {seed + 1}/{args.num_seeds}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 2: Stratified retention by local density
# ---------------------------------------------------------------------------

def run_density_stratified(args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    eps = DENSITY_EPSILON

    for seed in range(args.num_seeds):
        x_tr, y_tr = generate_scenario_a(args.n_train, seed * 100 + 11)
        x_cal, y_cal = generate_scenario_a(args.n_cal, seed * 100 + 17)
        x_te, _ = generate_scenario_a(args.n_test, seed * 100 + 23)

        rng = np.random.default_rng(seed + 771)
        sampled = np.sort(
            rng.choice(args.n_test, size=min(args.sample_test, args.n_test), replace=False)
        )

        density_scores = compute_instance_density(x_te, x_cal)
        sampled_density = density_scores[sampled]
        quartile_edges = np.percentile(sampled_density, [25, 50, 75])

        def _quartile(d: float) -> str:
            if d <= quartile_edges[0]:
                return "Q1_dense"
            if d <= quartile_edges[1]:
                return "Q2"
            if d <= quartile_edges[2]:
                return "Q3"
            return "Q4_sparse"

        for model_name, model in get_models(seed).items():
            wrapper = ensure_ce_first_wrapper(model)
            fit_and_calibrate(wrapper, x_tr, y_tr, x_cal, y_cal)
            n_features = x_te.shape[1]

            for local_idx, inst_id in enumerate(sampled):
                x_inst = x_te[inst_id : inst_id + 1]
                g_expl = wrapper.explain_guarded_factual(
                    x_inst,
                    significance=eps,
                    n_neighbors=K_NEIGHBORS,
                    merge_adjacent=False,
                    normalize_guard=True,
                )
                audit = g_expl.get_guarded_audit()["instances"][0]
                summary = audit["summary"]
                tested = max(summary["intervals_tested"], 1)
                emitted = summary["intervals_emitted"]
                emitted_features = {
                    int(rec["feature"])
                    for rec in audit["intervals"]
                    if rec["emitted"]
                }
                rules = g_expl[0].get_rules()
                n_rules = len(rules.get("rule", []))
                violations = 0
                for idx in range(n_rules):
                    feat = int(rules["feature"][idx])
                    rep = safe_representative(rules["sampled_values"][idx])
                    if check_rule_violation(x_inst[0], feat, rep):
                        violations += 1

                rows.append({
                    "seed": seed,
                    "model": model_name,
                    "instance_id": int(inst_id),
                    "density_score": sampled_density[local_idx],
                    "density_quartile": _quartile(sampled_density[local_idx]),
                    "nonempty": int(n_rules > 0),
                    "feature_retention": len(emitted_features) / max(n_features, 1),
                    "rule_retention": emitted / tested,
                    "violation_rate": violations / max(n_rules, 1),
                    "rule_count": n_rules,
                })
        print(f"  Density stratified: seed {seed + 1}/{args.num_seeds}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 3: Calibration-size sensitivity
# ---------------------------------------------------------------------------

def run_calsize_sensitivity(args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    eps = CALSIZE_EPSILON
    max_cal = max(CALSIZE_GRID)
    # Use fewer seeds for the expensive cal-size sweep
    cal_seeds = min(args.num_seeds, args.calsize_seeds)

    for seed in range(cal_seeds):
        x_tr, y_tr = generate_scenario_a(args.n_train, seed * 100 + 11)
        x_cal_full, y_cal_full = generate_scenario_a(max_cal, seed * 100 + 17)
        x_te, _ = generate_scenario_a(args.n_test, seed * 100 + 23)

        rng = np.random.default_rng(seed + 771)
        sampled = np.sort(
            rng.choice(args.n_test, size=min(args.sample_test, args.n_test), replace=False)
        )

        for cal_size in CALSIZE_GRID:
            x_cal = x_cal_full[:cal_size]
            y_cal = y_cal_full[:cal_size]

            for model_name, model in get_models(seed).items():
                wrapper = ensure_ce_first_wrapper(model)
                fit_and_calibrate(wrapper, x_tr, y_tr, x_cal, y_cal)
                n_features = x_te.shape[1]

                for inst_id in sampled:
                    x_inst = x_te[inst_id : inst_id + 1]
                    g_expl = wrapper.explain_guarded_factual(
                        x_inst,
                        significance=eps,
                        n_neighbors=K_NEIGHBORS,
                        merge_adjacent=False,
                        normalize_guard=True,
                    )
                    audit = g_expl.get_guarded_audit()["instances"][0]
                    summary = audit["summary"]
                    tested = max(summary["intervals_tested"], 1)
                    emitted = summary["intervals_emitted"]
                    emitted_features = {
                        int(rec["feature"])
                        for rec in audit["intervals"]
                        if rec["emitted"]
                    }
                    rules = g_expl[0].get_rules()
                    n_rules = len(rules.get("rule", []))
                    violations = 0
                    for idx in range(n_rules):
                        feat = int(rules["feature"][idx])
                        rep = safe_representative(rules["sampled_values"][idx])
                        if check_rule_violation(x_inst[0], feat, rep):
                            violations += 1

                    rows.append({
                        "seed": seed,
                        "model": model_name,
                        "instance_id": int(inst_id),
                        "cal_size": cal_size,
                        "nonempty": int(n_rules > 0),
                        "feature_retention": len(emitted_features) / max(n_features, 1),
                        "rule_retention": emitted / tested,
                        "violation_rate": violations / max(n_rules, 1),
                        "rule_count": n_rules,
                    })
            print(f"  Cal-size {cal_size}: seed {seed + 1}/{cal_seeds}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def build_selectivity_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "significance"], as_index=False)
        .agg(
            nonempty_rate=("nonempty", "mean"),
            feature_retention=("feature_retention", "mean"),
            rule_retention=("rule_retention", "mean"),
            violation_rate=("violation_rate", "mean"),
            mean_rules=("rule_count", "mean"),
        )
    )


def build_density_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "density_quartile"], as_index=False)
        .agg(
            nonempty_rate=("nonempty", "mean"),
            feature_retention=("feature_retention", "mean"),
            rule_retention=("rule_retention", "mean"),
            violation_rate=("violation_rate", "mean"),
            mean_rules=("rule_count", "mean"),
        )
    )


def build_calsize_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "cal_size"], as_index=False)
        .agg(
            nonempty_rate=("nonempty", "mean"),
            feature_retention=("feature_retention", "mean"),
            rule_retention=("rule_retention", "mean"),
            violation_rate=("violation_rate", "mean"),
            mean_rules=("rule_count", "mean"),
        )
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_selectivity_curves(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["nonempty_rate", "feature_retention", "rule_retention", "violation_rate"]
    titles = [
        "Non-empty Explanation Rate",
        "Feature Retention",
        "Rule Retention",
        "Violation Rate",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        for model in sorted(summary["model"].unique()):
            sub = summary[summary["model"] == model].sort_values("significance")
            ax.plot(sub["significance"], sub[metric], marker="o", label=model)
        ax.set_xlabel("significance (ε)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "selectivity_curves.png", dpi=160)
    plt.close()


def plot_density_stratified(summary: pd.DataFrame, out_dir: Path) -> None:
    quartile_order = ["Q1_dense", "Q2", "Q3", "Q4_sparse"]
    metrics = ["nonempty_rate", "rule_retention", "violation_rate"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, metrics):
        for model in sorted(summary["model"].unique()):
            sub = summary[summary["model"] == model]
            sub = sub.set_index("density_quartile").reindex(quartile_order)
            ax.plot(quartile_order, sub[metric], marker="s", label=model)
        ax.set_xlabel("Density Quartile")
        ax.set_ylabel(metric)
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelrotation=15)
    plt.tight_layout()
    plt.savefig(out_dir / "density_stratified.png", dpi=160)
    plt.close()


def plot_calsize_sensitivity(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["nonempty_rate", "rule_retention", "violation_rate"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, metrics):
        for model in sorted(summary["model"].unique()):
            sub = summary[summary["model"] == model].sort_values("cal_size")
            ax.plot(sub["cal_size"], sub[metric], marker="o", label=model)
        ax.set_xlabel("Calibration Set Size")
        ax.set_ylabel(metric)
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "calsize_sensitivity.png", dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    sel_summary: pd.DataFrame,
    dens_summary: pd.DataFrame,
    cal_summary: pd.DataFrame,
    run_cfg: Dict[str, Any],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Perspective 3: Selectivity Cost and Subgroup Impact",
        "",
        "## Setup",
        f"- Seeds: {run_cfg['num_seeds']} (cal-size: {run_cfg['calsize_seeds']})",
        f"- Models: {', '.join(run_cfg['models'])}",
        f"- Epsilon sweep: {list(SELECTIVITY_EPSILONS)}",
        f"- Calibration sizes: {list(CALSIZE_GRID)}",
        f"- Density/cal-size epsilon: {DENSITY_EPSILON}",
        "",
        "## Analysis 1: Selective Explanation Curves",
        "",
        "Metrics over the full epsilon sweep.  As epsilon increases, the guard",
        "becomes more aggressive: more rules are removed, more instances may",
        "lose all explanations, but fewer constraint violations survive.",
        "",
        sel_summary.to_markdown(index=False),
        "",
        "## Analysis 2: Stratified Retention by Density",
        "",
        f"Density quartiles computed from mean distance to {K_NEIGHBORS} nearest",
        f"calibration neighbours.  Q1 = densest, Q4 = sparsest.  ε = {DENSITY_EPSILON}.",
        "",
        dens_summary.to_markdown(index=False),
        "",
        "## Analysis 3: Calibration-Size Sensitivity",
        "",
        f"Fixed ε = {CALSIZE_EPSILON}, varying calibration set size.",
        "",
        cal_summary.to_markdown(index=False),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Perspective 3: Selectivity cost.")
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "artifacts" / "perspective3",
    )
    p.add_argument("--num-seeds", type=int, default=30)
    p.add_argument("--calsize-seeds", type=int, default=10)
    p.add_argument("--sample-test", type=int, default=200)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-cal", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--quick", action="store_true", help="Smoke-test mode.")
    p.add_argument("--large", action="store_true", help="Large evidence profile.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.large:
        args.num_seeds = max(args.num_seeds, 40)
        args.calsize_seeds = max(args.calsize_seeds, 20)
        args.sample_test = max(args.sample_test, 600)
        args.n_train = max(args.n_train, 12000)
        args.n_cal = max(args.n_cal, 2000)
        args.n_test = max(args.n_test, 4000)
    if args.quick:
        args.num_seeds = min(args.num_seeds, 2)
        args.calsize_seeds = min(args.calsize_seeds, 2)
        args.sample_test = min(args.sample_test, 16)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    run_cfg: Dict[str, Any] = {
        "num_seeds": args.num_seeds,
        "calsize_seeds": args.calsize_seeds,
        "sample_test": args.sample_test,
        "n_train": args.n_train,
        "n_cal": args.n_cal,
        "n_test": args.n_test,
        "models": ["logreg", "rf"],
    }
    (out / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2), encoding="utf-8",
    )

    print("=== Analysis 1: Selective Explanation Curves ===")
    sel_df = run_selectivity_curves(args)
    sel_df.to_csv(out / "selectivity_curves.csv", index=False)
    sel_summary = build_selectivity_summary(sel_df)
    sel_summary.to_csv(out / "selectivity_curves_summary.csv", index=False)
    plot_selectivity_curves(sel_summary, out)

    print("\n=== Analysis 2: Density Stratified ===")
    dens_df = run_density_stratified(args)
    dens_df.to_csv(out / "density_stratified.csv", index=False)
    dens_summary = build_density_summary(dens_df)
    dens_summary.to_csv(out / "density_stratified_summary.csv", index=False)
    plot_density_stratified(dens_summary, out)

    print("\n=== Analysis 3: Calibration-Size Sensitivity ===")
    cal_df = run_calsize_sensitivity(args)
    cal_df.to_csv(out / "calsize_sensitivity.csv", index=False)
    cal_summary = build_calsize_summary(cal_df)
    cal_summary.to_csv(out / "calsize_sensitivity_summary.csv", index=False)
    plot_calsize_sensitivity(cal_summary, out)

    write_report(sel_summary, dens_summary, cal_summary, run_cfg, out / "report.md")
    print(f"\nDone.  Outputs in: {out}")


if __name__ == "__main__":
    main()
