"""Perspective 4: Component ablation.

Question: does each added component (multi-bin discretisation, per-feature
guard, bin merging / interaction gate) contribute measurable benefit?

Four variants are compared on the same Scenario A synthetic data used in
Perspective 1:

  V1  Standard CE (binary discretisation, no guard)
  V2  CE with multi-bin discretisation only
  V3  CE with multi-bin + per-feature guard (merge_adjacent=False)
  V4  CE with multi-bin + per-feature guard + merge (merge_adjacent=True)

Metrics:
  violation_rate — fraction of emitted rules violating x1 <= 2*x0 + 3.
  rule_count — number of emitted factual rules per instance.
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
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from calibrated_explanations.ce_agent_utils import (
    ensure_ce_first_wrapper,
    fit_and_calibrate,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SIGNIFICANCE = (0.01, 0.05, 0.10, 0.20)
K_NEIGHBORS = 5

VARIANT_LABELS = {
    "V1_binary": "V1: Binary (standard CE)",
    "V2_multibin": "V2: Multi-bin only",
    "V3_multibin_guard": "V3: Multi-bin + guard",
    "V4_multibin_guard_merge": "V4: Multi-bin + guard + merge",
}


# ---------------------------------------------------------------------------
# Data generation (2D Scenario A)
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
    """Return True if perturbing feature to representative violates constraint."""
    if representative is None:
        return False
    candidate = np.array(x_instance, copy=True)
    candidate[feature] = representative
    return not bool(scenario_a_constraint(candidate.reshape(1, -1))[0])


def extract_metrics(
    explanation: Any, x_instance: np.ndarray,
) -> Tuple[int, float]:
    """Return (rule_count, violation_rate) for a single explanation."""
    rules = explanation.get_rules()
    n_rules = len(rules.get("rule", []))
    if n_rules == 0:
        return 0, 0.0
    violations = 0
    for idx in range(n_rules):
        feat = int(rules["feature"][idx])
        rep = safe_representative(rules["sampled_values"][idx])
        if check_rule_violation(x_instance, feat, rep):
            violations += 1
    return n_rules, violations / n_rules


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------

def run_v1_binary(wrapper: Any, x_inst: np.ndarray) -> Any:
    """V1: standard binary discretisation factual."""
    return wrapper.explain_factual(x_inst)


def run_v2_multibin(wrapper: Any, x_inst: np.ndarray) -> Any:
    """V2: multi-bin discretisation, no guard."""
    assert wrapper.explainer is not None
    return wrapper.explainer.explanation_orchestrator.invoke_factual(
        x=x_inst,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        discretizer="entropy",
        _use_plugin=True,
    )


def run_v3_guard(wrapper: Any, x_inst: np.ndarray, eps: float) -> Any:
    """V3: multi-bin + guard, no merge."""
    return wrapper.explain_guarded_factual(
        x_inst,
        significance=eps,
        n_neighbors=K_NEIGHBORS,
        merge_adjacent=False,
        normalize_guard=True,
    )


def run_v4_guard_merge(wrapper: Any, x_inst: np.ndarray, eps: float) -> Any:
    """V4: multi-bin + guard + merge."""
    return wrapper.explain_guarded_factual(
        x_inst,
        significance=eps,
        n_neighbors=K_NEIGHBORS,
        merge_adjacent=True,
        normalize_guard=True,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_ablation(args: argparse.Namespace) -> pd.DataFrame:
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

            # --- Phase 1: V1 (binary) for all instances ---
            # Batch first to avoid oscillating discretiser setup.
            v1_cache: Dict[int, Tuple[int, float]] = {}
            for inst_id in sampled:
                x_inst = x_te[inst_id : inst_id + 1]
                v1_expl = run_v1_binary(wrapper, x_inst)
                v1_cache[int(inst_id)] = extract_metrics(v1_expl[0], x_inst[0])

            # --- Phase 2: V2 (multi-bin, no guard) for all instances ---
            v2_cache: Dict[int, Tuple[int, float]] = {}
            for inst_id in sampled:
                x_inst = x_te[inst_id : inst_id + 1]
                v2_expl = run_v2_multibin(wrapper, x_inst)
                v2_cache[int(inst_id)] = extract_metrics(v2_expl[0], x_inst[0])

            # --- Phase 3: V3/V4 (guarded) per epsilon ---
            for eps in DEFAULT_SIGNIFICANCE:
                for inst_id in sampled:
                    x_inst = x_te[inst_id : inst_id + 1]
                    iid = int(inst_id)
                    v1_rc, v1_vr = v1_cache[iid]
                    v2_rc, v2_vr = v2_cache[iid]

                    rows.append({
                        "seed": seed, "model": model_name,
                        "instance_id": iid, "variant": "V1_binary",
                        "significance": eps,
                        "rule_count": v1_rc, "violation_rate": v1_vr,
                    })
                    rows.append({
                        "seed": seed, "model": model_name,
                        "instance_id": iid, "variant": "V2_multibin",
                        "significance": eps,
                        "rule_count": v2_rc, "violation_rate": v2_vr,
                    })

                    v3_expl = run_v3_guard(wrapper, x_inst, eps)
                    v3_rc, v3_vr = extract_metrics(v3_expl[0], x_inst[0])
                    rows.append({
                        "seed": seed, "model": model_name,
                        "instance_id": iid, "variant": "V3_multibin_guard",
                        "significance": eps,
                        "rule_count": v3_rc, "violation_rate": v3_vr,
                    })

                    v4_expl = run_v4_guard_merge(wrapper, x_inst, eps)
                    v4_rc, v4_vr = extract_metrics(v4_expl[0], x_inst[0])
                    rows.append({
                        "seed": seed, "model": model_name,
                        "instance_id": iid, "variant": "V4_multibin_guard_merge",
                        "significance": eps,
                        "rule_count": v4_rc, "violation_rate": v4_vr,
                    })

        print(f"  Ablation: seed {seed + 1}/{args.num_seeds}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-level aggregation with Wilcoxon tests vs V2 (multi-bin baseline)."""
    # Aggregate to seed level first to avoid treating per-instance rows as
    # independent observations within the same fitted model.
    seed_level = (
        df.groupby(
            ["model", "seed", "variant", "significance"], as_index=False,
        )
        .agg(
            violation_rate=("violation_rate", "mean"),
            rule_count=("rule_count", "mean"),
        )
    )

    summary_rows: List[Dict[str, Any]] = []
    for model in sorted(seed_level["model"].unique()):
        for eps in sorted(seed_level["significance"].unique()):
            sub = seed_level[
                (seed_level["model"] == model)
                & (seed_level["significance"] == eps)
            ]
            v2_vals = sub[sub["variant"] == "V2_multibin"]
            for variant in sorted(sub["variant"].unique()):
                vsub = sub[sub["variant"] == variant]
                row: Dict[str, Any] = {
                    "model": model,
                    "significance": eps,
                    "variant": variant,
                    "violation_rate_mean": float(vsub["violation_rate"].mean()),
                    "rule_count_mean": float(vsub["rule_count"].mean()),
                }

                # Wilcoxon vs V2 for violation_rate (if variant != V2)
                if variant != "V2_multibin" and not v2_vals.empty:
                    merged = vsub.merge(
                        v2_vals, on=["model", "seed", "significance"],
                        suffixes=("_var", "_v2"),
                    )
                    if len(merged) >= 3:
                        diffs_vr = merged["violation_rate_var"] - merged["violation_rate_v2"]
                        if not np.allclose(diffs_vr.to_numpy(), 0.0):
                            stat = wilcoxon(diffs_vr, alternative="two-sided")
                            row["wilcoxon_p_vs_v2_violation"] = float(stat.pvalue)
                        else:
                            row["wilcoxon_p_vs_v2_violation"] = 1.0
                    else:
                        row["wilcoxon_p_vs_v2_violation"] = float("nan")
                else:
                    row["wilcoxon_p_vs_v2_violation"] = float("nan")

                summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_ablation(summary: pd.DataFrame, out_dir: Path) -> None:
    variant_order = ["V1_binary", "V2_multibin", "V3_multibin_guard", "V4_multibin_guard_merge"]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    for metric, ylabel, title in [
        ("violation_rate_mean", "Violation Rate", "Ablation: Violation Rate"),
        ("rule_count_mean", "Rules / Instance", "Ablation: Rule Count"),
    ]:
        fig, axes = plt.subplots(
            1, len(summary["model"].unique()), figsize=(6 * len(summary["model"].unique()), 5),
            squeeze=False,
        )
        for col_idx, model in enumerate(sorted(summary["model"].unique())):
            ax = axes[0, col_idx]
            sub = summary[summary["model"] == model]
            for v_idx, variant in enumerate(variant_order):
                vsub = sub[sub["variant"] == variant].sort_values("significance")
                if vsub.empty:
                    continue
                ax.plot(
                    vsub["significance"], vsub[metric],
                    marker="o", color=colors[v_idx],
                    label=VARIANT_LABELS.get(variant, variant),
                )
            ax.set_xlabel("significance (ε)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} — {model}")
            ax.legend(fontsize=7, loc="best")

        plt.tight_layout()
        fname = "ablation_violation_rate.png" if "violation" in metric else "ablation_rule_count.png"
        plt.savefig(out_dir / fname, dpi=160)
        plt.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    summary: pd.DataFrame, run_cfg: Dict[str, Any], out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Perspective 4: Component Ablation",
        "",
        "## Setup",
        f"- Seeds: {run_cfg['num_seeds']}",
        f"- Models: {', '.join(run_cfg['models'])}",
        f"- Significance levels: {list(DEFAULT_SIGNIFICANCE)}",
        "",
        "## Variants",
        "",
        "| Label | Description |",
        "|-------|-------------|",
    ]
    for k, v in VARIANT_LABELS.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Results",
        "",
        "Wilcoxon p-value is computed against V2 (multi-bin baseline) for the",
        "violation_rate metric, isolating the guard's contribution beyond",
        "discretisation granularity.",
        "",
        summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "If V2 already reduces violations compared to V1, part of the benefit",
        "is attributable to finer binning rather than the guard.  V3 vs V2",
        "isolates the guard's marginal contribution.  V4 vs V3 tests whether",
        "bin merging adds value beyond the per-bin guard.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Perspective 4: Component ablation.")
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "artifacts" / "perspective4",
    )
    p.add_argument("--num-seeds", type=int, default=30)
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
        args.sample_test = max(args.sample_test, 600)
        args.n_train = max(args.n_train, 12000)
        args.n_cal = max(args.n_cal, 5000)
        args.n_test = max(args.n_test, 4000)
    if args.quick:
        args.num_seeds = min(args.num_seeds, 2)
        args.sample_test = min(args.sample_test, 16)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    run_cfg: Dict[str, Any] = {
        "num_seeds": args.num_seeds,
        "sample_test": args.sample_test,
        "n_train": args.n_train,
        "n_cal": args.n_cal,
        "n_test": args.n_test,
        "significance": list(DEFAULT_SIGNIFICANCE),
        "models": ["logreg", "rf"],
    }
    (out / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2), encoding="utf-8",
    )

    print("=== Component Ablation ===")
    ablation_df = run_ablation(args)
    ablation_df.to_csv(out / "ablation_results.csv", index=False)
    summary = build_summary(ablation_df)
    summary.to_csv(out / "ablation_summary.csv", index=False)
    plot_ablation(summary, out)
    write_report(summary, run_cfg, out / "report.md")
    print(f"\nDone.  Outputs in: {out}")


if __name__ == "__main__":
    main()
