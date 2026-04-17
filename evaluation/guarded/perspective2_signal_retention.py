"""Perspective 2: Explanatory signal retention under guarded filtering.

Question: after guard filtering, do explanations still recover known local
drivers, and are they stable under calibration resampling?

This evaluates a 4-dimensional binary classification where features 0 and 1
are informative (logits = 0.9*x0 + 0.7*x1) and features 2 and 3 are
independent noise.  The constraint x1 <= 2*x0 + 3 applies to the
informative features.

Analysis 1 — Driver recovery:
  For each test instance, rank features by |weight| from the factual
  explanation and check whether the top features are informative.

Analysis 2 — Stability under calibration resampling:
  Bootstrap the calibration set, re-explain the same instances, and compute
  Jaccard overlap of emitted rule-condition sets.

Metrics:
  driver_recovery_top1 — fraction where top feature is informative.
  driver_recovery_top2 — fraction where both top-2 features are informative.
  noise_rule_fraction — fraction of emitted rules for noise features.
  stability_jaccard — mean Jaccard overlap under calibration resampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from calibrated_explanations.ce_agent_utils import (
    ensure_ce_first_wrapper,
    fit_and_calibrate,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INFORMATIVE_FEATURES: FrozenSet[int] = frozenset({0, 1})
N_FEATURES = 4
DEFAULT_SIGNIFICANCE = (0.05, 0.10, 0.20)
STABILITY_SIGNIFICANCE = 0.10


# ---------------------------------------------------------------------------
# Data generation (4D: 2 informative + 2 noise, with constraint)
# ---------------------------------------------------------------------------

def scenario_a_constraint(x: np.ndarray) -> np.ndarray:
    """Domain constraint x1 <= 2*x0 + 3 (only involves features 0, 1)."""
    return x[:, 1] <= (2.0 * x[:, 0] + 3.0)


def generate_4d_data(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Binary classification in R^4: features 0,1 informative; 2,3 noise."""
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


def build_splits(
    seed: int, n_train: int, n_cal: int, n_test: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train = generate_4d_data(n_train, seed * 100 + 11)
    x_cal, y_cal = generate_4d_data(n_cal, seed * 100 + 17)
    x_test, y_test = generate_4d_data(n_test, seed * 100 + 23)
    return x_train, y_train, x_cal, y_cal, x_test, y_test


def get_models(seed: int) -> Dict[str, Any]:
    return {
        "logreg": LogisticRegression(max_iter=1200, random_state=seed),
        "rf": RandomForestClassifier(
            n_estimators=240, max_depth=8, random_state=seed, n_jobs=1,
        ),
    }


# ---------------------------------------------------------------------------
# Rule extraction helpers
# ---------------------------------------------------------------------------

def _run_multibin_factual(wrapper: Any, x: np.ndarray) -> Any:
    """Multi-bin factual without guard (isolates guard effect)."""
    assert wrapper.explainer is not None
    return wrapper.explainer.explanation_orchestrator.invoke_factual(
        x=x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        discretizer="entropy",
        _use_plugin=True,
    )


def extract_feature_weights(explanation: Any) -> Dict[int, float]:
    """Feature index -> max |weight| across emitted rules."""
    rules = explanation.get_rules()
    weights: Dict[int, float] = {}
    for idx, feat in enumerate(rules.get("feature", [])):
        feat = int(feat)
        predict = rules["predict"][idx]
        w = abs(float(predict)) if predict is not None else 0.0
        if feat not in weights or w > weights[feat]:
            weights[feat] = w
    return weights


def rank_features(weights: Dict[int, float], n_features: int) -> List[int]:
    """Return feature indices sorted by descending importance."""
    all_w = {i: weights.get(i, 0.0) for i in range(n_features)}
    return sorted(all_w.keys(), key=lambda f: all_w[f], reverse=True)


def extract_rule_keys(explanation: Any) -> List[str]:
    """List of 'feature:condition' strings for Jaccard comparison."""
    rules = explanation.get_rules()
    return [
        f"{int(rules['feature'][i])}:{rules['rule'][i]}"
        for i in range(len(rules.get("rule", [])))
    ]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


# ---------------------------------------------------------------------------
# Analysis 1: Driver recovery
# ---------------------------------------------------------------------------

def run_driver_recovery(args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in range(args.num_seeds):
        x_tr, y_tr, x_cal, y_cal, x_te, _ = build_splits(
            seed, args.n_train, args.n_cal, args.n_test,
        )
        rng = np.random.default_rng(seed + 771)
        sampled = np.sort(
            rng.choice(args.n_test, size=min(args.sample_test, args.n_test), replace=False)
        )

        for model_name, model in get_models(seed).items():
            wrapper = ensure_ce_first_wrapper(model)
            fit_and_calibrate(wrapper, x_tr, y_tr, x_cal, y_cal)

            for inst_id in sampled:
                x_inst = x_te[inst_id : inst_id + 1]

                # --- Standard multi-bin (no guard) ---
                std_expl = _run_multibin_factual(wrapper, x_inst)
                std_w = extract_feature_weights(std_expl[0])
                std_ranked = rank_features(std_w, N_FEATURES)
                std_noise = sum(
                    1 for f in std_expl[0].get_rules().get("feature", [])
                    if int(f) not in INFORMATIVE_FEATURES
                )
                std_total = len(std_expl[0].get_rules().get("feature", []))
                rows.append({
                    "seed": seed,
                    "model": model_name,
                    "instance_id": int(inst_id),
                    "method": "standard_multibin",
                    "significance": float("nan"),
                    "top1_correct": int(bool(std_w) and std_ranked[0] in INFORMATIVE_FEATURES),
                    "top2_correct": int(
                        bool(std_w)
                        and std_ranked[0] in INFORMATIVE_FEATURES
                        and std_ranked[1] in INFORMATIVE_FEATURES
                    ),
                    "noise_rules": std_noise,
                    "total_rules": std_total,
                    "noise_rule_fraction": std_noise / max(std_total, 1),
                })

                # --- Guarded at each significance ---
                for eps in args.significance:
                    g_expl = wrapper.explain_guarded_factual(
                        x_inst,
                        significance=eps,
                        n_neighbors=5,
                        merge_adjacent=False,
                        normalize_guard=True,
                    )
                    g_w = extract_feature_weights(g_expl[0])
                    g_ranked = rank_features(g_w, N_FEATURES)
                    g_noise = sum(
                        1 for f in g_expl[0].get_rules().get("feature", [])
                        if int(f) not in INFORMATIVE_FEATURES
                    )
                    g_total = len(g_expl[0].get_rules().get("feature", []))
                    rows.append({
                        "seed": seed,
                        "model": model_name,
                        "instance_id": int(inst_id),
                        "method": "guarded",
                        "significance": eps,
                        "top1_correct": int(bool(g_w) and g_ranked[0] in INFORMATIVE_FEATURES),
                        "top2_correct": int(
                            bool(g_w)
                            and g_ranked[0] in INFORMATIVE_FEATURES
                            and g_ranked[1] in INFORMATIVE_FEATURES
                        ),
                        "noise_rules": g_noise,
                        "total_rules": g_total,
                        "noise_rule_fraction": g_noise / max(g_total, 1),
                    })
        print(f"  Driver recovery: seed {seed + 1}/{args.num_seeds}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 2: Stability under calibration resampling
# ---------------------------------------------------------------------------

def run_stability(args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    eps = STABILITY_SIGNIFICANCE
    for seed in range(args.num_seeds):
        x_tr, y_tr, x_cal, y_cal, x_te, _ = build_splits(
            seed, args.n_train, args.n_cal, args.n_test,
        )
        rng = np.random.default_rng(seed + 771)
        sampled = np.sort(
            rng.choice(args.n_test, size=min(args.sample_test, args.n_test), replace=False)
        )

        for model_name in get_models(seed):
            wrapper = ensure_ce_first_wrapper(get_models(seed)[model_name])
            fit_and_calibrate(wrapper, x_tr, y_tr, x_cal, y_cal)

            # Reference explanations
            ref_std: List[List[str]] = []
            ref_guard: List[List[str]] = []
            for inst_id in sampled:
                x_inst = x_te[inst_id : inst_id + 1]
                std_expl = _run_multibin_factual(wrapper, x_inst)
                g_expl = wrapper.explain_guarded_factual(
                    x_inst, significance=eps, n_neighbors=5,
                    merge_adjacent=False, normalize_guard=True,
                )
                ref_std.append(extract_rule_keys(std_expl[0]))
                ref_guard.append(extract_rule_keys(g_expl[0]))

            # Bootstrap replicates
            for draw in range(args.bootstrap_draws):
                boot_rng = np.random.default_rng(seed * 5000 + draw + 901)
                boot_idx = boot_rng.choice(len(x_cal), size=len(x_cal), replace=True)
                boot_wrapper = ensure_ce_first_wrapper(get_models(seed)[model_name])
                fit_and_calibrate(
                    boot_wrapper, x_tr, y_tr, x_cal[boot_idx], y_cal[boot_idx],
                )

                for local_idx, inst_id in enumerate(sampled):
                    x_inst = x_te[inst_id : inst_id + 1]
                    std_boot = _run_multibin_factual(boot_wrapper, x_inst)
                    g_boot = boot_wrapper.explain_guarded_factual(
                        x_inst, significance=eps, n_neighbors=5,
                        merge_adjacent=False, normalize_guard=True,
                    )
                    rows.append({
                        "seed": seed,
                        "model": model_name,
                        "instance_id": int(inst_id),
                        "draw": draw,
                        "method": "standard_multibin",
                        "stability_jaccard": jaccard(
                            ref_std[local_idx],
                            extract_rule_keys(std_boot[0]),
                        ),
                    })
                    rows.append({
                        "seed": seed,
                        "model": model_name,
                        "instance_id": int(inst_id),
                        "draw": draw,
                        "method": "guarded",
                        "stability_jaccard": jaccard(
                            ref_guard[local_idx],
                            extract_rule_keys(g_boot[0]),
                        ),
                    })
        print(f"  Stability: seed {seed + 1}/{args.num_seeds}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def build_driver_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "significance", "model"], as_index=False, dropna=False)
        .agg(
            top1_recovery=("top1_correct", "mean"),
            top2_recovery=("top2_correct", "mean"),
            noise_rule_frac=("noise_rule_fraction", "mean"),
            mean_rules=("total_rules", "mean"),
        )
    )


def build_stability_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "model"], as_index=False)
        .agg(
            mean_jaccard=("stability_jaccard", "mean"),
            std_jaccard=("stability_jaccard", "std"),
        )
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_driver_recovery(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model in sorted(summary["model"].unique()):
        sub = summary[summary["model"] == model]
        std = sub[sub["method"] == "standard_multibin"]
        guarded = sub[sub["method"] == "guarded"].sort_values("significance")
        if std.empty or guarded.empty:
            continue

        # Top-1
        ax = axes[0]
        ax.axhline(
            y=float(std["top1_recovery"].iloc[0]),
            linestyle="--", label=f"{model} standard",
        )
        ax.plot(
            guarded["significance"], guarded["top1_recovery"],
            marker="o", label=f"{model} guarded",
        )
        ax.set_xlabel("significance (ε)")
        ax.set_ylabel("Top-1 Driver Recovery")
        ax.set_title("Top-1 Recovery")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

        # Top-2
        ax = axes[1]
        ax.axhline(
            y=float(std["top2_recovery"].iloc[0]),
            linestyle="--", label=f"{model} standard",
        )
        ax.plot(
            guarded["significance"], guarded["top2_recovery"],
            marker="o", label=f"{model} guarded",
        )
        ax.set_xlabel("significance (ε)")
        ax.set_ylabel("Top-2 Driver Recovery")
        ax.set_title("Top-2 Recovery")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_dir / "driver_recovery.png", dpi=160)
    plt.close()


def plot_stability(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{r['model']}\n{r['method']}" for _, r in summary.iterrows()]
    values = summary["mean_jaccard"].tolist()
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean Jaccard Stability")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Calibration Resample Stability (ε={STABILITY_SIGNIFICANCE})")
    plt.tight_layout()
    plt.savefig(out_dir / "stability.png", dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    driver_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    run_cfg: Dict[str, Any],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Perspective 2: Signal Retention",
        "",
        "## Setup",
        f"- Seeds: {run_cfg['num_seeds']}",
        f"- Models: {', '.join(run_cfg['models'])}",
        f"- Features: {N_FEATURES} (informative: 0, 1; noise: 2, 3)",
        f"- Significance levels: {run_cfg['significance']}",
        f"- Bootstrap draws: {run_cfg['bootstrap_draws']}",
        f"- Stability significance: {STABILITY_SIGNIFICANCE}",
        "",
        "## Driver Recovery",
        "",
        "Fraction of instances where the top-ranked feature(s) by |weight|",
        "are informative (features 0 and 1).  Standard multi-bin uses the same",
        "discretiser as guarded but without the conformal guard.",
        "",
        driver_summary.to_markdown(index=False),
        "",
        "## Stability Under Calibration Resampling",
        "",
        "Mean Jaccard overlap of emitted (feature, condition) sets between a",
        f"reference calibration and {run_cfg['bootstrap_draws']} bootstrap resamples.",
        "",
        stability_summary.to_markdown(index=False),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Perspective 2: Signal retention.")
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "artifacts" / "perspective2",
    )
    p.add_argument("--num-seeds", type=int, default=30)
    p.add_argument("--bootstrap-draws", type=int, default=20)
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
        args.bootstrap_draws = min(args.bootstrap_draws, 3)
        args.sample_test = min(args.sample_test, 16)

    args.significance = list(DEFAULT_SIGNIFICANCE)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    run_cfg: Dict[str, Any] = {
        "num_seeds": args.num_seeds,
        "bootstrap_draws": args.bootstrap_draws,
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

    print("=== Analysis 1: Driver Recovery ===")
    driver_df = run_driver_recovery(args)
    driver_df.to_csv(out / "driver_recovery.csv", index=False)
    driver_summary = build_driver_summary(driver_df)
    driver_summary.to_csv(out / "driver_recovery_summary.csv", index=False)
    plot_driver_recovery(driver_summary, out)

    print("\n=== Analysis 2: Stability ===")
    stab_df = run_stability(args)
    stab_df.to_csv(out / "stability.csv", index=False)
    stab_summary = build_stability_summary(stab_df)
    stab_summary.to_csv(out / "stability_summary.csv", index=False)
    plot_stability(stab_summary, out)

    write_report(driver_summary, stab_summary, run_cfg, out / "report.md")
    print(f"\nDone.  Outputs in: {out}")


if __name__ == "__main__":
    main()
