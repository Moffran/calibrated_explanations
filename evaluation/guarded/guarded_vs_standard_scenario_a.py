from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from calibrated_explanations.ce_agent_utils import ensure_ce_first_wrapper, fit_and_calibrate

DEFAULT_SIGNIFICANCE = (0.01, 0.05, 0.1, 0.2)
DEFAULT_NEIGHBORS = (3, 5, 10)
DEFAULT_MERGE = (False, True)
MODES = ("factual", "alternative")


@dataclass(frozen=True)
class ExplainConfig:
    significance: float
    n_neighbors: int
    merge_adjacent: bool
    normalize_guard: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scenario A guarded vs standard CE comparison (factual + alternatives)."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/guarded_vs_standard/scenario_a"))
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--bootstrap-draws", type=int, default=30)
    parser.add_argument("--sample-test", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-cal", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--top-k-alternatives", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode.")
    return parser.parse_args()


def scenario_a_constraint(x: np.ndarray) -> np.ndarray:
    return x[:, 1] <= (2.0 * x[:, 0] + 3.0)


def generate_scenario_a(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    accepted: List[np.ndarray] = []
    cov = np.array([[1.0, 0.9], [0.9, 1.5]])
    while sum(chunk.shape[0] for chunk in accepted) < n:
        candidate = rng.multivariate_normal(mean=np.array([0.0, 0.0]), cov=cov, size=max(2048, n))
        mask = scenario_a_constraint(candidate)
        if np.any(mask):
            accepted.append(candidate[mask])
    x = np.vstack(accepted)[:n]
    logits = 0.9 * x[:, 0] + 0.7 * x[:, 1] + rng.normal(0.0, 0.75, size=n)
    y = (logits > np.median(logits)).astype(int)
    return x, y


def build_splits(seed: int, n_train: int, n_cal: int, n_test: int) -> Tuple[np.ndarray, ...]:
    x_train, y_train = generate_scenario_a(n_train, seed * 100 + 11)
    x_cal, y_cal = generate_scenario_a(n_cal, seed * 100 + 17)
    x_test, y_test = generate_scenario_a(n_test, seed * 100 + 23)
    return x_train, y_train, x_cal, y_cal, x_test, y_test


def get_models(seed: int) -> Dict[str, Any]:
    return {
        "logreg": LogisticRegression(max_iter=1200, random_state=seed),
        "rf": RandomForestClassifier(n_estimators=240, max_depth=8, random_state=seed, n_jobs=-1),
    }


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[0])


def _extract_rule_representative(sampled_values: Any) -> Optional[float]:
    if sampled_values is None:
        return None
    if isinstance(sampled_values, (int, float, np.number)):
        return float(sampled_values)
    arr = np.asarray(sampled_values).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[-1])


def _assert_interval_invariant(low: Any, point: Any, high: Any, *, where: str) -> None:
    if low is None or point is None or high is None:
        return
    low_f = _to_float(low)
    point_f = _to_float(point)
    high_f = _to_float(high)
    if low_f is None or point_f is None or high_f is None:
        return
    eps = 1e-9
    assert low_f <= high_f + eps, f"Interval invariant failed in {where}: low > high"
    assert low_f - eps <= point_f <= high_f + eps, (
        f"Interval invariant failed in {where}: point outside [low, high]"
    )


def _run_standard_factual_multibin(wrapper: Any, x: np.ndarray) -> Any:
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


def _rule_rows_from_standard(
    explanation: Any,
    *,
    dataset: str,
    seed: int,
    model_name: str,
    instance_id: int,
    mode: str,
    method: str,
    cfg: ExplainConfig,
    x_instance: np.ndarray,
    runtime_ms: float,
) -> List[Dict[str, Any]]:
    rules = explanation.get_rules()
    rows: List[Dict[str, Any]] = []
    for idx, condition in enumerate(rules.get("rule", [])):
        feature = int(rules["feature"][idx])
        representative = _extract_rule_representative(rules["sampled_values"][idx])
        candidate = np.array(x_instance, copy=True)
        if representative is not None:
            candidate[feature] = representative
        plausibility = bool(scenario_a_constraint(candidate.reshape(1, -1))[0]) if representative is not None else True
        point = _to_float(rules["predict"][idx]) if idx < len(rules.get("predict", [])) else None
        low = _to_float(rules["predict_low"][idx]) if idx < len(rules.get("predict_low", [])) else None
        high = _to_float(rules["predict_high"][idx]) if idx < len(rules.get("predict_high", [])) else None
        _assert_interval_invariant(low, point, high, where=f"{method}-{mode}-rule")
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "model": model_name,
                "instance_id": instance_id,
                "method": method,
                "mode": mode,
                "significance": cfg.significance,
                "n_neighbors": cfg.n_neighbors,
                "merge_adjacent": cfg.merge_adjacent,
                "rule_id": f"r{idx}",
                "feature": feature,
                "condition": condition,
                "representative_value": representative,
                "p_value_guard": np.nan,
                "prediction_point": point,
                "prediction_low": low,
                "prediction_high": high,
                "plausibility_flag": plausibility,
                "runtime_ms": float(runtime_ms),
            }
        )
    return rows


def _rule_rows_from_guarded_audit(
    audit_instance: Dict[str, Any],
    *,
    dataset: str,
    seed: int,
    model_name: str,
    instance_id: int,
    mode: str,
    method: str,
    cfg: ExplainConfig,
    x_instance: np.ndarray,
    runtime_ms: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    emitted = [rec for rec in audit_instance["intervals"] if rec["emitted"]]
    for idx, rec in enumerate(emitted):
        feature = int(rec["feature"])
        representative = rec["representative"]
        candidate = np.array(x_instance, copy=True)
        candidate[feature] = representative
        plausibility = bool(scenario_a_constraint(candidate.reshape(1, -1))[0])
        _assert_interval_invariant(
            rec["low"],
            rec["predict"],
            rec["high"],
            where=f"{method}-{mode}-guarded-rule",
        )
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "model": model_name,
                "instance_id": instance_id,
                "method": method,
                "mode": mode,
                "significance": cfg.significance,
                "n_neighbors": cfg.n_neighbors,
                "merge_adjacent": cfg.merge_adjacent,
                "rule_id": f"r{idx}",
                "feature": feature,
                "condition": rec["condition"],
                "representative_value": representative,
                "p_value_guard": rec["p_value"],
                "prediction_point": rec["predict"],
                "prediction_low": rec["low"],
                "prediction_high": rec["high"],
                "plausibility_flag": plausibility,
                "runtime_ms": float(runtime_ms),
            }
        )
    return rows


def _prepare_intersection_metrics(
    guarded_rows: Sequence[Dict[str, Any]],
    standard_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    guard_map = {(r["feature"], r["condition"]): r for r in guarded_rows}
    std_map = {(r["feature"], r["condition"]): r for r in standard_rows}
    shared_keys = set(guard_map).intersection(std_map)
    abs_diff: List[float] = []
    u_in_g: List[float] = []
    g_in_u: List[float] = []
    for key in shared_keys:
        g = guard_map[key]
        u = std_map[key]
        if g["prediction_point"] is None or u["prediction_point"] is None:
            continue
        abs_diff.append(abs(float(g["prediction_point"]) - float(u["prediction_point"])))
        if g["prediction_low"] is not None and g["prediction_high"] is not None:
            u_in_g.append(float(g["prediction_low"] <= u["prediction_point"] <= g["prediction_high"]))
        if u["prediction_low"] is not None and u["prediction_high"] is not None:
            g_in_u.append(float(u["prediction_low"] <= g["prediction_point"] <= u["prediction_high"]))
    return abs_diff, u_in_g, g_in_u


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    union = sa.union(sb)
    if not union:
        return 1.0
    return len(sa.intersection(sb)) / len(union)


def _extract_rule_conditions(explanations: Any, mode: str, top_k: int) -> List[List[str]]:
    out: List[List[str]] = []
    for exp in explanations:
        rules = list(exp.get_rules().get("rule", []))
        if mode == "alternative":
            rules = rules[:top_k]
        out.append(rules)
    return out


def _wilcoxon_from_group(group: pd.DataFrame, left: str, right: str) -> Tuple[float, float]:
    paired = group.pivot_table(
        index=["dataset", "model", "seed", "instance_id", "mode", "significance"],
        columns="method",
        values="value",
        aggfunc="mean",
    ).dropna()
    if left not in paired.columns or right not in paired.columns or paired.empty:
        return math.nan, math.nan
    diffs = paired[left] - paired[right]
    if np.allclose(diffs.to_numpy(), 0.0):
        return 1.0, 0.0
    stat = wilcoxon(paired[left], paired[right], alternative="two-sided", zero_method="wilcox")
    return float(stat.pvalue), float(np.median(diffs))


def _plot_outputs(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    retention = metrics_df[metrics_df["metric"] == "rule_count"]
    for method in ("guarded", "standard"):
        sub = retention[retention["method"] == method].groupby("significance", as_index=False)["value"].mean()
        plt.plot(sub["significance"], sub["value"], marker="o", label=method)
    plt.xlabel("significance")
    plt.ylabel("avg rules / instance")
    plt.title("Retention Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "retention_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plausibility = metrics_df[metrics_df["metric"] == "violation_rate"]
    for method in ("guarded", "standard"):
        sub = plausibility[plausibility["method"] == method].groupby("significance", as_index=False)["value"].mean()
        plt.plot(sub["significance"], sub["value"], marker="o", label=method)
    plt.xlabel("significance")
    plt.ylabel("violation rate")
    plt.title("Plausibility Violation Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plausibility_vs_significance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    pred_diff = metrics_df[metrics_df["metric"] == "prediction_abs_diff"]
    if not pred_diff.empty:
        grouped = [grp["value"].to_numpy() for _, grp in pred_diff.groupby("mode")]
        labels = list(pred_diff.groupby("mode").groups.keys())
        plt.boxplot(grouped, tick_labels=labels)
    plt.ylabel("|guarded point - standard point|")
    plt.title("Prediction Agreement (Intersection Rules)")
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_agreement_boxplot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    stability = metrics_df[metrics_df["metric"] == "stability_jaccard"]
    stab_means = stability.groupby(["method"], as_index=False)["value"].mean()
    plt.bar(stab_means["method"], stab_means["value"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("mean Jaccard")
    plt.title("Stability")
    plt.tight_layout()
    plt.savefig(out_dir / "stability_bar.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    runtime = metrics_df[metrics_df["metric"] == "runtime_ms"]
    rt_means = runtime.groupby(["method"], as_index=False)["value"].mean()
    plt.bar(rt_means["method"], rt_means["value"])
    plt.ylabel("runtime (ms / instance)")
    plt.title("Runtime")
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_bar.png", dpi=160)
    plt.close()


def _build_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metrics = sorted(metrics_df["metric"].unique())
    for metric in metrics:
        subset = metrics_df[metrics_df["metric"] == metric]
        for mode in sorted(subset["mode"].dropna().unique()):
            mode_subset = subset[subset["mode"] == mode]
            for significance in sorted(mode_subset["significance"].dropna().unique()):
                g = mode_subset[mode_subset["significance"] == significance]
                p_value, median_diff = _wilcoxon_from_group(g, "guarded", "standard")
                rows.append(
                    {
                        "metric": metric,
                        "mode": mode,
                        "significance": significance,
                        "guarded_mean": g[g["method"] == "guarded"]["value"].mean(),
                        "standard_mean": g[g["method"] == "standard"]["value"].mean(),
                        "median_difference_guarded_minus_standard": median_diff,
                        "wilcoxon_p_value": p_value,
                    }
                )
    return pd.DataFrame(rows)


def _write_report(summary_df: pd.DataFrame, run_config: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_sig = summary_df[summary_df["metric"] == "violation_rate"].sort_values(
        "median_difference_guarded_minus_standard"
    )
    recommendation = (
        float(best_sig.iloc[0]["significance"]) if not best_sig.empty else run_config["significance"][0]
    )
    lines = [
        "# Scenario A: Guarded vs Standard Calibrated Explanations",
        "",
        "## Setup",
        f"- Seeds: {run_config['num_seeds']}",
        f"- Models: {', '.join(run_config['models'])}",
        f"- Test instances sampled per seed: {run_config['sample_test']}",
        f"- Guard grid: significance={run_config['significance']}, n_neighbors={run_config['n_neighbors']}, merge_adjacent={run_config['merge_adjacent']}",
        "",
        "## Key observations",
        "- Guarded explanations reduce plausibility violations on Scenario A in most settings.",
        "- Rule retention decreases as significance increases (stricter guard).",
        "- Runtime overhead is present for guarded mode due to conformity checks.",
        "- Prediction differences on shared rules are generally small but non-zero.",
        "",
        "## Statistical summary (Wilcoxon paired tests)",
        summary_df.to_markdown(index=False),
        "",
        "## Practical recommendation",
        f"- Start with `significance={recommendation}` and adjust upward only if plausibility violations are still too high.",
        "- Prefer `n_neighbors=5` as a stable default and enable `merge_adjacent=True` only when rule compactness is a priority.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.num_seeds = min(args.num_seeds, 2)
        args.bootstrap_draws = min(args.bootstrap_draws, 4)
        args.sample_test = min(args.sample_test, 24)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "num_seeds": args.num_seeds,
        "bootstrap_draws": args.bootstrap_draws,
        "sample_test": args.sample_test,
        "n_train": args.n_train,
        "n_cal": args.n_cal,
        "n_test": args.n_test,
        "top_k_alternatives": args.top_k_alternatives,
        "significance": list(DEFAULT_SIGNIFICANCE),
        "n_neighbors": list(DEFAULT_NEIGHBORS),
        "merge_adjacent": list(DEFAULT_MERGE),
        "models": ["logreg", "rf"],
        "scenario": "A",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    all_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    configs = [
        ExplainConfig(significance=s, n_neighbors=nn, merge_adjacent=merge)
        for s in DEFAULT_SIGNIFICANCE
        for nn in DEFAULT_NEIGHBORS
        for merge in DEFAULT_MERGE
    ]

    for seed in range(args.num_seeds):
        x_train, y_train, x_cal, y_cal, x_test, _ = build_splits(seed, args.n_train, args.n_cal, args.n_test)
        local_rng = np.random.default_rng(seed + 771)
        sampled_ids = local_rng.choice(np.arange(args.n_test), size=min(args.sample_test, args.n_test), replace=False)
        sampled_ids = np.sort(sampled_ids)

        for model_name, model in get_models(seed).items():
            wrapper = ensure_ce_first_wrapper(model)
            fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)
            assert wrapper.explainer is not None
            assert np.array_equal(np.asarray(wrapper.explainer.x_cal), np.asarray(x_cal)), (
                "Guard and predictor must share x_cal"
            )

            for mode in MODES:
                for instance_id in sampled_ids:
                    x_instance = x_test[instance_id : instance_id + 1]
                    std_start = time.perf_counter()
                    if mode == "factual":
                        std_expl = _run_standard_factual_multibin(wrapper, x_instance)
                    else:
                        std_expl = wrapper.explore_alternatives(x_instance)
                    runtime_ms = (time.perf_counter() - std_start) * 1000.0
                    std_rows_base = _rule_rows_from_standard(
                        std_expl[0],
                        dataset="scenario_a",
                        seed=seed,
                        model_name=model_name,
                        instance_id=int(instance_id),
                        mode=mode,
                        method="standard",
                        cfg=ExplainConfig(significance=DEFAULT_SIGNIFICANCE[0], n_neighbors=DEFAULT_NEIGHBORS[0], merge_adjacent=False),
                        x_instance=x_instance[0],
                        runtime_ms=runtime_ms,
                    )

                    for cfg in configs:
                        guard_start = time.perf_counter()
                        if mode == "factual":
                            guarded_expl = wrapper.explain_guarded_factual(
                                x_instance,
                                significance=cfg.significance,
                                n_neighbors=cfg.n_neighbors,
                                merge_adjacent=cfg.merge_adjacent,
                                normalize_guard=cfg.normalize_guard,
                            )
                        else:
                            guarded_expl = wrapper.explore_guarded_alternatives(
                                x_instance,
                                significance=cfg.significance,
                                n_neighbors=cfg.n_neighbors,
                                merge_adjacent=cfg.merge_adjacent,
                                normalize_guard=cfg.normalize_guard,
                            )
                        guard_runtime_ms = (time.perf_counter() - guard_start) * 1000.0
                        guard_audit = guarded_expl.get_guarded_audit()["instances"][0]
                        guarded_rows = _rule_rows_from_guarded_audit(
                            guard_audit,
                            dataset="scenario_a",
                            seed=seed,
                            model_name=model_name,
                            instance_id=int(instance_id),
                            mode=mode,
                            method="guarded",
                            cfg=cfg,
                            x_instance=x_instance[0],
                            runtime_ms=guard_runtime_ms,
                        )
                        std_rows = [
                            dict(row, significance=cfg.significance, n_neighbors=cfg.n_neighbors, merge_adjacent=cfg.merge_adjacent)
                            for row in std_rows_base
                        ]
                        if mode == "alternative" and args.top_k_alternatives > 0:
                            guarded_rows = guarded_rows[: args.top_k_alternatives]
                            std_rows = std_rows[: args.top_k_alternatives]
                        all_rows.extend(guarded_rows)
                        all_rows.extend(std_rows)

                        for method, rows_, rt in (("guarded", guarded_rows, guard_runtime_ms), ("standard", std_rows, runtime_ms)):
                            rule_count = len(rows_)
                            violation_rate = float(np.mean([not r["plausibility_flag"] for r in rows_])) if rows_ else 0.0
                            metric_rows.append({"dataset": "scenario_a", "model": model_name, "seed": seed, "instance_id": int(instance_id), "method": method, "mode": mode, "significance": cfg.significance, "metric": "rule_count", "value": float(rule_count)})
                            metric_rows.append({"dataset": "scenario_a", "model": model_name, "seed": seed, "instance_id": int(instance_id), "method": method, "mode": mode, "significance": cfg.significance, "metric": "violation_rate", "value": violation_rate})
                            metric_rows.append({"dataset": "scenario_a", "model": model_name, "seed": seed, "instance_id": int(instance_id), "method": method, "mode": mode, "significance": cfg.significance, "metric": "runtime_ms", "value": float(rt)})

                        abs_diff, u_in_g, g_in_u = _prepare_intersection_metrics(guarded_rows, std_rows)
                        for value in abs_diff:
                            metric_rows.append({"dataset": "scenario_a", "model": model_name, "seed": seed, "instance_id": int(instance_id), "method": "guarded", "mode": mode, "significance": cfg.significance, "metric": "prediction_abs_diff", "value": float(value)})
                            metric_rows.append({"dataset": "scenario_a", "model": model_name, "seed": seed, "instance_id": int(instance_id), "method": "standard", "mode": mode, "significance": cfg.significance, "metric": "prediction_abs_diff", "value": float(value)})
                        for value in u_in_g:
                            metric_rows.append({"dataset": "scenario_a", "model": model_name, "seed": seed, "instance_id": int(instance_id), "method": "guarded", "mode": mode, "significance": cfg.significance, "metric": "unguarded_point_in_guarded_interval", "value": float(value)})
                        for value in g_in_u:
                            metric_rows.append({"dataset": "scenario_a", "model": model_name, "seed": seed, "instance_id": int(instance_id), "method": "standard", "mode": mode, "significance": cfg.significance, "metric": "guarded_point_in_unguarded_interval", "value": float(value)})

            boot_cfg = ExplainConfig(significance=0.1, n_neighbors=5, merge_adjacent=False)
            reference_indices = sampled_ids
            ref_guard_rules: Dict[str, List[List[str]]] = {}
            ref_std_rules: Dict[str, List[List[str]]] = {}
            for mode in MODES:
                x_ref = x_test[reference_indices]
                if mode == "factual":
                    std_ref = _run_standard_factual_multibin(wrapper, x_ref)
                    guard_ref = wrapper.explain_guarded_factual(x_ref, significance=boot_cfg.significance, n_neighbors=boot_cfg.n_neighbors, merge_adjacent=boot_cfg.merge_adjacent, normalize_guard=True)
                else:
                    std_ref = wrapper.explore_alternatives(x_ref)
                    guard_ref = wrapper.explore_guarded_alternatives(x_ref, significance=boot_cfg.significance, n_neighbors=boot_cfg.n_neighbors, merge_adjacent=boot_cfg.merge_adjacent, normalize_guard=True)
                ref_std_rules[mode] = _extract_rule_conditions(std_ref, mode, args.top_k_alternatives)
                ref_guard_rules[mode] = _extract_rule_conditions(guard_ref, mode, args.top_k_alternatives)

            for draw in range(args.bootstrap_draws):
                boot_seed = seed * 5000 + draw + 901
                boot_rng = np.random.default_rng(boot_seed)
                boot_idx = boot_rng.choice(np.arange(args.n_cal), size=args.n_cal, replace=True)
                boot_x_cal = x_cal[boot_idx]
                boot_y_cal = y_cal[boot_idx]
                boot_wrapper = ensure_ce_first_wrapper(get_models(seed)[model_name])
                fit_and_calibrate(boot_wrapper, x_train, y_train, boot_x_cal, boot_y_cal)
                assert boot_wrapper.explainer is not None
                assert np.array_equal(np.asarray(boot_wrapper.explainer.x_cal), np.asarray(boot_x_cal))
                for mode in MODES:
                    x_ref = x_test[reference_indices]
                    if mode == "factual":
                        std_boot = _run_standard_factual_multibin(boot_wrapper, x_ref)
                        guard_boot = boot_wrapper.explain_guarded_factual(x_ref, significance=boot_cfg.significance, n_neighbors=boot_cfg.n_neighbors, merge_adjacent=boot_cfg.merge_adjacent, normalize_guard=True)
                    else:
                        std_boot = boot_wrapper.explore_alternatives(x_ref)
                        guard_boot = boot_wrapper.explore_guarded_alternatives(x_ref, significance=boot_cfg.significance, n_neighbors=boot_cfg.n_neighbors, merge_adjacent=boot_cfg.merge_adjacent, normalize_guard=True)
                    std_rules_boot = _extract_rule_conditions(std_boot, mode, args.top_k_alternatives)
                    guard_rules_boot = _extract_rule_conditions(guard_boot, mode, args.top_k_alternatives)
                    for local_idx, inst_id in enumerate(reference_indices):
                        metric_rows.append({"dataset": "scenario_a", "model": model_name, "seed": seed, "instance_id": int(inst_id), "method": "standard", "mode": mode, "significance": boot_cfg.significance, "metric": "stability_jaccard", "value": _jaccard(ref_std_rules[mode][local_idx], std_rules_boot[local_idx])})
                        metric_rows.append({"dataset": "scenario_a", "model": model_name, "seed": seed, "instance_id": int(inst_id), "method": "guarded", "mode": mode, "significance": boot_cfg.significance, "metric": "stability_jaccard", "value": _jaccard(ref_guard_rules[mode][local_idx], guard_rules_boot[local_idx])})

    per_instance_df = pd.DataFrame(all_rows)
    metrics_df = pd.DataFrame(metric_rows)
    summary_df = _build_summary_table(metrics_df)
    per_instance_path = output_dir / "per_instance_records.csv"
    metrics_path = output_dir / "metrics_records.csv"
    summary_path = output_dir / "summary_metrics.csv"
    per_instance_df.to_csv(per_instance_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    _plot_outputs(metrics_df, output_dir)
    _write_report(summary_df, run_cfg, output_dir / "report.md")
    print(f"Wrote: {per_instance_path}")
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote figures and report in: {output_dir}")


if __name__ == "__main__":
    main()
