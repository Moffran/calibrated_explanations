"""Scenario 2: multi-dataset multiclass correctness evaluation.

Paper mapping: C2 / RQ2 (empirical).

Contribution C2: CE multiclass reject can optionally act as a binary correctness proxy.
Multiclass probabilities are binarized to [1-p_max, p_max] before conformal scoring
(correctness encoding: col-0 = top-1 is not correct, col-1 = top-1 is correct).
Hinge NCF is used for both 'default' and 'ensured', producing column-specific
nonconformity scores:
  - alpha[:,0] = p_max  (score for "top-1 is not correct")
  - alpha[:,1] = 1-p_max  (score for "top-1 is correct")

This enables four prediction set outcomes:
  - {1} singleton: positive correctness-proxy singleton; accepted for top-1 use
  - {0} singleton: negative correctness-proxy singleton; top-1 is not accepted
  - {0,1}: ambiguity; the correctness proxy cannot distinguish the two events
  - {}: novelty; neither correctness-proxy event is conforming

This scenario intentionally opts into the multiclass-only
experimental.multiclass_top1_correctness strategy, where accepted instances are
restricted to {1} singletons. The reject_rate column is therefore a selective
non-acceptance rate for the top-1 prediction, not a default multiclass reject rule
and not a K-class conformal label-set reject rate. A {0} singleton is reported as a
proxy-negative singleton; it can occur even when the top-1 class remains the most
likely individual class because the other classes' probability mass is aggregated
into col-0.

An `expected_collapse` flag marks rows where reject_rate > 0.95, indicating
near-complete rejection which may occur on small or uniform-probability datasets.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from calibrated_explanations import RejectPolicySpec

from .common_reject import (
    RunConfig,
    _markdown_table_from_df,
    build_classification_bundle,
    singleton_precision_recall,
    task_specs,
    write_csv_json_md,
)


def _mean_or_nan(values: np.ndarray, mask: np.ndarray) -> float:
    """Return the masked mean or NaN when the mask is empty."""
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.asarray(values)[mask]))


def proxy_correctness_diagnostics(
    y_true: np.ndarray,
    top1_pred: np.ndarray,
    positive_singleton: np.ndarray,
    proxy_negative_singleton: np.ndarray,
) -> dict[str, float | int | bool]:
    """Compute empirical diagnostics in the binary top-1 correctness proxy space."""
    true_proxy_label = (np.asarray(top1_pred) == np.asarray(y_true)).astype(int)
    positive_singleton = np.asarray(positive_singleton, dtype=bool)
    proxy_negative_singleton = np.asarray(proxy_negative_singleton, dtype=bool)
    singleton_mask = positive_singleton | proxy_negative_singleton

    predicted_proxy_label = np.full(len(true_proxy_label), -1, dtype=int)
    predicted_proxy_label[positive_singleton] = 1
    predicted_proxy_label[proxy_negative_singleton] = 0
    proxy_correct = predicted_proxy_label == true_proxy_label
    singleton_metrics = singleton_precision_recall(
        np.column_stack([proxy_negative_singleton, positive_singleton]),
        true_proxy_label,
    )

    return {
        "proxy_singleton_count": int(np.sum(singleton_mask)),
        "proxy_singleton_accuracy_defined": bool(np.any(singleton_mask)),
        "proxy_singleton_accuracy": _mean_or_nan(proxy_correct, singleton_mask),
        **singleton_metrics,
        "accepted_top1_accuracy": _mean_or_nan(true_proxy_label == 1, positive_singleton),
        "proxy_negative_singleton_accuracy": _mean_or_nan(
            true_proxy_label == 0,
            proxy_negative_singleton,
        ),
    }


def run(config: RunConfig) -> None:
    """Measure empirical multiclass correctness-proxy behavior."""
    rows: list[dict[str, float | str | int | bool]] = []
    for spec in task_specs("multiclass", quick=config.quick):
        bundle = build_classification_bundle(spec, config)
        for epsilon in (0.05, 0.10):
            confidence = 1.0 - float(epsilon)
            for ncf in ("default", "ensured"):
                result = bundle.wrapper.predict(
                    bundle.x_test,
                    reject_policy=RejectPolicySpec.flag(ncf=ncf, w=0.5),
                    strategy="experimental.multiclass_top1_correctness",
                    confidence=confidence,
                )
                metadata = result.metadata or {}
                set_sizes = np.asarray(
                    metadata.get(
                        "prediction_set_size",
                        np.zeros(len(bundle.x_test), dtype=int),
                    ),
                    dtype=int,
                ).reshape(-1)
                prediction_set_raw = metadata.get("prediction_set")

                if prediction_set_raw is not None:
                    prediction_set = np.asarray(prediction_set_raw, dtype=bool)
                    # {1}: positive correctness-proxy singleton for the top-1 class.
                    positive_singleton = (set_sizes == 1) & prediction_set[:, 1]
                    # {0}: negative correctness-proxy singleton, not an alternative-label set.
                    proxy_negative_singleton = (set_sizes == 1) & ~prediction_set[:, 1]
                else:
                    # Fallback when prediction_set is unavailable: use rejected mask;
                    # cannot distinguish {0} from {1} singletons in this path.
                    rejected_fallback = np.asarray(result.rejected, dtype=bool)
                    positive_singleton = ~rejected_fallback
                    proxy_negative_singleton = np.zeros(len(bundle.x_test), dtype=bool)

                # Accepted = only {1} correctness-proxy singletons.
                accepted = positive_singleton
                non_accepted_rate = float(np.mean(~accepted))
                diagnostics = proxy_correctness_diagnostics(
                    bundle.y_test,
                    bundle.baseline_pred,
                    positive_singleton,
                    proxy_negative_singleton,
                )
                positive_singleton_rate = float(np.mean(positive_singleton))
                proxy_negative_singleton_rate = float(np.mean(proxy_negative_singleton))

                # Collapse: near-total rejection (can occur on small or uniform datasets)
                expected_collapse = non_accepted_rate > 0.95

                ambiguity_mask_raw = metadata.get("ambiguity_mask")
                if ambiguity_mask_raw is not None:
                    ambiguity_rate = float(np.mean(np.asarray(ambiguity_mask_raw, dtype=bool)))
                else:
                    ambiguity_rate = float(metadata.get("ambiguity_rate", np.nan))

                novelty_mask_raw = metadata.get("novelty_mask")
                if novelty_mask_raw is not None:
                    novelty_rate = float(np.mean(np.asarray(novelty_mask_raw, dtype=bool)))
                else:
                    novelty_rate = float(metadata.get("novelty_rate", np.nan))

                rows.append(
                    {
                        "dataset": spec.name,
                        "epsilon": float(epsilon),
                        "ncf": ncf,
                        "n_cal": int(len(bundle.x_cal)),
                        "n_test": int(len(bundle.x_test)),
                        "n_classes": int(len(np.unique(bundle.y_test))),
                        "proxy_singleton_accuracy": diagnostics["proxy_singleton_accuracy"],
                        "proxy_singleton_accuracy_defined": diagnostics[
                            "proxy_singleton_accuracy_defined"
                        ],
                        "proxy_singleton_count": diagnostics["proxy_singleton_count"],
                        "singleton_precision": diagnostics["singleton_precision"],
                        "singleton_recall": diagnostics["singleton_recall"],
                        "singleton_correct_count": diagnostics["singleton_correct_count"],
                        "singleton_count": diagnostics["singleton_count"],
                        "singleton_precision_recall_defined": diagnostics[
                            "singleton_precision_recall_defined"
                        ],
                        "accepted_top1_accuracy": diagnostics["accepted_top1_accuracy"],
                        "proxy_negative_singleton_accuracy": diagnostics[
                            "proxy_negative_singleton_accuracy"
                        ],
                        "non_accepted_rate": non_accepted_rate,
                        "reject_rate": non_accepted_rate,
                        "positive_singleton_rate": positive_singleton_rate,
                        "correct_singleton_rate": positive_singleton_rate,
                        "proxy_negative_singleton_rate": proxy_negative_singleton_rate,
                        "error_singleton_rate": proxy_negative_singleton_rate,
                        "ambiguity_rate": ambiguity_rate,
                        "novelty_rate": novelty_rate,
                        "expected_collapse": expected_collapse,
                        "guarantee_status": "empirical",
                    }
                )

    df = pd.DataFrame(rows)
    collapse_count = int(df["expected_collapse"].sum()) if not df.empty else 0
    meta = {
        "scenario": "scenario_2_multiclass_correctness",
        "display_name": "Scenario 2 - Multiclass correctness proxy",
        "paper_contribution": "C2",
        "paper_rq": "RQ2",
        "guarantee_status": "empirical",
        "quick": config.quick,
        "highlights": [
            "Primary empirical accuracy is computed in the binary proxy space: singleton {1}/{0} is compared with 1[top-1 prediction is correct].",
            "Accepted top-1 accuracy remains a precision-style diagnostic on {1} rows only; it is not the proxy classifier accuracy.",
            "This scenario opts into the multiclass-only experimental.multiclass_top1_correctness strategy.",
            "It evaluates CE multiclass reject as a binary correctness proxy, not a default rule and not a K-class prediction-set method.",
            "Accepted instances are restricted to {1} positive correctness-proxy singletons.",
            "{0} singletons are proxy-negative singletons: the aggregate non-top1 event is conforming, but no specific alternative class is selected.",
            "reject_rate is retained as a compatibility alias for non_accepted_rate in this proxy scenario.",
            "Hinge NCF is used for both 'default' and 'ensured' paths. Margin NCF was removed (it produced identical scores for both columns, making singletons impossible).",
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "mean_proxy_singleton_accuracy": (
                float(df["proxy_singleton_accuracy"].mean()) if not df.empty else float("nan")
            ),
            "mean_singleton_precision": (
                float(df["singleton_precision"].mean()) if not df.empty else float("nan")
            ),
            "mean_singleton_recall": (
                float(df["singleton_recall"].mean()) if not df.empty else float("nan")
            ),
            "mean_accepted_top1_accuracy": (
                float(df["accepted_top1_accuracy"].mean()) if not df.empty else float("nan")
            ),
            "mean_proxy_negative_singleton_accuracy": (
                float(df["proxy_negative_singleton_accuracy"].mean())
                if not df.empty
                else float("nan")
            ),
            "mean_non_accepted_rate": (
                float(df["non_accepted_rate"].mean()) if not df.empty else float("nan")
            ),
            "mean_reject_rate": float(df["reject_rate"].mean()) if not df.empty else float("nan"),
            "mean_positive_singleton_rate": (
                float(df["positive_singleton_rate"].mean()) if not df.empty else float("nan")
            ),
            "mean_correct_singleton_rate": (
                float(df["correct_singleton_rate"].mean()) if not df.empty else float("nan")
            ),
            "mean_proxy_negative_singleton_rate": (
                float(df["proxy_negative_singleton_rate"].mean()) if not df.empty else float("nan")
            ),
            "mean_error_singleton_rate": (
                float(df["error_singleton_rate"].mean()) if not df.empty else float("nan")
            ),
            "collapse_events": collapse_count,
        },
    }
    # --- Extra sections ---
    extra_sections: list[str] = []

    if not df.empty:
        # Section: Per-dataset proxy accuracy (mean over epsilon for each ncf)
        per_dataset = (
            df.groupby(["dataset", "ncf"])
            .agg(
                n_classes=("n_classes", "first"),
                proxy_singleton_accuracy=("proxy_singleton_accuracy", "mean"),
                singleton_precision=("singleton_precision", "mean"),
                singleton_recall=("singleton_recall", "mean"),
                non_accepted_rate=("non_accepted_rate", "mean"),
            )
            .reset_index()
        )
        extra_sections += [
            "## Per-dataset proxy accuracy",
            "",
            _markdown_table_from_df(per_dataset),
            "",
        ]

        # Section: NCF comparison
        ncf_comp = (
            df.groupby("ncf")
            .agg(
                mean_proxy_singleton_accuracy=("proxy_singleton_accuracy", "mean"),
                mean_singleton_precision=("singleton_precision", "mean"),
                mean_singleton_recall=("singleton_recall", "mean"),
                mean_non_accepted_rate=("non_accepted_rate", "mean"),
                mean_ambiguity_rate=("ambiguity_rate", "mean"),
                collapse_events=("expected_collapse", "sum"),
            )
            .reset_index()
        )
        extra_sections += [
            "## NCF comparison",
            "",
            _markdown_table_from_df(ncf_comp),
            "",
        ]

    write_csv_json_md("scenario_2_multiclass_correctness", df, meta, extra_sections=extra_sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
