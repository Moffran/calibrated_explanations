"""Scenario 2: multi-dataset multiclass correctness evaluation.

Paper mapping: C2 / RQ2 (empirical).

Contribution C2: CE multiclass reject acts as a conformal correctness classifier.
Multiclass probabilities are binarized to [1-p_max, p_max] before conformal scoring
(correctness encoding: col-0 = wrong, col-1 = correct). Hinge NCF is used for
both 'default' and 'ensured', producing column-specific nonconformity scores:
  - alpha[:,0] = p_max  (score for "wrong" class)
  - alpha[:,1] = 1-p_max  (score for "correct" class)

This enables four prediction set outcomes:
  - {1} singleton: conformal confident prediction is correct → accepted
  - {0} singleton: conformal confident prediction is wrong → error-rejected
  - {0,1}: ambiguity rejection (can't distinguish correct from wrong)
  - {}: novelty rejection (neither class is plausible)

Accepted instances are restricted to {1} singletons only. The reject rate counts
all non-{1}-singleton outcomes, including {0} singletons (error-rejected).

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
    accepted_accuracy,
    build_classification_bundle,
    task_specs,
    write_csv_json_md,
)


def run(config: RunConfig) -> None:
    """Measure empirical multiclass correctness on the accepted subset."""
    rows: list[dict[str, float | str | int | bool]] = []
    for spec in task_specs("multiclass", quick=config.quick):
        bundle = build_classification_bundle(spec, config)
        for epsilon in (0.05, 0.10):
            confidence = 1.0 - float(epsilon)
            for ncf in ("default", "ensured"):
                result = bundle.wrapper.predict(
                    bundle.x_test,
                    reject_policy=RejectPolicySpec.flag(ncf=ncf, w=0.5),
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
                    # {1} singleton: size==1 and the "correct" column is in the set
                    correct_singleton = (set_sizes == 1) & prediction_set[:, 1]
                    # {0} singleton: size==1 and only the "wrong" column is in the set
                    error_singleton = (set_sizes == 1) & ~prediction_set[:, 1]
                else:
                    # Fallback when prediction_set is unavailable: use rejected mask;
                    # cannot distinguish {0} from {1} singletons in this path.
                    rejected_fallback = np.asarray(result.rejected, dtype=bool)
                    correct_singleton = ~rejected_fallback
                    error_singleton = np.zeros(len(bundle.x_test), dtype=bool)

                # Accepted = only {1} singletons (confident correct)
                accepted = correct_singleton
                reject_rate = float(np.mean(~accepted))
                top1_accuracy = accepted_accuracy(bundle.y_test, bundle.baseline_pred, accepted)
                correct_singleton_rate = float(np.mean(correct_singleton))
                error_singleton_rate = float(np.mean(error_singleton))

                # Collapse: near-total rejection (can occur on small or uniform datasets)
                expected_collapse = reject_rate > 0.95

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
                        "accepted_top1_accuracy": top1_accuracy,
                        "reject_rate": reject_rate,
                        "correct_singleton_rate": correct_singleton_rate,
                        "error_singleton_rate": error_singleton_rate,
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
        "display_name": "Scenario 2 — Multiclass correctness classifier",
        "paper_contribution": "C2",
        "paper_rq": "RQ2",
        "guarantee_status": "empirical",
        "quick": config.quick,
        "highlights": [
            "Accepted top-1 accuracy is reported empirically; the formal guarantee remains a proof obligation.",
            "This scenario evaluates CE multiclass reject as a correctness classifier, not a K-class prediction-set method.",
            "Accepted instances are restricted to {1} singletons (confident correct); {0} singletons (confident wrong) are error-rejected.",
            "Hinge NCF is used for both 'default' and 'ensured' paths. Margin NCF was removed (it produced identical scores for both columns, making singletons impossible).",
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "mean_accepted_top1_accuracy": (
                float(df["accepted_top1_accuracy"].mean()) if not df.empty else float("nan")
            ),
            "mean_reject_rate": float(df["reject_rate"].mean()) if not df.empty else float("nan"),
            "mean_correct_singleton_rate": (
                float(df["correct_singleton_rate"].mean()) if not df.empty else float("nan")
            ),
            "mean_error_singleton_rate": (
                float(df["error_singleton_rate"].mean()) if not df.empty else float("nan")
            ),
            "collapse_events": collapse_count,
        },
    }
    write_csv_json_md("scenario_2_multiclass_correctness", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
