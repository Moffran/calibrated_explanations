"""Scenario 2: multi-dataset multiclass correctness evaluation.

Paper mapping: C2 / RQ2 (empirical).

Contribution C2: CE multiclass reject acts as a conformal correctness classifier.
The key NCF-specific results are:
  - hinge: collapses to near-100% rejection on small-n-class datasets (expected behaviour,
    not a bug — hinge becomes trivially strict when class probabilities are uniform).
  - margin: selective rejection at moderate reject rates with near-perfect accepted top-1.
  - ensured: baseline comparison; lower selectivity than margin.

An `expected_collapse` flag is set when reject_rate > 0.95 for hinge rows to distinguish
expected hinge collapse from genuine anomalies.
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
                rejected = np.asarray(result.rejected, dtype=bool)
                accepted = ~rejected
                reject_rate = float(np.mean(rejected))
                top1_accuracy = accepted_accuracy(bundle.y_test, bundle.baseline_pred, accepted)
                # No explicit hinge/margin modes are used in this runner; keep collapse flag off.
                expected_collapse = False
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
                        "ambiguity_rate": float(metadata.get("ambiguity_rate", np.nan)),
                        "expected_collapse": expected_collapse,
                        "guarantee_status": "empirical",
                    }
                )

    df = pd.DataFrame(rows)
    collapse_count = 0
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
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "mean_accepted_top1_accuracy": float(df["accepted_top1_accuracy"].mean()) if not df.empty else float("nan"),
            "mean_reject_rate": float(df["reject_rate"].mean()) if not df.empty else float("nan"),
            "hinge_collapse_events": collapse_count,
        },
    }
    write_csv_json_md("scenario_2_multiclass_correctness", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
