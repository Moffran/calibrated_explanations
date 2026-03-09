"""Scenario F: multi-dataset multiclass correctness evaluation."""

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
    rows: list[dict[str, float | str | int]] = []
    for spec in task_specs("multiclass", quick=config.quick):
        bundle = build_classification_bundle(spec, config)
        for epsilon in (0.05, 0.10):
            confidence = 1.0 - float(epsilon)
            for ncf in ("hinge", "margin"):
                result = bundle.wrapper.predict(
                    bundle.x_test,
                    reject_policy=RejectPolicySpec.flag(ncf=ncf, w=0.5),
                    confidence=confidence,
                )
                metadata = result.metadata or {}
                rejected = np.asarray(result.rejected, dtype=bool)
                accepted = ~rejected
                top1_accuracy = accepted_accuracy(bundle.y_test, bundle.baseline_pred, accepted)
                rows.append(
                    {
                        "dataset": spec.name,
                        "epsilon": float(epsilon),
                        "ncf": ncf,
                        "n_cal": int(len(bundle.x_cal)),
                        "n_test": int(len(bundle.x_test)),
                        "accepted_top1_accuracy": top1_accuracy,
                        "reject_rate": float(metadata.get("reject_rate", np.nan)),
                        "ambiguity_rate": float(metadata.get("ambiguity_rate", np.nan)),
                        "guarantee_status": "empirical",
                    }
                )

    df = pd.DataFrame(rows)
    meta = {
        "scenario": "scenario_f_multiclass_coverage",
        "display_name": "Scenario F — Multiclass correctness evaluation",
        "quick": config.quick,
        "highlights": [
            "Accepted top-1 accuracy is reported empirically while the formal guarantee remains a proof obligation.",
            "The artifact explicitly marks `guarantee_status=empirical` to avoid over-claiming.",
            "This scenario evaluates CE multiclass reject as a correctness classifier, not as a K-class prediction-set method.",
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "mean_accepted_top1_accuracy": float(df["accepted_top1_accuracy"].mean()) if not df.empty else float("nan"),
            "mean_reject_rate": float(df["reject_rate"].mean()) if not df.empty else float("nan"),
        },
    }
    write_csv_json_md("scenario_f_multiclass_coverage", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
