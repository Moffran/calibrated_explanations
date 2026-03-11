"""Scenario H: NCF and weight grid across binary and multiclass datasets."""

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
    """Sweep public reject NCF variants across task types."""
    rows: list[dict[str, float | str]] = []
    datasets = task_specs("binary", quick=config.quick) + task_specs("multiclass", quick=config.quick)
    for spec in datasets:
        bundle = build_classification_bundle(spec, config)
        baseline_accuracy = float(np.mean(bundle.baseline_pred == bundle.y_test))
        for ncf in ("default", "entropy", "ensured"):
            for w in (0.3, 0.5, 0.7, 1.0):
                result = bundle.wrapper.predict(
                    bundle.x_test,
                    reject_policy=RejectPolicySpec.flag(ncf=ncf, w=w),
                    confidence=0.95,
                )
                metadata = result.metadata or {}
                rejected = np.asarray(result.rejected, dtype=bool)
                accepted = ~rejected
                acc = accepted_accuracy(bundle.y_test, bundle.baseline_pred, accepted)
                rows.append(
                    {
                        "task_type": spec.task_type,
                        "dataset": spec.name,
                        "ncf": ncf,
                        "w": float(w),
                        "accept_rate": float(np.mean(accepted)),
                        "accepted_accuracy": acc,
                        "accepted_accuracy_delta": acc - baseline_accuracy if np.isfinite(acc) else float("nan"),
                    }
                )

    df = pd.DataFrame(rows)
    meta = {
        "scenario": "scenario_h_ncf_grid",
        "display_name": "Scenario H — NCF grid",
        "quick": config.quick,
        "highlights": [
            "This grid compares default, ensured, and legacy entropy-mapped reject NCFs across binary and multiclass settings.",
            "Accepted accuracy delta is always empirical and benchmarked against the non-reject baseline.",
        ],
        "outcome": {
            "rows": int(len(df)),
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "best_accuracy_delta": float(df["accepted_accuracy_delta"].max()) if not df.empty else float("nan"),
        },
    }
    write_csv_json_md("scenario_h_ncf_grid", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
