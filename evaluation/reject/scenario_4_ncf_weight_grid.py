"""Scenario 4: NCF and blend-weight grid across binary and multiclass datasets.

Paper mapping: C2 / RQ4 (empirical).

Key findings this scenario must reproduce:
  - margin at w=0.3 collapses to near-zero accept rates on some datasets.
  - hinge generally achieves higher accepted accuracy delta than ensured.
  - w >= 0.7 converges NCF behavior toward hinge-like selectivity.

Notes:
  - Column is named `accept_rate` (not `coverage`) because this measures the fraction of
    test instances accepted — not ICP label-set coverage.  Conflating the two is an error
    flagged in the paper.
  - Entropy NCF is excluded from this suite.
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

_NCFS = ("default", "ensured")
_W_VALUES = (0.3, 0.5, 0.7, 1.0)


def run(config: RunConfig) -> None:
    """Sweep public reject NCF variants across task types."""
    rows: list[dict[str, float | str]] = []
    datasets = task_specs("binary", quick=config.quick) + task_specs("multiclass", quick=config.quick)
    for spec in datasets:
        bundle = build_classification_bundle(spec, config)
        baseline_accuracy = float(np.mean(bundle.baseline_pred == bundle.y_test))
        for ncf in _NCFS:
            for w in _W_VALUES:
                result = bundle.wrapper.predict(
                    bundle.x_test,
                    reject_policy=RejectPolicySpec.flag(ncf=ncf, w=w),
                    confidence=0.95,
                )
                rejected = np.asarray(result.rejected, dtype=bool)
                accepted = ~rejected
                acc = accepted_accuracy(bundle.y_test, bundle.baseline_pred, accepted)
                accept_rate = float(np.mean(accepted))
                rows.append(
                    {
                        "task_type": spec.task_type,
                        "dataset": spec.name,
                        "ncf": ncf,
                        "w": float(w),
                        # NOTE: `accept_rate` is the fraction of test instances accepted.
                        # This is NOT ICP label-set coverage.
                        "accept_rate": accept_rate,
                        "accepted_accuracy": acc,
                        "accepted_accuracy_delta": acc - baseline_accuracy if np.isfinite(acc) else float("nan"),
                    }
                )

    df = pd.DataFrame(rows)
    meta = {
        "scenario": "scenario_4_ncf_weight_grid",
        "display_name": "Scenario 4 — NCF and blend weight grid",
        "paper_contribution": "C2",
        "paper_rq": "RQ4",
        "guarantee_status": "empirical",
        "quick": config.quick,
        "highlights": [
            "NCFs tested: default, ensured. Entropy and explicit hinge/margin modes are excluded from this suite.",
            "accept_rate is the fraction of accepted instances — NOT ICP label-set coverage.",
            "w >= 0.7 converges NCF behavior; w=0.3 amplifies differences between NCFs where present.",
            "Accepted accuracy delta is always empirical and benchmarked against the non-reject baseline.",
        ],
        "outcome": {
            "rows": int(len(df)),
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "best_accuracy_delta": float(df["accepted_accuracy_delta"].max()) if not df.empty else float("nan"),
            "ncfs_tested": list(_NCFS),
            "w_values_tested": list(_W_VALUES),
        },
    }
    write_csv_json_md("scenario_4_ncf_weight_grid", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
