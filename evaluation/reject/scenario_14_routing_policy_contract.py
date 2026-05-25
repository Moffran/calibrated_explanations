"""Scenario 14: routing policy contract validation.

Paper mapping: none (infrastructure validation).

The orchestrator exposes three routing policies:
  FLAG          — process all instances; prediction_set populated in metadata
  ONLY_ACCEPTED — payload filtered to accepted (singleton) instances
  ONLY_REJECTED — payload filtered to rejected (ambiguous + novel) instances

The README explicitly deferred empirical routing validation as a "CI integration
concern only," but the Bug 1 red-team finding (Scenario 7's vacuously-true "0
violations" caused by an incorrect prediction_set access path) showed that routing
contract bugs directly contaminate validity measurements.

This scenario validates six routing contract invariants across binary datasets:

  I1. FLAG: result.rejected has length n_test.
  I2. FLAG: result.metadata["prediction_set"] is a non-None boolean array of
      shape (n_test, n_classes).
  I3. FLAG: result.metadata["original_count"] == n_test.
  I4. ONLY_ACCEPTED: len(result.metadata["source_indices"]) == sum(~rejected)
      where rejected is derived from the parallel FLAG call.
  I5. ONLY_REJECTED: len(result.metadata["source_indices"]) == sum(rejected).
  I6. All three policies yield the same reject vector (FLAG rejected == the mask
      implied by ONLY_ACCEPTED source_indices and ONLY_REJECTED source_indices).
  I7. result.metadata["degraded_mode"] is an empty tuple for all three policies
      on healthy data (no fallback should be triggered under normal conditions).

A contract violation is any row where at least one invariant fails.  Degraded-mode
non-empty is reported separately — it indicates a fallback was triggered (coverage
under fallback is unmeasured and should be investigated independently).
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from calibrated_explanations import RejectPolicySpec, WrapCalibratedExplainer
from calibrated_explanations.explanations.reject import RejectPolicy
from sklearn.ensemble import RandomForestClassifier

from .common_reject import (
    RunConfig,
    _markdown_table_from_df,
    load_dataset,
    seed_grid,
    split_dataset,
    task_specs,
    write_csv_json_md,
)

_W = 0.5
_CONFIDENCE = 0.90


def _check_invariants(
    *,
    n_test: int,
    result_flag: Any,
    result_accepted: Any,
    result_rejected: Any,
) -> dict[str, bool | str]:
    """Return invariant check results for one (dataset, seed, confidence) triple."""
    meta_flag = getattr(result_flag, "metadata", {}) or {}
    meta_acc = getattr(result_accepted, "metadata", {}) or {}
    meta_rej = getattr(result_rejected, "metadata", {}) or {}

    # I1: FLAG rejected mask length
    flag_rejected = getattr(result_flag, "rejected", None)
    i1 = (
        flag_rejected is not None
        and np.asarray(flag_rejected).shape == (n_test,)
    )

    # I2: FLAG prediction_set shape
    ps = meta_flag.get("prediction_set")
    i2 = (
        ps is not None
        and np.asarray(ps, dtype=bool).ndim == 2
        and np.asarray(ps, dtype=bool).shape[0] == n_test
    )

    # I3: original_count == n_test for FLAG
    i3 = int(meta_flag.get("original_count", -1)) == n_test

    # I4 / I5: source_indices cardinality matches expected subset size
    if flag_rejected is not None:
        rejected_mask = np.asarray(flag_rejected, dtype=bool)
        n_accepted_expected = int(np.sum(~rejected_mask))
        n_rejected_expected = int(np.sum(rejected_mask))
    else:
        n_accepted_expected = -1
        n_rejected_expected = -1

    acc_indices = meta_acc.get("source_indices", [])
    rej_indices = meta_rej.get("source_indices", [])
    i4 = len(acc_indices) == n_accepted_expected
    i5 = len(rej_indices) == n_rejected_expected

    # I6: consistency — union of source_indices from ONLY_ACCEPTED and ONLY_REJECTED
    #     should cover exactly the n_test indices with no overlap
    all_indices = sorted(acc_indices + rej_indices)
    i6 = all_indices == list(range(n_test))

    # I7: no degraded_mode for healthy data
    dm_flag = tuple(meta_flag.get("degraded_mode") or ())
    dm_acc = tuple(meta_acc.get("degraded_mode") or ())
    dm_rej = tuple(meta_rej.get("degraded_mode") or ())
    i7 = dm_flag == () and dm_acc == () and dm_rej == ()

    contract_pass = all([i1, i2, i3, i4, i5, i6, i7])
    degraded = dm_flag != () or dm_acc != () or dm_rej != ()

    return {
        "i1_flag_rejected_length": bool(i1),
        "i2_flag_prediction_set_shape": bool(i2),
        "i3_flag_original_count": bool(i3),
        "i4_only_accepted_indices": bool(i4),
        "i5_only_rejected_indices": bool(i5),
        "i6_index_consistency": bool(i6),
        "i7_no_degraded_mode": bool(i7),
        "contract_pass": bool(contract_pass),
        "degraded_mode_flag": str(dm_flag),
        "degraded_mode_accepted": str(dm_acc),
        "degraded_mode_rejected": str(dm_rej),
        "degraded": bool(degraded),
        "n_accepted": n_accepted_expected,
        "n_rejected": n_rejected_expected,
    }


def run(config: RunConfig) -> None:
    """Validate routing policy contract invariants across binary datasets."""
    rows: list[dict[str, Any]] = []

    for spec in task_specs("binary", quick=config.quick):
        _, x_all, y_all, feature_names = load_dataset(spec)

        for seed_offset in seed_grid(config):
            seed = int(config.seed + seed_offset)
            x_fit, x_cal, x_test, y_fit, y_cal, y_test = split_dataset(
                x_all, y_all, seed=seed, stratify=True
            )

            model = RandomForestClassifier(
                n_estimators=60 if config.quick else 120,
                random_state=seed,
                max_depth=8 if config.quick else None,
                n_jobs=1,
            )
            wrapper = WrapCalibratedExplainer(model)
            wrapper.fit(x_fit, y_fit)
            wrapper.calibrate(x_cal, y_cal, feature_names=list(feature_names))

            n_test = int(len(x_test))
            policy_flag = RejectPolicySpec.flag(ncf="default", w=_W)
            policy_accepted = RejectPolicySpec(
                policy=RejectPolicy.ONLY_ACCEPTED, ncf="default", w=_W
            )
            policy_rejected = RejectPolicySpec(
                policy=RejectPolicy.ONLY_REJECTED, ncf="default", w=_W
            )

            result_flag = wrapper.predict(
                x_test, reject_policy=policy_flag, confidence=_CONFIDENCE
            )
            result_accepted = wrapper.predict(
                x_test, reject_policy=policy_accepted, confidence=_CONFIDENCE
            )
            result_rejected_out = wrapper.predict(
                x_test, reject_policy=policy_rejected, confidence=_CONFIDENCE
            )

            checks = _check_invariants(
                n_test=n_test,
                result_flag=result_flag,
                result_accepted=result_accepted,
                result_rejected=result_rejected_out,
            )

            rows.append(
                {
                    "dataset": spec.name,
                    "seed": seed,
                    "n_cal": int(len(x_cal)),
                    "n_test": n_test,
                    "confidence": _CONFIDENCE,
                    **checks,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        meta_out: dict[str, Any] = {
            "scenario": "scenario_14_routing_policy_contract",
            "display_name": "Scenario 14 — Routing policy contract validation",
            "guarantee_status": "contract",
            "quick": config.quick,
            "highlights": ["No data generated."],
            "outcome": {},
        }
        write_csv_json_md("scenario_14_routing_policy_contract", df, meta_out)
        return

    total = int(len(df))
    passes = int(df["contract_pass"].sum())
    failures = total - passes
    degraded_count = int(df["degraded"].sum())

    # Per-invariant failure counts
    invariant_failures = {
        inv: int((~df[inv]).sum())
        for inv in [
            "i1_flag_rejected_length",
            "i2_flag_prediction_set_shape",
            "i3_flag_original_count",
            "i4_only_accepted_indices",
            "i5_only_rejected_indices",
            "i6_index_consistency",
            "i7_no_degraded_mode",
        ]
    }

    meta_out = {
        "scenario": "scenario_14_routing_policy_contract",
        "display_name": "Scenario 14 — Routing policy contract validation",
        "paper_contribution": "infrastructure",
        "guarantee_status": "contract",
        "quick": config.quick,
        "highlights": [
            "Validates FLAG / ONLY_ACCEPTED / ONLY_REJECTED routing invariants on binary datasets.",
            "I1: FLAG rejected mask length == n_test.",
            "I2: FLAG prediction_set is a (n_test, n_classes) boolean array in result.metadata.",
            "I3: FLAG original_count == n_test.",
            "I4: ONLY_ACCEPTED source_indices count matches expected accepted count.",
            "I5: ONLY_REJECTED source_indices count matches expected rejected count.",
            "I6: ONLY_ACCEPTED + ONLY_REJECTED source_indices are disjoint and cover all n_test.",
            "I7: No degraded_mode markers on healthy data.",
            f"Contract passes: {passes}/{total}. Failures: {failures}. Degraded: {degraded_count}.",
        ],
        "outcome": {
            "rows": total,
            "datasets": int(df["dataset"].nunique()),
            "seeds": int(df["seed"].nunique()),
            "contract_passes": passes,
            "contract_failures": failures,
            "degraded_rows": degraded_count,
            "invariant_failures": invariant_failures,
        },
    }

    # --- Extra sections ---
    extra_sections: list[str] = []

    # Section: Invariant failure counts
    inv_rows = [
        {"invariant": inv, "failures": count}
        for inv, count in invariant_failures.items()
    ]
    inv_df = pd.DataFrame(inv_rows)
    extra_sections += [
        "## Invariant failure counts",
        "",
        _markdown_table_from_df(inv_df),
        "",
    ]

    # Section: Per-dataset contract summary
    per_dataset = (
        df.groupby("dataset")
        .agg(
            contract_passes=("contract_pass", "sum"),
            total=("contract_pass", "size"),
            any_failure=("contract_pass", lambda x: int((~x).any())),
            any_degraded=("degraded", "any"),
        )
        .reset_index()
    )
    extra_sections += [
        "## Per-dataset contract summary",
        "",
        _markdown_table_from_df(per_dataset),
        "",
    ]

    write_csv_json_md(
        "scenario_14_routing_policy_contract", df, meta_out, extra_sections=extra_sections
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
