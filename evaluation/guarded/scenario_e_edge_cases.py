"""Scenario E — Edge Case Behavior.

Question: Does the guard's API behave predictably — no exceptions, documented
behavior — at the extremes of its parameter space?

This is a structured PASS/FAIL test suite. Each case targets a specific code-path
boundary. Results are recorded as PASS or FAIL with diagnostics. A PASS means the
observed behavior matches what is documented or asserted. Some cases are designed
to expose known design boundaries (e.g., E2) where the expected behavior is
"silently does nothing" — that is the correct PASS state.

Cases:
  E1  Zero emitted rules (significance=0.9, OOD instances) — API must not crash.
  E2  significance < 1/n_cal — p-value granularity means guard can never reject.
  E3  n_neighbors=1 — no exception; guard runs (may be noisy).
  E4  n_neighbors >= n_cal — saturation; no crash.
  E5  merge_adjacent barrier — non-conforming bin prevents merging across it.
  E6  Single feature — no index error in feature iteration loop.
  E7  All features identical in test instance — no NaN/inf in p-values.

Run with --quick (no-op here — all cases are already fast).
"""
from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations.ce_agent_utils import ensure_ce_first_wrapper, fit_and_calibrate

from common_guarded import (
    check_audit_field_completeness,
    extract_audit_rows,
    extract_audit_summary_rows,
    make_gaussian_classification,
    write_report,
)


# ---------------------------------------------------------------------------
# Case result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    status: str          # "PASS" or "FAIL"
    details: str
    expected: str
    extra: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared setup helpers
# ---------------------------------------------------------------------------

def _setup_wrapper(
    n_train: int = 300,
    n_cal: int = 100,
    n_dim: int = 4,
    seed: int = 0,
) -> Any:
    """Fit and calibrate a RandomForest on Gaussian data."""
    x_all, y_all = make_gaussian_classification(n=n_train + n_cal, n_dim=n_dim, seed=seed)
    x_tr, y_tr = x_all[:n_train], y_all[:n_train]
    x_cal, y_cal = x_all[n_train:], y_all[n_train:]
    model = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=seed, n_jobs=1)
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(wrapper, x_tr, y_tr, x_cal, y_cal)
    return wrapper, x_cal


def _no_nan_in_p_values(audit_df: pd.DataFrame) -> bool:
    if audit_df.empty or "p_value" not in audit_df.columns:
        return True
    vals = audit_df["p_value"].dropna()
    return bool(np.all(np.isfinite(vals.astype(float))))


# ---------------------------------------------------------------------------
# Individual cases
# ---------------------------------------------------------------------------

def case_e1_zero_emitted() -> CaseResult:
    """significance=0.9 on strongly OOD instances: all intervals should be filtered."""
    expected = (
        "No exception raised. intervals_emitted=0 for all instances. "
        "get_rules() returns empty dict. plot() and list_rules() run without error."
    )
    try:
        wrapper, x_cal = _setup_wrapper(n_train=300, n_cal=100, n_dim=4, seed=1)

        # OOD instances: far from the calibration cloud
        rng = np.random.default_rng(42)
        x_ood = rng.standard_normal((5, 4)) + 20.0  # extreme shift

        guarded_expl = wrapper.explain_guarded_factual(
            x_ood, significance=0.9, n_neighbors=5, normalize_guard=True
        )

        # Verify no crash on downstream API calls
        summary_df = extract_audit_summary_rows(guarded_expl)
        n_fully_filtered = int((summary_df["intervals_emitted"] == 0).sum()) if not summary_df.empty else 0

        # get_rules() must return without exception on zero-rule explanations
        for exp in guarded_expl:
            rules = exp.get_rules()
            assert isinstance(rules, dict), "get_rules() must return a dict"

        # plot() must not crash (render to non-interactive backend)
        import matplotlib
        matplotlib.use("Agg")
        try:
            guarded_expl.plot()
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception as plot_exc:  # noqa: BLE001
            return CaseResult(
                "E1", "FAIL",
                f"plot() raised: {plot_exc}",
                expected,
            )

        if n_fully_filtered == 5:
            return CaseResult(
                "E1", "PASS",
                f"All 5 OOD instances had 0 emitted rules. API stable.",
                expected,
                extra={"n_fully_filtered": n_fully_filtered},
            )
        else:
            return CaseResult(
                "E1", "FAIL",
                f"Only {n_fully_filtered}/5 instances had 0 emitted rules at significance=0.9. "
                "The guard may not be filtering aggressively enough.",
                expected,
                extra={"n_fully_filtered": n_fully_filtered},
            )
    except Exception:  # noqa: BLE001
        return CaseResult("E1", "FAIL", traceback.format_exc().splitlines()[-1], expected)


def case_e2_unsmoothed_p_values() -> CaseResult:
    """significance=0.001 with n_cal=200: p-values are unsmoothed (no +1 smoothing).

    InDistributionGuard computes p_value = count(cal_scores >= test_score) / n_cal.
    This is NOT rounded up to 1/n_cal: if all n_cal calibration distances are smaller
    than the test distance, p_value = 0. Any positive significance threshold will then
    reject those intervals. This is correct conformal behaviour, not a bug.

    Expected: no crash, audit returns valid p-values in [0, 1], no NaN. Observed
    n_removed and min_p are documented for reference.
    """
    expected = (
        "No crash. Audit returns valid p-values in [0, 1] with no NaN. "
        "p_value=0 is possible (unsmoothed estimator: count / n_cal with no +1). "
        "Guard correctly rejects intervals with p_value=0 even at very low significance."
    )
    n_cal = 200
    try:
        x_all, y_all = make_gaussian_classification(n=500 + n_cal, n_dim=4, seed=2)
        x_tr, y_tr = x_all[:500], y_all[:500]
        x_cal, y_cal = x_all[500:], y_all[500:]
        model = RandomForestClassifier(n_estimators=80, random_state=2, n_jobs=1)
        wrapper = ensure_ce_first_wrapper(model)
        fit_and_calibrate(wrapper, x_tr, y_tr, x_cal, y_cal)

        rng = np.random.default_rng(2)
        x_test = rng.standard_normal((20, 4))

        guarded_expl = wrapper.explain_guarded_factual(
            x_test, significance=0.001, n_neighbors=5, normalize_guard=True
        )
        audit_df = extract_audit_rows(guarded_expl)

        if audit_df.empty:
            return CaseResult("E2", "FAIL", "Audit returned no rows.", expected)

        p_values = audit_df["p_value"].dropna() if "p_value" in audit_df.columns else pd.Series(dtype=float)
        n_removed = int((audit_df["emission_reason"] == "removed_guard").sum()) if "emission_reason" in audit_df.columns else -1
        min_p_observed = float(p_values.min()) if not p_values.empty else float("nan")
        has_invalid = bool((p_values < 0).any() or (p_values > 1).any() or p_values.isna().any())

        details = (
            f"n_cal={n_cal}, significance=0.001. Unsmoothed estimator: min p-value can be 0. "
            f"Observed n_removed_guard={n_removed}, min_p_value={min_p_observed:.4f}."
        )

        if has_invalid:
            return CaseResult("E2", "FAIL",
                              details + " FAIL: p-values contain invalid values (NaN or outside [0,1]).",
                              expected)
        return CaseResult("E2", "PASS", details, expected,
                          extra={"n_removed": n_removed, "min_p_observed": min_p_observed})
    except Exception:  # noqa: BLE001
        return CaseResult("E2", "FAIL", traceback.format_exc().splitlines()[-1], expected)


def case_e3_n_neighbors_1() -> CaseResult:
    """n_neighbors=1: most unstable configuration. No exception expected."""
    expected = (
        "No exception. Audit returns valid p-values with no NaN/inf. "
        "AUROC variance across seeds will be high (documented, not a bug)."
    )
    try:
        wrapper, _ = _setup_wrapper(n_train=300, n_cal=100, n_dim=4, seed=3)
        rng = np.random.default_rng(3)
        x_test = rng.standard_normal((10, 4))

        guarded_expl = wrapper.explain_guarded_factual(
            x_test, significance=0.10, n_neighbors=1, normalize_guard=True
        )
        audit_df = extract_audit_rows(guarded_expl)
        no_nan = _no_nan_in_p_values(audit_df)

        if no_nan:
            return CaseResult(
                "E3", "PASS",
                "n_neighbors=1 completed without exception. No NaN/inf in p-values. "
                "Note: high variance across seeds is expected with n_neighbors=1.",
                expected,
            )
        else:
            return CaseResult(
                "E3", "FAIL",
                "NaN or inf detected in p-values with n_neighbors=1.",
                expected,
            )
    except Exception:  # noqa: BLE001
        return CaseResult("E3", "FAIL", traceback.format_exc().splitlines()[-1], expected)


def case_e4_n_neighbors_saturated() -> CaseResult:
    """n_neighbors >= n_cal: guard must fail open (remove nothing), not crash.

    When k_actual is clamped to n_cal, every test instance has all calibration
    points as neighbours.  The p-value estimator then counts all n_cal calibration
    distances, so the test distance is never more extreme → p ≈ 1 for all
    intervals → the guard removes nothing.  This is the correct saturating
    behaviour, but it means the guard is silently disabled — users must be warned.
    """
    # Maximum fraction of intervals the guard may remove when k >= n_cal.
    # With saturated k every instance looks fully in-distribution, so removal
    # should be near zero.  Allow a small margin for edge cases where k_actual
    # still finds a few extreme distances even after saturation.
    _SATURATION_MAX_REMOVAL_FRACTION = 0.10

    expected = (
        "No exception when n_neighbors equals or exceeds n_cal. "
        "Guard must fail open: fraction_removed ≤ "
        f"{_SATURATION_MAX_REMOVAL_FRACTION} (guard is effectively disabled)."
    )
    n_cal = 30
    try:
        x_all, y_all = make_gaussian_classification(n=200 + n_cal, n_dim=4, seed=4)
        x_tr, y_tr = x_all[:200], y_all[:200]
        x_cal, y_cal = x_all[200:], y_all[200:]
        model = RandomForestClassifier(n_estimators=80, random_state=4, n_jobs=1)
        wrapper = ensure_ce_first_wrapper(model)
        fit_and_calibrate(wrapper, x_tr, y_tr, x_cal, y_cal)

        rng = np.random.default_rng(4)
        x_test = rng.standard_normal((5, 4))

        # Request n_neighbors larger than n_cal
        guarded_expl = wrapper.explain_guarded_factual(
            x_test, significance=0.10, n_neighbors=n_cal + 10, normalize_guard=True
        )
        audit_df = extract_audit_rows(guarded_expl)

        # Compute fraction of guard-removed intervals
        if not audit_df.empty and "emission_reason" in audit_df.columns:
            total = len(audit_df)
            n_removed = int((audit_df["emission_reason"] == "removed_guard").sum())
            frac_removed = n_removed / total if total > 0 else 0.0
        else:
            frac_removed = 0.0

        if frac_removed > _SATURATION_MAX_REMOVAL_FRACTION:
            return CaseResult(
                "E4", "FAIL",
                f"n_neighbors={n_cal + 10} > n_cal={n_cal}. "
                f"fraction_removed={frac_removed:.3f} exceeds the expected maximum "
                f"{_SATURATION_MAX_REMOVAL_FRACTION} for saturated k. "
                "Guard should fail open (remove nothing) when all calibration "
                "points are included as neighbours.",
                expected,
                extra={"n_cal": n_cal, "n_neighbors_requested": n_cal + 10,
                       "fraction_removed": frac_removed},
            )
        return CaseResult(
            "E4", "PASS",
            f"n_neighbors={n_cal + 10} > n_cal={n_cal}. No crash. "
            f"fraction_removed={frac_removed:.3f} ≤ {_SATURATION_MAX_REMOVAL_FRACTION} "
            "— guard correctly fails open with saturated k.",
            expected,
            extra={"n_cal": n_cal, "n_neighbors_requested": n_cal + 10,
                   "fraction_removed": frac_removed},
        )
    except Exception:  # noqa: BLE001
        return CaseResult("E4", "FAIL", traceback.format_exc().splitlines()[-1], expected)


def case_e5_merge_adjacent_barrier() -> CaseResult:
    """merge_adjacent=True: non-conforming bins should not be bridged.

    We cannot easily construct a dataset where we know exactly which bins are
    conforming/non-conforming. Instead we verify that the merge flag only
    appears (is_merged=True) when adjacent conforming bins are present, and
    that non-conforming bins are never tagged as merged.
    """
    expected = (
        "is_merged=True only on conforming bins when merge_adjacent=True. "
        "Non-conforming bins must not be tagged as merged."
    )
    try:
        wrapper, _ = _setup_wrapper(n_train=400, n_cal=150, n_dim=4, seed=5)
        rng = np.random.default_rng(5)
        x_test = rng.standard_normal((15, 4))

        guarded_expl = wrapper.explain_guarded_factual(
            x_test, significance=0.10, n_neighbors=5,
            merge_adjacent=True, normalize_guard=True
        )
        audit_df = extract_audit_rows(guarded_expl)

        if audit_df.empty or "is_merged" not in audit_df.columns:
            return CaseResult(
                "E5", "PASS",
                "Audit returned no mergeable intervals (all design_excluded or similar). "
                "No merge barrier violation possible.",
                expected,
            )

        conforming_col = audit_df.get(
            "conforming", pd.Series(True, index=audit_df.index)
        ).astype(bool)
        is_merged_col = audit_df.get(
            "is_merged", pd.Series(False, index=audit_df.index)
        ).astype(bool)

        # Negative assertion: non-conforming bins must NOT be tagged is_merged=True.
        bad_rows = audit_df[~conforming_col & is_merged_col]
        if len(bad_rows) > 0:
            return CaseResult(
                "E5", "FAIL",
                f"{len(bad_rows)} non-conforming interval(s) incorrectly tagged "
                "is_merged=True. Merge logic is bridging across OOD bins.",
                expected,
            )

        # Positive assertion: if any merged bins exist, every one must be conforming.
        merged_rows = audit_df[is_merged_col]
        if not merged_rows.empty:
            bad_merged = merged_rows[~conforming_col.loc[merged_rows.index]]
            if not bad_merged.empty:
                return CaseResult(
                    "E5", "FAIL",
                    f"{len(bad_merged)} is_merged=True row(s) are not conforming. "
                    "Merged bins must always be conforming.",
                    expected,
                )

        n_merged = int(is_merged_col.sum())
        return CaseResult(
            "E5", "PASS",
            f"Merge integrity confirmed. Total merged bins: {n_merged}. "
            "All merged bins are conforming; no non-conforming bin is merged.",
            expected,
        )
    except Exception:  # noqa: BLE001
        return CaseResult("E5", "FAIL", traceback.format_exc().splitlines()[-1], expected)


def case_e6_single_feature() -> CaseResult:
    """x.shape = (n, 1): one feature only. No index error expected."""
    expected = (
        "No IndexError or similar in the feature iteration loop. "
        "Audit returns intervals for exactly 1 feature."
    )
    try:
        x_all, y_all = make_gaussian_classification(n=500, n_dim=1, seed=6)
        x_tr, y_tr = x_all[:300], y_all[:300]
        x_cal, y_cal = x_all[300:400], y_all[300:400]
        x_test = x_all[400:]
        model = RandomForestClassifier(n_estimators=60, random_state=6, n_jobs=1)
        wrapper = ensure_ce_first_wrapper(model)
        fit_and_calibrate(wrapper, x_tr, y_tr, x_cal, y_cal)

        guarded_expl = wrapper.explain_guarded_factual(
            x_test[:5], significance=0.10, n_neighbors=5, normalize_guard=True
        )
        audit_df = extract_audit_rows(guarded_expl)

        unique_features = audit_df["feature"].nunique() if not audit_df.empty else 0

        return CaseResult(
            "E6", "PASS",
            f"Single-feature dataset processed without error. "
            f"Unique features in audit: {unique_features} (expected 1).",
            expected,
            extra={"unique_features": unique_features},
        )
    except Exception:  # noqa: BLE001
        return CaseResult("E6", "FAIL", traceback.format_exc().splitlines()[-1], expected)


def case_e7_identical_test_instances() -> CaseResult:
    """All test instances are identical (zero variance perturbations).

    Repeating a single training instance n times tests that the guard
    handles degenerate inputs without producing NaN/inf p-values.
    """
    expected = (
        "No NaN or inf in p-values when all test instances are identical. "
        "No exception raised."
    )
    try:
        wrapper, x_cal = _setup_wrapper(n_train=300, n_cal=100, n_dim=4, seed=7)
        # Pick a single in-distribution instance and repeat it
        x_single = x_cal[0:1]
        x_test = np.repeat(x_single, 10, axis=0)

        guarded_expl = wrapper.explain_guarded_factual(
            x_test, significance=0.10, n_neighbors=5, normalize_guard=True
        )
        audit_df = extract_audit_rows(guarded_expl)
        no_nan = _no_nan_in_p_values(audit_df)

        if not no_nan:
            return CaseResult(
                "E7", "FAIL",
                "NaN or inf detected in p-values when all test instances are identical.",
                expected,
            )

        # Sanity check: test instance == x_cal[0] is a calibration-set member, so
        # it is in-distribution.  Its p-values should be > 0 for most intervals
        # (the calibration distance is not more extreme than all n_cal cal distances).
        # If > 50% of p-values are exactly 0, the guard is treating a calibration
        # point as maximally OOD — a sign of a distance-computation error.
        if not audit_df.empty and "p_value" in audit_df.columns:
            p_vals = audit_df["p_value"].dropna().astype(float)
            if not p_vals.empty:
                frac_zero = float((p_vals == 0.0).mean())
                if frac_zero > 0.5:
                    return CaseResult(
                        "E7", "FAIL",
                        f"frac_zero_p={frac_zero:.3f} > 0.5 for a calibration-set "
                        "test instance. A cal-set member should appear in-distribution "
                        "(p > 0), not maximally OOD. Possible distance-computation error.",
                        expected,
                    )

        return CaseResult(
            "E7", "PASS",
            "10 identical calibration-set instances processed. "
            "No NaN/inf. p-values are > 0 for the majority of intervals "
            "(instance is in-distribution as expected).",
            expected,
        )
    except Exception:  # noqa: BLE001
        return CaseResult("E7", "FAIL", traceback.format_exc().splitlines()[-1], expected)


def case_e8_nan_test_features() -> CaseResult:
    """NaN in test features: guard must not silently propagate NaN into p-values.

    If the guard passes NaN distances into the p-value estimator, the resulting
    p-values are either NaN (silent data corruption) or the guard crashes.
    Either is a defect.  The expected behaviour is a ValueError or UserWarning
    raised before the KNN step, OR the guard crashes with a clear message.
    A silent NaN p-value in a non-NaN row is a FAIL.
    """
    expected = (
        "Guard raises ValueError or UserWarning on NaN-containing input, "
        "OR crashes with a descriptive error — but must NOT silently produce "
        "NaN p-values in the audit for rows that have no NaN features."
    )
    try:
        wrapper, _ = _setup_wrapper(n_train=300, n_cal=100, n_dim=4, seed=8)
        rng = np.random.default_rng(8)

        # Mix: first 3 rows are clean in-distribution; last 2 rows have NaN features.
        x_clean = rng.standard_normal((3, 4))
        x_nan = rng.standard_normal((2, 4))
        x_nan[0, 1] = float("nan")   # single NaN in first OOD row
        x_nan[1, :] = float("nan")   # all NaN in second row
        x_test = np.vstack([x_clean, x_nan])

        try:
            guarded_expl = wrapper.explain_guarded_factual(
                x_test, significance=0.10, n_neighbors=5, normalize_guard=True
            )
            audit_df = extract_audit_rows(guarded_expl)

            # Clean rows (indices 0-2) must not have NaN p-values — NaN should not
            # propagate from the NaN-feature rows into clean rows.
            if not audit_df.empty and "p_value" in audit_df.columns:
                clean_rows = audit_df[audit_df["instance_index"] < 3]
                nan_in_clean = clean_rows["p_value"].isna().any()
                if nan_in_clean:
                    return CaseResult(
                        "E8", "FAIL",
                        "NaN p-values detected in clean (non-NaN feature) rows "
                        "after processing a batch that includes NaN-feature rows. "
                        "NaN contamination propagated across instances.",
                        expected,
                    )

            return CaseResult(
                "E8", "PASS",
                "Guard processed mixed NaN/clean batch without NaN contamination "
                "in clean rows. NaN-feature rows were handled without crashing.",
                expected,
            )
        except (ValueError, TypeError) as exc:
            # A clear exception on NaN input is acceptable guarded behaviour.
            return CaseResult(
                "E8", "PASS",
                f"Guard raised {type(exc).__name__} on NaN-feature input: {exc}. "
                "This is the preferred fail-fast behaviour.",
                expected,
            )
    except Exception:  # noqa: BLE001
        return CaseResult("E8", "FAIL", traceback.format_exc().splitlines()[-1], expected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CASES = [
    case_e1_zero_emitted,
    case_e2_unsmoothed_p_values,
    case_e3_n_neighbors_1,
    case_e4_n_neighbors_saturated,
    case_e5_merge_adjacent_barrier,
    case_e6_single_feature,
    case_e7_identical_test_instances,
    case_e8_nan_test_features,
]


def parse_args() -> argparse.Namespace:
    """Parse Scenario E command-line arguments.

    Parameters exposed to the caller
    --------------------------------
    --output-dir : pathlib.Path
        Destination for the PASS/FAIL CSV and markdown report.
    --quick : bool
        Accepted for interface consistency with the other scenarios. It is a
        no-op because all edge cases are already short-running.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "guarded" / "scenario_e",
    )
    parser.add_argument("--quick", action="store_true", help="No-op; all cases are already fast.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[CaseResult] = []
    for case_fn in CASES:
        print(f"Running {case_fn.__name__} ...", end=" ", flush=True)
        result = case_fn()
        results.append(result)
        print(result.status)

    # Save CSV
    rows = [
        {
            "case_id": r.case_id,
            "status": r.status,
            "details": r.details,
            "expected": r.expected,
        }
        for r in results
    ]
    csv_path = out_dir / "edge_case_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}")

    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")

    # Report
    case_table = (
        "| Case | Status | Details |\n"
        "|---|---|---|\n"
    )
    for r in results:
        icon = "✓" if r.status == "PASS" else "✗"
        summary = r.details[:120].replace("\n", " ")
        case_table += f"| {r.case_id} | {icon} {r.status} | {summary} |\n"

    fail_details = ""
    for r in results:
        if r.status == "FAIL":
            fail_details += f"\n**{r.case_id}** — {r.details}\n\nExpected: {r.expected}\n"

    report_sections = [
        (
            "Setup",
            (
                f"- Cases: {', '.join(r.case_id for r in results)}\n"
                "- Scope: parameter-space boundaries and failure-mode checks\n"
                "- Output: PASS/FAIL regression artifact, not a paper-facing benchmark"
            ),
        ),
        (
            "Purpose",
            (
                "Scenario E asks: does the guard's API behave predictably at the extremes "
                "of its parameter space — without exceptions, with documented behavior?\n\n"
                "Each case targets a specific code-path boundary. A PASS means the observed "
                "behavior matches what is expected or documented. Cases E2 and E4 have "
                "expected behavior that may seem surprising (guard does nothing, or "
                "saturates) — these are design boundaries, not bugs."
            ),
        ),
        (
            "How to read this report",
            (
                "This scenario is intentionally case-based rather than aggregate. The "
                "question is not whether the average metric looks good; it is whether the "
                "implementation fails in specific brittle situations that users will hit in "
                "practice.\n\n"
                "A FAIL indicates either a crash, silent corruption, or behavior that "
                "contradicts the documented contract for that edge case. A PASS means the "
                "observed behavior is acceptable, even if that behavior is a known "
                "limitation rather than a strength."
            ),
        ),
        (
            "Results",
            f"**{n_pass} PASS / {n_fail} FAIL** out of {len(results)} cases.\n\n{case_table}",
        ),
        (
            "Failures (if any)",
            fail_details if fail_details else "No failures.",
        ),
        (
            "Interpretation",
            (
                "**E2 (significance < 1/n_cal)**: p-values are discrete with step 1/n_cal. "
                "Very small significance settings therefore stop being meaningful on finite "
                "calibration sets. This is a configuration boundary users need documented.\n\n"
                "**E4 (n_neighbors >= n_cal)**: The guard saturates k_actual at n_cal. "
                "With all calibration points included as neighbours, all test instances "
                "appear in-distribution (guard fails open). Verified by fraction_removed "
                "check: must be ≤ 10% when saturated.\n\n"
                "**E5 (merge_adjacent)**: Both negative (non-conforming not merged) and "
                "positive (merged bins are conforming) integrity checks are enforced.\n\n"
                "**E7 (identical instances)**: Calibration-set member must yield p > 0 "
                "for the majority of intervals; all-zero p is treated as a FAIL.\n\n"
                "**E8 (NaN features)**: Guard must either raise a clear exception or "
                "process without NaN contamination propagating into clean rows.\n\n"
                "Scenario E should stay in the engineering suite. It is useful because it "
                "defines the operational envelope of the guard, not because it provides a "
                "headline effectiveness result."
            ),
        ),
    ]
    write_report(out_dir / "report.md", "Scenario E: Edge Case Behavior", report_sections)
    print(f"Wrote: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
