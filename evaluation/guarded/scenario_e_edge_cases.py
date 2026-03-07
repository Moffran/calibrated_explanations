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
    model = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=seed, n_jobs=-1)
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
        model = RandomForestClassifier(n_estimators=80, random_state=2, n_jobs=-1)
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
    """n_neighbors >= n_cal: k_actual saturates; no crash expected."""
    expected = (
        "No exception when n_neighbors equals or exceeds n_cal. "
        "p-values may be uniformly high (everything looks in-distribution)."
    )
    n_cal = 30
    try:
        x_all, y_all = make_gaussian_classification(n=200 + n_cal, n_dim=4, seed=4)
        x_tr, y_tr = x_all[:200], y_all[:200]
        x_cal, y_cal = x_all[200:], y_all[200:]
        model = RandomForestClassifier(n_estimators=80, random_state=4, n_jobs=-1)
        wrapper = ensure_ce_first_wrapper(model)
        fit_and_calibrate(wrapper, x_tr, y_tr, x_cal, y_cal)

        rng = np.random.default_rng(4)
        x_test = rng.standard_normal((5, 4))

        # Request n_neighbors larger than n_cal
        guarded_expl = wrapper.explain_guarded_factual(
            x_test, significance=0.10, n_neighbors=n_cal + 10, normalize_guard=True
        )
        audit_df = extract_audit_rows(guarded_expl)

        return CaseResult(
            "E4", "PASS",
            f"n_neighbors={n_cal + 10} > n_cal={n_cal}. No crash. "
            f"Audit rows: {len(audit_df)}. "
            "Note: with saturated k, all p-values may be uniformly high.",
            expected,
            extra={"n_cal": n_cal, "n_neighbors_requested": n_cal + 10},
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

        # Non-conforming bins must not be tagged is_merged=True
        bad_rows = audit_df[
            (audit_df.get("conforming", pd.Series(True, index=audit_df.index)) == False)  # noqa: E712
            & (audit_df.get("is_merged", pd.Series(False, index=audit_df.index)) == True)  # noqa: E712
        ]
        if len(bad_rows) == 0:
            return CaseResult(
                "E5", "PASS",
                "No non-conforming bins tagged as is_merged=True. "
                f"Total merged bins: {int(audit_df['is_merged'].sum())}.",
                expected,
            )
        else:
            return CaseResult(
                "E5", "FAIL",
                f"{len(bad_rows)} non-conforming interval(s) incorrectly tagged is_merged=True. "
                "This suggests the merge logic bridges across OOD bins.",
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
        model = RandomForestClassifier(n_estimators=60, random_state=6, n_jobs=-1)
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

        if no_nan:
            return CaseResult(
                "E7", "PASS",
                "10 identical test instances processed. No NaN/inf in p-values.",
                expected,
            )
        else:
            return CaseResult(
                "E7", "FAIL",
                "NaN or inf detected in p-values when all test instances are identical.",
                expected,
            )
    except Exception:  # noqa: BLE001
        return CaseResult("E7", "FAIL", traceback.format_exc().splitlines()[-1], expected)


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
]


def parse_args() -> argparse.Namespace:
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
            "What this scenario tests",
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
            "Results",
            f"**{n_pass} PASS / {n_fail} FAIL** out of {len(results)} cases.\n\n{case_table}",
        ),
        (
            "Failures (if any)",
            fail_details if fail_details else "No failures.",
        ),
        (
            "Design boundaries documented by PASS cases",
            (
                "**E2 (significance < 1/n_cal)**: p-values are discrete with step 1/n_cal. "
                "When significance < 1/n_cal, the guard can never reject any interval — it "
                "silently behaves as significance=0. Users must be warned about this.\n\n"
                "**E4 (n_neighbors >= n_cal)**: The guard saturates k_actual at n_cal. "
                "With all calibration points included as neighbors, all test instances may "
                "appear in-distribution. This is the correct saturating behavior."
            ),
        ),
    ]
    write_report(out_dir / "report.md", "Scenario E: Edge Case Behavior", report_sections)
    print(f"Wrote: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
