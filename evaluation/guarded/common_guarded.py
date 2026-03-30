"""Shared utilities for the guarded explanation evaluation suite.

All scenario scripts import from here to avoid boilerplate duplication.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Guard configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardConfig:
    """Immutable guard hyperparameter bundle passed to explain_guarded_* calls."""
    significance: float = 0.1
    n_neighbors: int = 5
    merge_adjacent: bool = False
    normalize_guard: bool = True


# Minimum absolute difference in mean fraction_removed between OOD and ID instances
# required for the guard to be considered "responsive" in Scenario C.
OOD_RESPONSIVENESS_MIN_DELTA: float = 0.05

# Fields that must appear in every interval record of get_guarded_audit().
# Defined by ADR-032 Addendum.
REQUIRED_AUDIT_FIELDS: frozenset[str] = frozenset({
    "feature", "feature_name", "lower", "upper",
    "representative",
    "p_value", "conforming", "emitted", "emission_reason",
    "condition", "predict", "low", "high",
})


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_splits(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int,
    n_cal: int,
    n_test: int,
    seed: int,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic train / calibration / test split.

    Returns (x_train, y_train, x_cal, y_cal, x_test, y_test).
    Caps each split at the available data size; raises if total requested
    exceeds the dataset length.
    """
    total = n_train + n_cal + n_test
    if total > len(x):
        raise ValueError(
            f"Requested {total} samples but dataset has only {len(x)}. "
            "Reduce n_train/n_cal/n_test."
        )
    strat = y if stratify else None
    x_rest, x_test, y_rest, y_test = train_test_split(
        x, y, test_size=n_test, random_state=seed, stratify=strat
    )
    strat_rest = y_rest if stratify else None
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_rest, y_rest, test_size=n_cal, random_state=seed + 1, stratify=strat_rest
    )
    return x_train[:n_train], y_train[:n_train], x_cal[:n_cal], y_cal[:n_cal], x_test, y_test


def make_gaussian_classification(
    n: int, n_dim: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic Gaussian data with a linear binary classification boundary.

    Labels are balanced by thresholding the linear score at its median,
    then independently adding noise to ensure the problem is non-trivial.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_dim))
    direction = np.ones(n_dim) / np.sqrt(n_dim)
    logits = x @ direction + rng.standard_normal(n) * 0.5
    y = (logits > np.median(logits)).astype(int)
    return x, y


def make_ood_shift(
    x: np.ndarray, shift_vector: np.ndarray, seed: int
) -> np.ndarray:
    """Return a copy of x shifted by shift_vector with small per-instance noise.

    The noise breaks ties so that all shifted instances are not identical
    after shift, which would trivially collapse KNN distances.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.shape) * 0.02
    return x + shift_vector + noise


# ---------------------------------------------------------------------------
# Audit extraction
# ---------------------------------------------------------------------------

def extract_audit_rows(guarded_explanations: Any) -> pd.DataFrame:
    """Flatten get_guarded_audit() into a DataFrame with one row per interval.

    Columns: all fields from each interval record, plus ``instance_index``
    from the parent instance dict.
    """
    audit = guarded_explanations.get_guarded_audit()
    rows: List[dict] = []
    for inst in audit["instances"]:
        idx = inst["instance_index"]
        for interval in inst["intervals"]:
            row = dict(interval)
            row["instance_index"] = idx
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def extract_audit_summary_rows(guarded_explanations: Any) -> pd.DataFrame:
    """Flatten get_guarded_audit() summary into a DataFrame with one row per instance."""
    audit = guarded_explanations.get_guarded_audit()
    rows: List[dict] = []
    for inst in audit["instances"]:
        row = dict(inst["summary"])
        row["instance_index"] = inst["instance_index"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Invariant checking
# ---------------------------------------------------------------------------

def check_interval_invariant(audit_df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Return rows from audit_df where predict < low or predict > high.

    Only rows where all three values are non-null and finite are considered.
    An empty return means no violations — the invariant holds everywhere.
    """
    if audit_df.empty:
        return pd.DataFrame()
    required = {"predict", "low", "high"}
    if not required.issubset(audit_df.columns):
        return pd.DataFrame()
    df = audit_df.copy()
    for col in ("predict", "low", "high"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mask_valid = df["predict"].notna() & df["low"].notna() & df["high"].notna()
    df = df[mask_valid]
    if df.empty:
        return pd.DataFrame()
    violating = df[(df["predict"] < df["low"] - eps) | (df["predict"] > df["high"] + eps)]
    return violating.reset_index(drop=True)


def check_audit_field_completeness(
    guarded_explanations: Any,
) -> Tuple[bool, List[dict]]:
    """Check that every interval record has all REQUIRED_AUDIT_FIELDS.

    Returns (all_complete, missing_list) where missing_list contains dicts
    describing any missing fields: {instance_index, interval_index, missing_fields}.
    """
    audit = guarded_explanations.get_guarded_audit()
    missing_list: List[dict] = []
    for inst in audit["instances"]:
        idx = inst["instance_index"]
        for interval_idx, interval in enumerate(inst["intervals"]):
            present = set(interval.keys())
            missing = REQUIRED_AUDIT_FIELDS - present
            if missing:
                missing_list.append({
                    "instance_index": idx,
                    "interval_index": interval_idx,
                    "missing_fields": sorted(missing),
                })
    return (len(missing_list) == 0), missing_list


# ---------------------------------------------------------------------------
# OOD detection metrics
# ---------------------------------------------------------------------------

def compute_ood_detection_metrics(
    p_values_id: Sequence[float],
    p_values_ood: Sequence[float],
) -> dict:
    """Compute AUROC from per-instance combined p-value lists.

    ``p_values_id`` and ``p_values_ood`` are instance-level combined p-values,
    for example from ``fisher_p_value_per_instance``.

    Returns a dict with: auroc, n_id, n_ood, median_combined_p_id,
    median_combined_p_ood.
    """
    from sklearn.metrics import roc_auc_score  # lazy import

    arr_id = np.asarray(p_values_id, dtype=float)
    arr_ood = np.asarray(p_values_ood, dtype=float)

    # Anomaly score = 1 - p_value; OOD label = 1
    scores = np.concatenate([1.0 - arr_id, 1.0 - arr_ood])
    labels = np.concatenate([np.zeros(len(arr_id)), np.ones(len(arr_ood))])

    if len(np.unique(labels)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(labels, scores))

    return {
        "auroc": auroc,
        "n_id": len(arr_id),
        "n_ood": len(arr_ood),
        "median_combined_p_id": float(np.median(arr_id)) if len(arr_id) > 0 else float("nan"),
        "median_combined_p_ood": float(np.median(arr_ood)) if len(arr_ood) > 0 else float("nan"),
    }


def mean_p_value_per_instance(audit_df: pd.DataFrame) -> pd.Series:
    """Aggregate p_values to one mean p-value per instance_index.

    .. deprecated::
        Arithmetic mean of p-values is not a valid combining method: a single
        interval with p≈0 (strong OOD signal) is diluted by other p≈0.5
        intervals, masking the detection signal.  Prefer
        ``fisher_p_value_per_instance`` for AUROC computation.
    """
    if audit_df.empty or "p_value" not in audit_df.columns:
        return pd.Series(dtype=float)
    return (
        audit_df.dropna(subset=["p_value"])
        .groupby("instance_index")["p_value"]
        .mean()
    )


def fisher_p_value_per_instance(audit_df: pd.DataFrame) -> pd.Series:
    """Aggregate p_values per instance using Fisher's combined test.

    Fisher's method: statistic = -2 * sum(ln(p_i)) ~ chi2(2k) under H0
    (all p_i are Uniform[0,1], i.e. every interval is in-distribution).

    Returns the combined p-value per ``instance_index``.  A low value
    means the instance is likely OOD; a high value means in-distribution.
    This is the correct quantity to use as the anomaly score for AUROC
    computation — arithmetic mean is not.

    Edge cases:
    - Any p_i == 0  →  combined p = 0  (definite OOD dominates).
    - All p_i missing  →  NaN.
    """
    from scipy.stats import chi2  # lazy import

    if audit_df.empty or "p_value" not in audit_df.columns:
        return pd.Series(dtype=float)

    def _combine(p_vals: pd.Series) -> float:
        vals = p_vals.dropna().astype(float)
        if len(vals) == 0:
            return float("nan")
        if (vals == 0.0).any():
            return 0.0
        stat = -2.0 * float(np.sum(np.log(vals.clip(lower=1e-300))))
        return float(1.0 - chi2.cdf(stat, df=2 * len(vals)))

    return (
        audit_df.dropna(subset=["p_value"])
        .groupby("instance_index")["p_value"]
        .apply(_combine)
    )


def check_ood_responsiveness(
    summary_df: pd.DataFrame,
    min_delta: float = OOD_RESPONSIVENESS_MIN_DELTA,
) -> Tuple[bool, float, float]:
    """Check whether the guard removes significantly more intervals for OOD vs ID.

    Requires ``summary_df`` to have columns: ``is_ood``,
    ``intervals_removed_guard``, ``intervals_tested``.

    Returns ``(passes, mean_frac_ood, mean_frac_id)`` where ``passes`` is
    True iff ``mean_frac_ood - mean_frac_id >= min_delta``.
    """
    if summary_df.empty or "is_ood" not in summary_df.columns:
        return False, float("nan"), float("nan")
    if not {"intervals_removed_guard", "intervals_tested"}.issubset(summary_df.columns):
        return False, float("nan"), float("nan")
    frame = summary_df.copy()
    is_ood = frame["is_ood"].astype(bool)
    frame["fraction_removed"] = (
        frame["intervals_removed_guard"] / frame["intervals_tested"].clip(lower=1)
    )
    ood_frac = float(frame[is_ood]["fraction_removed"].mean())
    id_frac = float(frame[~is_ood]["fraction_removed"].mean())
    return (ood_frac - id_frac) >= min_delta, ood_frac, id_frac


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(
    path: Path,
    title: str,
    sections: List[Tuple[str, str]],
) -> None:
    """Write a markdown report to path.

    sections is a list of (heading, body) tuples. The body can be a plain
    string or a markdown-formatted string including tables.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body.strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
