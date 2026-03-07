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
    use_bonferroni: bool = False
    normalize_guard: bool = True


# Fields that must appear in every interval record of get_guarded_audit().
# Defined by ADR-032 Addendum.
REQUIRED_AUDIT_FIELDS: frozenset[str] = frozenset({
    "feature", "feature_name", "lower", "upper", "representative",
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
    significance: float,
) -> dict:
    """Compute AUROC and FPR@significance from per-instance p-value lists.

    p_values_id: mean guard p-value per in-distribution instance.
    p_values_ood: mean guard p-value per OOD instance.
    significance: the significance level used as the FPR threshold.

    Returns a dict with: auroc, fpr_at_significance, n_id, n_ood,
    median_p_id, median_p_ood.
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

    # FPR = fraction of ID intervals flagged as OOD (p_value < significance)
    fpr = float(np.mean(arr_id < significance)) if len(arr_id) > 0 else float("nan")

    return {
        "auroc": auroc,
        "fpr_at_significance": fpr,
        "n_id": len(arr_id),
        "n_ood": len(arr_ood),
        "median_p_id": float(np.median(arr_id)) if len(arr_id) > 0 else float("nan"),
        "median_p_ood": float(np.median(arr_ood)) if len(arr_ood) > 0 else float("nan"),
    }


def mean_p_value_per_instance(audit_df: pd.DataFrame) -> pd.Series:
    """Aggregate p_values to one mean p-value per instance_index."""
    if audit_df.empty or "p_value" not in audit_df.columns:
        return pd.Series(dtype=float)
    return (
        audit_df.dropna(subset=["p_value"])
        .groupby("instance_index")["p_value"]
        .mean()
    )


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
