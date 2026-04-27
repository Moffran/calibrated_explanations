"""Shared utilities for the guarded explanation evaluation suite.

All scenario scripts import from here to avoid boilerplate duplication.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

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


def format_duration(seconds: float | None) -> str:
    """Format elapsed or remaining seconds for terminal progress messages."""
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


@dataclass
class ProgressTracker:
    """Small terminal progress helper for long guarded evaluation scripts."""

    label: str
    total: int
    min_interval_seconds: float = 10.0
    completed: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    _last_print_at: float = field(init=False, default=0.0)

    def start(self, detail: str = "") -> None:
        """Print the first progress line."""
        self._emit("start", detail, force=True)

    def note(self, detail: str) -> None:
        """Print an unconditional status note without changing progress."""
        self._emit("status", detail, force=True)

    def advance(self, detail: str = "", step: int = 1, *, force: bool = False) -> None:
        """Advance completed work and print when enough time has passed."""
        self.completed = min(max(self.completed + step, 0), max(self.total, 0))
        self._emit("progress", detail, force=force)

    def finish(self, detail: str = "") -> None:
        """Mark all work complete and print the final progress line."""
        self.completed = max(self.completed, self.total)
        self._emit("done", detail, force=True)

    def snapshot(self, detail: str = "") -> dict[str, Any]:
        """Return a JSON-serializable progress snapshot."""
        elapsed = time.perf_counter() - self.started_at
        percent = 100.0 if self.total <= 0 else 100.0 * self.completed / self.total
        rate = self.completed / elapsed if elapsed > 0 and self.completed > 0 else 0.0
        remaining = max(self.total - self.completed, 0)
        eta_seconds = remaining / rate if rate > 0 else None
        return {
            "label": self.label,
            "completed": int(self.completed),
            "total": int(self.total),
            "remaining": int(remaining),
            "percent": round(percent, 2),
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
            "detail": detail,
        }

    def _emit(self, kind: str, detail: str, *, force: bool) -> None:
        now = time.perf_counter()
        should_print = (
            force
            or self.completed >= self.total
            or now - self._last_print_at >= self.min_interval_seconds
        )
        if not should_print:
            return
        self._last_print_at = now
        snapshot = self.snapshot(detail)
        percent = snapshot["percent"]
        elapsed = format_duration(float(snapshot["elapsed_seconds"]))
        eta = format_duration(snapshot["eta_seconds"])
        parts = [
            f"[{self.label}] {kind}:",
            f"{snapshot['completed']}/{snapshot['total']}",
            f"({percent:.1f}%)",
            f"elapsed={elapsed}",
            f"eta={eta}",
        ]
        if detail:
            parts.append(f"- {detail}")
        print(" ".join(parts), flush=True)


def reset_intermediate_outputs(paths: Sequence[Path]) -> None:
    """Remove stale intermediate files for a new guarded evaluation run."""
    for path in paths:
        if path.exists():
            path.unlink()


def append_intermediate_rows(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> int:
    """Append row dictionaries to a CSV file and return the number written."""
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if columns is not None:
        frame = frame.reindex(columns=list(columns))
    write_header = not path.exists()
    frame.to_csv(path, mode="a", header=write_header, index=False)
    return len(frame)


def write_intermediate_frame(frame: pd.DataFrame, path: Path) -> None:
    """Write a complete intermediate frame atomically where possible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def write_progress_snapshot(
    path: Path,
    tracker: ProgressTracker,
    *,
    detail: str = "",
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write a small JSON file describing current guarded evaluation progress."""
    payload = tracker.snapshot(detail)
    if extra:
        payload.update(dict(extra))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def dataframe_to_markdown(obj: Any, **kwargs: Any) -> str:
    """Render a pandas object as markdown, with a CSV fallback."""
    try:
        return obj.to_markdown(**kwargs)
    except ImportError:
        index = bool(kwargs.get("index", True))
        if isinstance(obj, pd.Series):
            csv_text = obj.to_frame(name=obj.name or "value").to_csv()
        else:
            csv_text = obj.to_csv(index=index)
        return f"```csv\n{csv_text.strip()}\n```"


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

    auroc = (
        float("nan")
        if len(np.unique(labels)) < 2
        else float(roc_auc_score(labels, scores))
    )

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
