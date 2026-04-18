# pylint: disable=too-many-locals
"""
Venn-Abers OvR Normalization Experiment
========================================

THEORETICAL BACKGROUND
-----------------------
Venn-Abers calibration (Vovk & Petej 2012) for K-class problems decomposes
into K independent binary One-vs-Rest (OvR) calibrators. Calibrator c is
trained on the binarised target (y == c) and produces a validity interval
[low_c, high_c] and a point estimate p_c = _select_interval_summary(low_c, high_c).

VA OvR COHERENCE PROPERTY
--------------------------
For each binary OvR calibrator on class c:
  - low_c  = P(Y=c | true label is NOT c)  (conservative lower bound)
  - high_c = P(Y=c | true label IS c)       (optimistic upper bound)

For all OTHER classes k ≠ c, when the true label IS class c:
  - low_k  = P(Y=k | true label is c, which is NOT k)

Therefore for any test instance:

  high_c + sum_{k≠c} low_k
  = P(Y=c | Y=c) + sum_{k≠c} P(Y=k | Y=c)
  = sum_{all k} P(Y=k | Y=c)
  = 1   (probability axiom)

This COHERENCE PROPERTY must hold for each class c individually and is
guaranteed by the probability axiom alone. It does not require any summing
across classes in the same scenario — each term is a probability in the
scenario "the true class is c".

As a consequence:
  high_c - low_c = 1 - sum_k low_k  (same gap for all c)

This means the interval widths are equal across all classes.

WHY NORMALIZATION IS WRONG
---------------------------
The FIXME block in venn_abers.py (lines 236-242) divides low AND high by
S = sum_c va_proba_c (the sum of point estimates). After normalization:

  norm_high_c + sum_{k≠c} norm_low_k
  = (high_c + sum_{k≠c} low_k) / S
  = 1 / S   ≠ 1  (unless S = 1 by coincidence)

The normalization DESTROYS the VA coherence property. The original bounds
[low_c, high_c] already carry valid calibration meaning — they must NOT
be normalized. Only the point estimate p_c = f(low_c, high_c) is affected
by the choice of interval_summary.

WHICH INTERVAL_SUMMARY NATURALLY SUMS TO 1?
---------------------------------------------
Let D = 1 - S_low be the common interval width (S_low = sum_c low_c).
Then high_c = low_c + D for all c.

  LOWER:            sum p_c = S_low                    (< 1 typically)
  UPPER:            sum p_c = S_low + K*D = K - (K-1)*S_low  (> 1 for K>1)
  MEAN:             sum p_c = S_low + K*D/2 = K/2 - (K/2-1)*S_low
  REGULARIZED_MEAN: sum p_c = sum_c (low_c + D) / (1 + D) = (S_low + K*D)/(1+D)
                             = (K - (K-1)*S_low) / (2 - S_low)
                             -> 1  only when S_low = 1 (or K = 2)

None of the standard options naturally sum to 1 under the coherence property
unless S_low = 1. In practice REGULARIZED_MEAN has the smallest deviation from 1
because its formula was designed for binary calibration and "accidentally" produces
moderate sums for OvR multiclass.

EXPERIMENT DESIGN
-----------------
For each dataset × repeat × fold × interval_summary:
  1. Fit RandomForestClassifier + WrapCalibratedExplainer (CE-first).
  2. Get raw (un-normalised) proba, low, high via normalize=False.
  3. Get normalised proba, low, high via normalize=True (current FIXME default).
  4. Measure coherence of raw bounds:   |high_c + sum_{k≠c} low_k - 1|
  5. Measure coherence of norm bounds:  |norm_high_c + sum_{k≠c} norm_low_k - 1|
  6. Measure how far raw point estimates sum from 1 (which IS is closest?).
  7. Measure how much normalization changes the point estimates.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle

from calibrated_explanations.calibration.venn_abers import IntervalSummary
from calibrated_explanations.ce_agent_utils import ensure_ce_first_wrapper


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "Multiclass", "multi")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = [
    "iris",
    "tae",
    "image",
    "wineW",
    "wineR",
    "wine",
    "glass",
    "vehicle",
    "cmc",
    "balance",
    "wave",
    "vowel",
    "cars",
    "steel",
    "heat",
    "cool",
    "user",
    "whole",
    "yeast",
]
INTERVAL_SUMMARIES = [
    IntervalSummary.REGULARIZED_MEAN,
    IntervalSummary.MEAN,
    IntervalSummary.LOWER,
    IntervalSummary.UPPER,
]
N_REPEATS = 5
N_SPLITS = 3
CALIBRATION_SIZE = 0.33


def load_dataset(dataset_name):
    """Load a dataset and remap labels to contiguous integer ids."""
    data = pd.read_csv(os.path.join(DATA_DIR, f"{dataset_name}.csv"), sep=";")
    x_values = data.iloc[:, :-1].values
    raw_labels = data.iloc[:, -1].values
    _, encoded_labels = np.unique(raw_labels, return_inverse=True)
    x_values, encoded_labels = shuffle(x_values, encoded_labels, random_state=0)
    return x_values, encoded_labels


def compute_fold_metrics(wrapper, x_test, interval_summary):
    """Compute coherence and normalisation metrics for one fold and interval_summary.

    Primary metric: does high_c + sum_{k≠c} low_k = 1 (VA coherence)?
    Does normalization destroy this property?

    Parameters
    ----------
    wrapper : WrapCalibratedExplainer
        A fitted and calibrated CE-first wrapper.
    x_test : ndarray of shape (n_test, n_features)
        Raw test feature matrix.
    interval_summary : IntervalSummary
        Which point-estimate strategy to use for the VA interval.

    Returns
    -------
    dict with scalar metrics (see module docstring for definitions).
    """
    # Thread normalize and interval_summary through the full public API:
    # wrapper.predict_proba -> CalibratedExplainer.predict_proba(**kwargs)
    #   -> interval_learner.predict_proba(x, output_interval=True, **kwargs)
    #     -> VennAbers.predict_proba(normalize=False/True, interval_summary=IS)
    raw_payload = wrapper.predict_proba(
        x_test, uq_interval=True, normalize=False, interval_summary=interval_summary
    )
    norm_payload = wrapper.predict_proba(
        x_test, uq_interval=True, normalize=True, interval_summary=interval_summary
    )
    raw_proba = np.asarray(raw_payload[0])
    raw_low = np.asarray(raw_payload[1][0])
    raw_high = np.asarray(raw_payload[1][1])
    norm_proba = np.asarray(norm_payload[0])
    norm_low = np.asarray(norm_payload[1][0])
    norm_high = np.asarray(norm_payload[1][1])

    # --- Row sums of point estimates (which IS naturally sums to 1?) ---
    raw_row_sums = raw_proba.sum(axis=1)       # (n_test,)
    norm_row_sums = norm_proba.sum(axis=1)     # should be ≈ 1.0

    # --- VA coherence property: high_c + sum_{k≠c} low_k = 1 ---
    # For each (row, class c): high_c + S_low - low_c = 1
    # where S_low = sum_k low_k  (all lower bounds, including low_c itself, minus low_c again)
    # Actually: sum_{k≠c} low_k = S_low - low_c
    # So coherence_c = high_c + (S_low - low_c)

    # Raw coherence
    raw_s_low = raw_low.sum(axis=1, keepdims=True)   # (n_test, 1) = sum_k low_k
    # (n_test, n_classes) — each cell = high_c + sum_{k≠c} low_k
    raw_coherence = raw_high + raw_s_low - raw_low
    raw_coh_dev = np.abs(raw_coherence - 1.0)        # deviation from 1

    # Normalised coherence (FIXME normalization divides all by S = sum(va_proba))
    norm_s_low = norm_low.sum(axis=1, keepdims=True)
    norm_coherence = norm_high + norm_s_low - norm_low
    norm_coh_dev = np.abs(norm_coherence - 1.0)

    # Analytically: norm_coherence_c = coherence_c / S = 1/S (same for all classes c).
    # Verify this prediction: std of norm_coherence across classes should be ≈ 0.
    norm_coh_class_std = norm_coherence.std(axis=1).mean()  # should be ≈ 0

    # --- Interval width: high_c - low_c should be equal across classes (from coherence) ---
    raw_interval_width = raw_high - raw_low         # (n_test, n_classes)
    raw_width_class_std = raw_interval_width.std(axis=1).mean()  # should be ≈ 0 under coherence

    # --- Delta from normalization on point estimates ---
    delta_proba = np.abs(norm_proba - raw_proba)

    return {
        # Row sums of raw point estimates
        "row_sum_raw_mean": float(raw_row_sums.mean()),
        "row_sum_raw_std": float(raw_row_sums.std()),
        "row_sum_raw_min": float(raw_row_sums.min()),
        "row_sum_raw_max": float(raw_row_sums.max()),
        # Row sums of normalised point estimates (should be ≈ 1.0)
        "row_sum_norm_mean": float(norm_row_sums.mean()),
        # VA coherence: raw bounds (should be ≈ 0 if VA property holds)
        "coherence_raw_mae": float(raw_coh_dev.mean()),
        "coherence_raw_max": float(raw_coh_dev.max()),
        "coherence_raw_mean": float(raw_coherence.mean()),   # should be ≈ 1.0
        # VA coherence: normalised bounds (destroyed by normalization)
        "coherence_norm_mae": float(norm_coh_dev.mean()),
        "coherence_norm_max": float(norm_coh_dev.max()),
        "coherence_norm_mean": float(norm_coherence.mean()), # = mean(1/S) ≠ 1
        # Validation: norm_coherence is the same for all classes c (std ≈ 0)
        "norm_coh_class_std": norm_coh_class_std,
        # Interval width uniformity across classes (should be ≈ 0 under coherence)
        "interval_width_class_std": raw_width_class_std,
        # How much normalization changes the point estimates
        "delta_proba_mae": float(delta_proba.mean()),
        "delta_proba_max": float(delta_proba.max()),
    }


def run_dataset_experiment(dataset_name):
    """Run the normalisation experiment for one dataset.

    Returns
    -------
    list of dicts, one per (repeat, fold, interval_summary) combination.
    """
    print(f"[{dataset_name}] starting ...")
    x_values, y_values = load_dataset(dataset_name)
    n_classes = len(np.unique(y_values))
    min_class_count = int(np.bincount(y_values).min())

    if min_class_count < N_SPLITS:
        print(
            f"[{dataset_name}] skipped: "
            f"min class count {min_class_count} < n_splits={N_SPLITS}"
        )
        return []

    rows = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat)
        for fold_idx, (train_idx, test_idx) in enumerate(
            splitter.split(x_values, y_values)
        ):
            split_seed = repeat * N_SPLITS + fold_idx
            x_train, x_test = x_values[train_idx], x_values[test_idx]
            y_train = y_values[train_idx]

            x_prop, x_cal, y_prop, y_cal = train_test_split(
                x_train,
                y_train,
                test_size=CALIBRATION_SIZE,
                random_state=split_seed,
                stratify=y_train,
            )

            if len(np.unique(y_prop)) != n_classes or len(np.unique(y_cal)) != n_classes:
                print(
                    f"[{dataset_name}] repeat={repeat} fold={fold_idx}: "
                    "class-coverage failure, skipping fold"
                )
                continue

            learner = RandomForestClassifier(random_state=split_seed)
            wrapper = ensure_ce_first_wrapper(learner)
            wrapper.fit(x_prop, y_prop)
            wrapper.calibrate(x_cal, y_cal, mode="classification")

            for is_enum in INTERVAL_SUMMARIES:
                try:
                    metrics = compute_fold_metrics(wrapper, x_test, is_enum)
                except Exception as exc:  # pragma: no cover - defensive fold guard
                    print(
                        f"[{dataset_name}] repeat={repeat} fold={fold_idx} "
                        f"IS={is_enum.name}: {exc}"
                    )
                    continue

                rows.append(
                    {
                        "dataset": dataset_name,
                        "n_classes": n_classes,
                        "repeat": repeat,
                        "fold": fold_idx,
                        "interval_summary": is_enum.name,
                        **metrics,
                    }
                )

    print(f"[{dataset_name}] done, {len(rows)} rows collected.")
    return rows


def print_summary(results_df):
    """Print a human-readable summary table to stdout."""
    summary_cols = [
        "row_sum_raw_mean",
        "row_sum_raw_std",
        "row_sum_norm_mean",
        "coherence_raw_mae",
        "coherence_raw_mean",
        "coherence_norm_mae",
        "coherence_norm_mean",
        "norm_coh_class_std",
        "interval_width_class_std",
        "delta_proba_mae",
        "delta_proba_max",
    ]
    summary = results_df.groupby("interval_summary")[summary_cols].mean()

    pd.set_option("display.float_format", "{:.6f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n=== VA Coherence & Normalisation Summary (mean across datasets/folds/repeats) ===")
    print(summary.T.to_string())

    print("\n--- KEY FINDINGS ---")

    # 1. Coherence of raw bounds
    print("\n1. VA coherence (raw bounds): high_c + sum_{k!=c} low_k ~= 1")
    for is_name in summary.index:
        coh_mae = summary.loc[is_name, "coherence_raw_mae"]
        coh_mean = summary.loc[is_name, "coherence_raw_mean"]
        print(f"   {is_name}: coherence_mean={coh_mean:.6f}, coherence_MAE={coh_mae:.6f}")

    # 2. Normalization destroys coherence
    print("\n2. VA coherence after normalization (mean shift from raw 0.958 -> ?):")
    for is_name in summary.index:
        raw_mae = summary.loc[is_name, "coherence_raw_mae"]
        raw_mean = summary.loc[is_name, "coherence_raw_mean"]
        norm_mae = summary.loc[is_name, "coherence_norm_mae"]
        norm_mean = summary.loc[is_name, "coherence_norm_mean"]
        mean_shift = norm_mean - raw_mean
        mae_change = norm_mae - raw_mae
        direction = "worsened" if norm_mae > raw_mae else "improved"
        print(
            f"   {is_name}: coherence_mean {raw_mean:.4f} -> {norm_mean:.4f} "
            f"(shift {mean_shift:+.4f}), MAE {raw_mae:.4f} -> {norm_mae:.4f} "
            f"({mae_change:+.4f}, {direction})"
        )

    # 3. Which IS naturally sums to 1?
    print("\n3. Raw row-sum deviation from 1 (smaller = closer to valid simplex):")
    for is_name in summary.index:
        row_sum = summary.loc[is_name, "row_sum_raw_mean"]
        dev = abs(row_sum - 1.0)
        print(f"   {is_name}: row_sum_raw_mean={row_sum:.6f}, |deviation|={dev:.6f}")

    # 4. Interval width uniformity (coherence prediction: widths should be equal)
    print("\n4. Interval width uniformity across classes (coherence predicts std ~= 0):")
    for is_name in summary.index:
        std = summary.loc[is_name, "interval_width_class_std"]
        print(f"   {is_name}: interval_width_class_std={std:.6f}")

    # 5. Theoretical prediction check: norm_coherence is same for all classes
    print("\n5. norm_coherence class-std (should be ~= 0: same 1/S for all classes):")
    for is_name in summary.index:
        std = summary.loc[is_name, "norm_coh_class_std"]
        print(f"   {is_name}: norm_coh_class_std={std:.6f}")


def main():
    """Run the full experiment across all datasets and save results."""
    all_rows = []
    for dataset_name in DATASETS:
        rows = run_dataset_experiment(dataset_name)
        all_rows.extend(rows)

    if not all_rows:
        print("No results collected.")
        return

    results_df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "results_va_normalization.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(results_df)} rows -> {csv_path}")

    print_summary(results_df)


if __name__ == "__main__":
    main()
