# pylint: disable=too-many-lines
"""
Experiment: Real-multiclass CE-first probability quality
========================================================

Purpose:
    Evaluate multiclass probability quality using only WrapCalibratedExplainer
    public APIs. Every prediction in the experiment is generated through
    wrapper.predict_proba(...).

Three CE-first arms are compared:
    - Uncal    : wrapper.predict_proba(x_test, calibrated=False)
    - CE       : standard calibrated wrapper
    - CE_multi : separately calibrated wrapper with multi_labels_enabled=True

Design:
    - 10 outer repeats x 5-fold stratified cross-validation
    - one learner is fitted per fold on the proper-training split
    - two wrappers share that same fitted learner instance and are calibrated
      independently on the same calibration split

Notes:
    - If CE_multi does not produce prediction outputs distinct from CE, that is
      recorded explicitly in the outputs and flagged in the console summary.
    - No direct learner.predict_proba calls are used in the experiment body.
"""

from __future__ import annotations

import os
import warnings
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.ce_agent_utils import ensure_ce_first_wrapper
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "Multiclass", "multi")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

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
METHODS = ("Uncal", "CE", "CE_multi")
OUTER_LOOPS = 10
N_SPLITS = 5
N_BINS = 10
CALIBRATION_SIZE = 0.33
COLORS = {"Uncal": "steelblue", "CE": "darkorange", "CE_multi": "forestgreen"}
PAIRWISE_METRICS = [
    "log_loss",
    "multiclass_brier",
    "per_class_ece",
    "per_class_brier",
    "top1_ece",
    "top1_brier",
    "accuracy",
]
LOWER_IS_BETTER = {
    "log_loss": True,
    "multiclass_brier": True,
    "per_class_ece": True,
    "per_class_brier": True,
    "top1_ece": True,
    "top1_brier": True,
    "accuracy": False,
}


def _get_pyplot():
    """Lazily import matplotlib pyplot for plotting-only code paths."""
    try:
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting outputs in experiment_real_multiclass.py. "
            "Install visualization extras to enable plot generation."
        ) from error
    return plt


def compute_bin_stats(y_true_binary, y_prob, n_bins=N_BINS):
    """Compute calibration-bin accuracy, confidence, and counts using one binning path."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    clipped_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_true_binary = np.asarray(y_true_binary, dtype=float)
    bin_ids = np.searchsorted(bins, clipped_prob, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    counts = np.bincount(bin_ids, minlength=n_bins)
    non_empty = counts > 0
    fraction_of_positives = np.array(
        [y_true_binary[bin_ids == index].mean() for index in np.where(non_empty)[0]],
        dtype=float,
    )
    mean_predicted_value = np.array(
        [clipped_prob[bin_ids == index].mean() for index in np.where(non_empty)[0]],
        dtype=float,
    )
    return counts[non_empty], fraction_of_positives, mean_predicted_value


def expected_calibration_error(y_true_binary, y_prob, n_bins=N_BINS):
    """Compute ECE with weights derived from the same bins used for calibration stats."""
    counts, fraction_of_positives, mean_predicted_value = compute_bin_stats(
        y_true_binary, y_prob, n_bins=n_bins
    )
    if counts.size == 0:
        return float("nan"), fraction_of_positives, mean_predicted_value
    weights = counts / counts.sum()
    ece = np.sum(weights * np.abs(fraction_of_positives - mean_predicted_value))
    return float(ece), fraction_of_positives, mean_predicted_value


def per_class_brier(y_true, proba, class_labels):
    """Mean one-vs-rest Brier score across all classes."""
    scores = [
        brier_score_loss((y_true == class_label).astype(int), proba[:, class_index])
        for class_index, class_label in enumerate(class_labels)
    ]
    return float(np.mean(scores))


def multiclass_brier_score(y_true, proba, class_labels):
    """Compute full-vector multiclass Brier score for normalized probability outputs."""
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    class_indices = np.array([label_to_index[label] for label in y_true], dtype=int)
    target = np.zeros_like(proba, dtype=float)
    target[np.arange(len(y_true)), class_indices] = 1.0
    return float(np.mean(np.sum((proba - target) ** 2, axis=1)))


def per_class_ece(y_true, proba, class_labels, n_bins=N_BINS):
    """Mean one-vs-rest ECE across all classes."""
    ece_scores = []
    for class_index, class_label in enumerate(class_labels):
        binary_target = (y_true == class_label).astype(int)
        ece_value, _, _ = expected_calibration_error(binary_target, proba[:, class_index], n_bins)
        ece_scores.append(ece_value)
    return float(np.mean(ece_scores))


def per_class_metric_rows(dataset_name, method_name, y_true, proba, class_labels, n_samples):
    """Return per-class one-vs-rest diagnostics for deeper error analysis."""
    rows = []
    for class_index, class_label in enumerate(class_labels):
        binary_target = (y_true == class_label).astype(int)
        ece_value, _, _ = expected_calibration_error(binary_target, proba[:, class_index], N_BINS)
        brier_value = brier_score_loss(binary_target, proba[:, class_index])
        prevalence = float(np.mean(binary_target))
        mean_probability = float(np.mean(proba[:, class_index]))
        rows.append(
            {
                "dataset": dataset_name,
                "method": method_name,
                "class_label": int(class_label),
                "n_samples": n_samples,
                "class_prevalence": round(prevalence, 6),
                "mean_predicted_probability": round(mean_probability, 6),
                "class_ece": round(float(ece_value), 4),
                "class_brier": round(float(brier_value), 4),
            }
        )
    return rows


def summarize_row_sum_quality(row_sums):
    """Return a qualitative row-sum severity label and remediation suggestion."""
    row_sums = np.asarray(row_sums, dtype=float)
    mae_from_one = float(np.mean(np.abs(row_sums - 1.0)))
    max_abs_error = float(np.max(np.abs(row_sums - 1.0)))

    if max_abs_error <= 1e-12:
        return {
            "row_sum_quality": "exactly_normalized",
            "row_sum_issue_severity": "none",
            "row_sum_suggestion": "No action needed. Probabilities already sum to 1.0 within numerical precision.",
        }
    if mae_from_one <= 1e-3 and max_abs_error <= 1e-2:
        return {
            "row_sum_quality": "near_normalized",
            "row_sum_issue_severity": "low",
            "row_sum_suggestion": (
                "Usually acceptable. If strict probability-simplex compliance is required, "
                "apply row-wise normalization and verify that calibration metrics do not degrade."
            ),
        }
    if mae_from_one <= 2e-2 and max_abs_error <= 5e-2:
        return {
            "row_sum_quality": "mildly_misnormalized",
            "row_sum_issue_severity": "medium",
            "row_sum_suggestion": (
                "Investigate calibration output construction. Row-wise normalization is a plausible "
                "repair, but compare pre/post-normalization log loss and multiclass Brier before adopting it."
            ),
        }
    return {
        "row_sum_quality": "strongly_misnormalized",
        "row_sum_issue_severity": "high",
        "row_sum_suggestion": (
            "Do not treat these outputs as proper multiclass probabilities. Either normalize rows and "
            "re-evaluate calibration, or redesign the calibration step so the wrapper emits simplex-valid probabilities directly."
        ),
    }


def multiclass_log_loss(y_true, proba, class_labels, eps=1e-15):
    """Compute multiclass negative log likelihood using class-aligned columns."""
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    class_indices = np.array([label_to_index[label] for label in y_true], dtype=int)
    selected = np.clip(proba[np.arange(len(y_true)), class_indices], eps, 1.0)
    return float(-np.mean(np.log(selected)))


def top1_metrics(y_true, proba):
    """Compute top-1 diagnostics from a probability matrix."""
    preds = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    corrects = (preds == y_true).astype(int)
    acc = accuracy_score(y_true, preds)
    top1_brier = brier_score_loss(corrects, conf)
    top1_ece, fraction_of_positives, mean_predicted_value = expected_calibration_error(
        corrects, conf
    )
    return {
        "accuracy": float(acc),
        "top1_brier": float(top1_brier),
        "top1_ece": float(top1_ece),
        "corrects": corrects,
        "conf": conf,
        "fop": fraction_of_positives,
        "mpv": mean_predicted_value,
    }


def compare_metric_vectors(left_values, right_values, lower_is_better, tolerance=1e-12):
    """Return wins/losses/ties and Wilcoxon p-value for paired metric vectors."""
    left_values = np.asarray(left_values, dtype=float)
    right_values = np.asarray(right_values, dtype=float)
    valid = ~(np.isnan(left_values) | np.isnan(right_values))
    left_values = left_values[valid]
    right_values = right_values[valid]
    if left_values.size == 0:
        return {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "wilcoxon_pvalue": float("nan"),
            "n_valid_pairs": 0,
        }
    diff = left_values - right_values
    if lower_is_better:
        wins = int(np.sum(diff < -tolerance))
        losses = int(np.sum(diff > tolerance))
    else:
        wins = int(np.sum(diff > tolerance))
        losses = int(np.sum(diff < -tolerance))
    ties = int(left_values.size - wins - losses)
    if left_values.size < 2:
        pvalue = float("nan")
    elif np.all(np.abs(diff) <= tolerance):
        pvalue = 1.0
    else:
        pvalue = float(wilcoxon(left_values, right_values, zero_method="wilcox").pvalue)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "wilcoxon_pvalue": pvalue,
        "n_valid_pairs": int(left_values.size),
    }


def compute_pairwise_statistics(results_df):
    """Compute wins/losses/ties and Wilcoxon tests across datasets."""
    pairwise_rows = []
    for metric_name in PAIRWISE_METRICS:
        pivot = results_df.pivot(index="dataset", columns="method", values=metric_name)
        for left_method, right_method in combinations(pivot.columns, 2):
            stats = compare_metric_vectors(
                pivot[left_method].to_numpy(),
                pivot[right_method].to_numpy(),
                lower_is_better=LOWER_IS_BETTER[metric_name],
            )
            pairwise_rows.append(
                {
                    "metric": metric_name,
                    "method_left": left_method,
                    "method_right": right_method,
                    **stats,
                }
            )
    return pd.DataFrame(pairwise_rows)


def compute_sample_weighted_summary(results_df, summary_cols):
    """Compute sample-weighted averages across datasets."""
    weighted_rows = []
    for method_name, method_frame in results_df.groupby("method"):
        weights = method_frame["n_samples"].to_numpy(dtype=float)
        weighted_row = {"method": method_name}
        for column in summary_cols:
            values = method_frame[column].to_numpy(dtype=float)
            valid = ~np.isnan(values)
            if not np.any(valid):
                weighted_row[column] = float("nan")
                continue
            weighted_row[column] = round(float(np.average(values[valid], weights=weights[valid])), 4)
        weighted_rows.append(weighted_row)
    return pd.DataFrame(weighted_rows).set_index("method")


def load_dataset(dataset_name):
    """Load a dataset and remap labels to contiguous integer ids."""
    df = pd.read_csv(os.path.join(DATA_DIR, f"{dataset_name}.csv"), sep=";")
    x_values = df.iloc[:, :-1].values
    raw_labels = df.iloc[:, -1].values
    _, encoded_labels = np.unique(raw_labels, return_inverse=True)
    x_values, encoded_labels = shuffle(x_values, encoded_labels, random_state=0)
    return x_values, encoded_labels


def _extract_proba_matrix(proba_output: Any) -> np.ndarray:
    """Extract the probability matrix from a wrapper predict_proba payload."""
    if isinstance(proba_output, tuple):
        return np.asarray(proba_output[0], dtype=float)
    return np.asarray(proba_output, dtype=float)


def _require_wrapper_state(wrapper: WrapCalibratedExplainer, *, fitted: bool, calibrated: bool, name: str):
    """Fail fast if a CE wrapper is not in the expected state."""
    if fitted and not getattr(wrapper, "fitted", False):
        raise RuntimeError(f"{name} wrapper is not fitted.")
    if calibrated and not getattr(wrapper, "calibrated", False):
        raise RuntimeError(f"{name} wrapper is not calibrated.")


def _build_shared_wrappers(x_prop, y_prop, x_cal, y_cal, split_seed):
    """Fit one learner and return two wrappers that share it."""
    learner = RandomForestClassifier(random_state=split_seed)
    fit_wrapper = ensure_ce_first_wrapper(learner)
    fit_wrapper.fit(x_prop, y_prop)
    _require_wrapper_state(fit_wrapper, fitted=True, calibrated=False, name="fit")

    shared_learner = fit_wrapper.learner
    ce_wrapper = ensure_ce_first_wrapper(shared_learner)
    ce_multi_wrapper = ensure_ce_first_wrapper(shared_learner)

    if ce_wrapper.learner is not ce_multi_wrapper.learner:
        raise RuntimeError("CE and CE_multi wrappers do not share the same learner instance.")

    ce_wrapper.calibrate(x_cal, y_cal, mode="classification")
    ce_multi_wrapper.calibrate(x_cal, y_cal, mode="classification", multi_labels_enabled=True)
    _require_wrapper_state(ce_wrapper, fitted=True, calibrated=True, name="CE")
    _require_wrapper_state(ce_multi_wrapper, fitted=True, calibrated=True, name="CE_multi")
    return ce_wrapper, ce_multi_wrapper


def run_fold_predictions(x_prop, y_prop, x_cal, y_cal, x_test, split_seed):
    """Return CE-first probability outputs for one fold."""
    ce_wrapper, ce_multi_wrapper = _build_shared_wrappers(
        x_prop=x_prop,
        y_prop=y_prop,
        x_cal=x_cal,
        y_cal=y_cal,
        split_seed=split_seed,
    )

    uncal_proba = _extract_proba_matrix(ce_wrapper.predict_proba(x_test, calibrated=False))
    ce_proba = _extract_proba_matrix(ce_wrapper.predict_proba(x_test))
    ce_multi_proba = _extract_proba_matrix(ce_multi_wrapper.predict_proba(x_test))
    ce_multi_max_abs_diff = float(np.max(np.abs(ce_multi_proba - ce_proba)))
    ce_multi_distinct = bool(not np.allclose(ce_multi_proba, ce_proba))

    return {
        "Uncal": uncal_proba,
        "CE": ce_proba,
        "CE_multi": ce_multi_proba,
        "metadata": {
            "shared_learner_verified": True,
            "ce_multi_prediction_distinct": ce_multi_distinct,
            "ce_multi_prediction_max_abs_diff": ce_multi_max_abs_diff,
        },
    }


def run_dataset_experiment(dataset_name):
    """Run the full experiment for one dataset or return a skip record."""
    print(f"[{dataset_name}] starting ...")
    x_values, y_values = load_dataset(dataset_name)
    class_labels = np.unique(y_values)
    n_classes = len(class_labels)
    min_class_count = int(np.bincount(y_values).min())

    if min_class_count < N_SPLITS:
        skip_reason = f"minimum class count {min_class_count} is smaller than n_splits={N_SPLITS}"
        print(f"[{dataset_name}] skipped: {skip_reason}")
        return None, {"dataset": dataset_name, "n_classes": n_classes, "reason": skip_reason}

    batches = {method_name: [] for method_name in METHODS}
    discarded_outer_loops = 0
    ce_multi_distinct_flags = []
    ce_multi_max_abs_diffs = []

    for outer in range(OUTER_LOOPS):
        fold_predictions = {
            method_name: np.zeros((len(y_values), n_classes), dtype=float) for method_name in METHODS
        }
        outer_invalid_reason = None

        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=outer)
        for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x_values, y_values)):
            split_seed = outer * N_SPLITS + fold_index
            x_train, x_test = x_values[train_idx], x_values[test_idx]
            y_train = y_values[train_idx]

            x_prop, x_cal, y_prop, y_cal = train_test_split(
                x_train,
                y_train,
                test_size=CALIBRATION_SIZE,
                random_state=split_seed,
                stratify=y_train,
            )

            unique_prop_y = np.unique(y_prop)
            unique_cal_y = np.unique(y_cal)
            if len(unique_prop_y) != n_classes or len(unique_cal_y) != n_classes:
                outer_invalid_reason = (
                    "class-coverage failure after stratified split "
                    f"(outer={outer}, fold={fold_index}, "
                    f"unique_y_prop={len(unique_prop_y)}, unique_y_cal={len(unique_cal_y)})"
                )
                break

            try:
                fold_result = run_fold_predictions(x_prop, y_prop, x_cal, y_cal, x_test, split_seed)
            except Exception as exc:  # pragma: no cover - defensive fold guard
                outer_invalid_reason = f"ce-first fold failure (outer={outer}, fold={fold_index}): {exc}"
                break

            for method_name in METHODS:
                fold_predictions[method_name][test_idx] = fold_result[method_name]
            ce_multi_distinct_flags.append(fold_result["metadata"]["ce_multi_prediction_distinct"])
            ce_multi_max_abs_diffs.append(fold_result["metadata"]["ce_multi_prediction_max_abs_diff"])

        if outer_invalid_reason is not None:
            discarded_outer_loops += 1
            print(f"[{dataset_name}] discarded outer={outer}: {outer_invalid_reason}")
            continue

        for method_name in METHODS:
            batches[method_name].append(fold_predictions[method_name])

    if not batches["Uncal"]:
        skip_reason = (
            "no valid outer repeats remained after class-coverage and CE-first checks "
            f"(discarded_outer_loops={discarded_outer_loops})"
        )
        print(f"[{dataset_name}] skipped: {skip_reason}")
        return None, {"dataset": dataset_name, "n_classes": n_classes, "reason": skip_reason}

    y_all = np.tile(y_values, len(batches["Uncal"]))
    method_outputs = {method_name: np.vstack(method_batches) for method_name, method_batches in batches.items()}
    valid_outer_loops = len(batches["Uncal"])
    ce_multi_distinct_rate = float(np.mean(ce_multi_distinct_flags)) if ce_multi_distinct_flags else 0.0
    ce_multi_max_abs_diff = float(np.max(ce_multi_max_abs_diffs)) if ce_multi_max_abs_diffs else 0.0

    if ce_multi_distinct_rate == 0.0:
        warnings.warn(
            f"[{dataset_name}] CE_multi predictions matched CE on every evaluated fold; "
            "multi_labels_enabled=True does not currently produce distinct prediction-time behavior.",
            UserWarning,
            stacklevel=2,
        )

    print(f"[{dataset_name}] computing metrics ...")
    eval_rows = []
    classwise_rows = []
    plot_data = {}
    for method_name in METHODS:
        proba = method_outputs[method_name]
        row_sums = proba.sum(axis=1)
        row_sum_summary = summarize_row_sum_quality(row_sums)
        top1 = top1_metrics(y_all, proba)
        plot_data[method_name] = {
            "row_sums": row_sums,
            "fop": top1["fop"],
            "mpv": top1["mpv"],
        }
        classwise_rows.extend(
            per_class_metric_rows(
                dataset_name=dataset_name,
                method_name=method_name,
                y_true=y_all,
                proba=proba,
                class_labels=class_labels,
                n_samples=len(y_all),
            )
        )
        eval_rows.append(
            {
                "dataset": dataset_name,
                "n_classes": n_classes,
                "n_samples": len(y_values),
                "n_evaluated_samples": len(y_all),
                "valid_outer_loops": valid_outer_loops,
                "discarded_outer_loops": discarded_outer_loops,
                "method": method_name,
                "is_multiclass_distribution": True,
                "shared_learner_required": True,
                "comparison_protocol": "shared_learner_ce_first",
                "calibration_holdout_fraction": CALIBRATION_SIZE,
                "ce_multi_prediction_distinct_rate": round(ce_multi_distinct_rate, 6),
                "ce_multi_prediction_max_abs_diff": round(ce_multi_max_abs_diff, 10),
                "row_sum_mean": round(float(row_sums.mean()), 6),
                "row_sum_std": round(float(row_sums.std()), 6),
                "row_sum_min": round(float(row_sums.min()), 6),
                "row_sum_max": round(float(row_sums.max()), 6),
                "row_sum_mae_from_one": round(float(np.mean(np.abs(row_sums - 1.0))), 6),
                "row_sum_quality": row_sum_summary["row_sum_quality"],
                "row_sum_issue_severity": row_sum_summary["row_sum_issue_severity"],
                "row_sum_suggestion": row_sum_summary["row_sum_suggestion"],
                "accuracy": round(top1["accuracy"], 4),
                "top1_brier": round(top1["top1_brier"], 4),
                "top1_ece": round(top1["top1_ece"], 4),
                "per_class_brier": round(per_class_brier(y_all, proba, class_labels), 4),
                "per_class_ece": round(per_class_ece(y_all, proba, class_labels), 4),
                "multiclass_brier": round(multiclass_brier_score(y_all, proba, class_labels), 4),
                "log_loss": round(multiclass_log_loss(y_all, proba, class_labels), 4),
            }
        )

    plt = _get_pyplot()
    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(14, 6))
    axis_left.plot([0, 1], [0, 1], "k:", label="Perfect")
    for method_name, method_plot_data in plot_data.items():
        color = COLORS[method_name]
        axis_left.plot(
            method_plot_data["mpv"],
            method_plot_data["fop"],
            "s-",
            color=color,
            label=method_name,
        )
        axis_right.hist(
            method_plot_data["row_sums"],
            bins="auto",
            alpha=0.5,
            label=method_name,
            density=True,
            color=color,
        )

    axis_left.set_title(f"{dataset_name} - Top-1 reliability")
    axis_left.set_xlabel("Mean predicted confidence")
    axis_left.set_ylabel("Fraction correct")
    axis_left.set_xlim(0, 1)
    axis_left.set_ylim(0, 1)
    axis_left.legend()

    axis_right.axvline(1.0, color="k", linestyle=":", linewidth=2, label="Sum = 1")
    axis_right.set_title(f"{dataset_name} - Row-sum distribution")
    axis_right.set_xlabel("Sum of class probabilities per instance")
    axis_right.set_ylabel("Density")
    axis_right.legend()

    plt.suptitle(f"{dataset_name} ({n_classes} classes)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{dataset_name}_multilabel.png"), dpi=100)
    plt.close()
    print(f"[{dataset_name}] done.")
    return {"eval_rows": eval_rows, "classwise_rows": classwise_rows}, None


def main():
    """Run the multiclass CE-first experiment end-to-end."""
    eval_rows = []
    classwise_rows = []
    skipped_rows = []

    for dataset_name in DATASETS:
        dataset_result, skip_row = run_dataset_experiment(dataset_name)
        if skip_row is not None:
            skipped_rows.append(skip_row)
            continue
        eval_rows.extend(dataset_result["eval_rows"])
        classwise_rows.extend(dataset_result["classwise_rows"])

    if skipped_rows:
        skipped_df = pd.DataFrame(skipped_rows)
        skipped_path = os.path.join(RESULTS_DIR, "multilabel_skipped_datasets.csv")
        skipped_df.to_csv(skipped_path, index=False, sep=";")
        print(f"Skipped datasets saved to {skipped_path}")

    if not eval_rows:
        print("No valid datasets were evaluated.")
        return

    results_df = pd.DataFrame(eval_rows)
    classwise_df = pd.DataFrame(classwise_rows)
    results_path = os.path.join(RESULTS_DIR, "multilabel_experiment.csv")
    classwise_path = os.path.join(RESULTS_DIR, "multilabel_classwise_metrics.csv")
    results_df.to_csv(results_path, index=False, sep=";")
    classwise_df.to_csv(classwise_path, index=False, sep=";")
    print(f"\nResults saved to {results_path}")
    print(f"Classwise metrics saved to {classwise_path}")

    summary_cols = [
        "log_loss",
        "multiclass_brier",
        "per_class_ece",
        "per_class_brier",
        "row_sum_mean",
        "row_sum_std",
        "row_sum_mae_from_one",
        "accuracy",
        "top1_brier",
        "top1_ece",
    ]
    summary = results_df.groupby("method")[summary_cols].mean().round(4)
    weighted_summary = compute_sample_weighted_summary(results_df, summary_cols)
    pairwise_stats = compute_pairwise_statistics(results_df)

    print("\n=== Grand average across datasets ===")
    print(summary.to_string())
    print("\n=== Sample-weighted average across datasets ===")
    print(weighted_summary.to_string())

    distinct_summary = (
        results_df[["dataset", "ce_multi_prediction_distinct_rate", "ce_multi_prediction_max_abs_diff"]]
        .drop_duplicates()
        .sort_values("dataset")
    )
    print("\n=== CE_multi distinctness by dataset ===")
    print(distinct_summary.to_string(index=False))

    summary.to_csv(os.path.join(RESULTS_DIR, "multilabel_summary.csv"), sep=";")
    weighted_summary.to_csv(
        os.path.join(RESULTS_DIR, "multilabel_weighted_summary.csv"),
        sep=";",
    )
    pairwise_stats.to_csv(
        os.path.join(RESULTS_DIR, "multilabel_pairwise_stats.csv"),
        index=False,
        sep=";",
    )

    plt = _get_pyplot()
    pivot = results_df.pivot(index="dataset", columns="method", values="row_sum_mean")
    figure, axis = plt.subplots(figsize=(14, 5))
    x_positions = np.arange(len(pivot))
    width = 0.25
    for method_index, method_name in enumerate(METHODS):
        if method_name in pivot.columns:
            axis.bar(
                x_positions + method_index * width,
                pivot[method_name],
                width=width,
                label=method_name,
                color=COLORS[method_name],
            )
    axis.axhline(1.0, color="k", linestyle=":", linewidth=1.5)
    axis.set_xticks(x_positions + width)
    axis.set_xticklabels(pivot.index, rotation=45, ha="right")
    axis.set_ylabel("Mean row sum of class probabilities")
    axis.set_title("Row-sum by dataset and method (ideal = 1.0)")
    axis.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "summary_row_sums.png"), dpi=100)
    plt.close()
    print("Summary plot saved.")


if __name__ == "__main__":
    main()
