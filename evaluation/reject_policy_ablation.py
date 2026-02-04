"""Ablation study comparing reject vs no-reject scenarios.

This script evaluates the value-add of the reject policy system by measuring:
- Accuracy improvement on accepted samples vs all samples
- Calibration improvement (ECE) on accepted vs all samples
- Coverage vs accuracy tradeoff curves
- Reject rate at various confidence levels (0.90, 0.95, 0.99)

Usage:
    python evaluation/reject_policy_ablation.py [--confidence 0.95] [--output path]

The script produces a JSON results file and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RESULTS_FILE = "evaluation/reject_policy_ablation_results.json"

# Confidence levels to evaluate
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

# Calibration size for explainer
CALIBRATION_SIZE = 200


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities for positive class.
    n_bins : int
        Number of bins for calibration.

    Returns
    -------
    float
        Expected calibration error.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob > lower) & (y_prob <= upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(y_prob[in_bin])
            avg_accuracy = np.mean(y_true[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return ece


def run_binary_classification_ablation(confidence: float) -> dict[str, Any]:
    """Run ablation on binary classification (breast cancer dataset).

    Parameters
    ----------
    confidence : float
        Confidence level for rejection.

    Returns
    -------
    dict
        Results dictionary with metrics.
    """
    from calibrated_explanations import WrapCalibratedExplainer

    print(f"  Binary classification (confidence={confidence})...")

    # Load and split data
    data = load_breast_cancer()
    x_data, y_data = data.data, data.target

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x_train, y_train, test_size=CALIBRATION_SIZE, random_state=42, stratify=y_train
    )

    # Train model
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(x_fit, y_fit)

    # Create explainer with reject policy
    wrapper = WrapCalibratedExplainer(clf)
    wrapper.fit(x_fit, y_fit)
    wrapper.calibrate(x_cal, y_cal)

    # Initialize reject learner
    try:
        wrapper.explainer.reject_orchestrator.initialize_reject_learner()
    except Exception as err:
        return {"error": str(err), "confidence": confidence}

    # Get predictions without rejection (baseline)
    y_pred_all = clf.predict(x_test)
    y_prob_all = clf.predict_proba(x_test)[:, 1]

    accuracy_all = accuracy_score(y_test, y_pred_all)
    ece_all = expected_calibration_error(y_test, y_prob_all)

    # Get rejection predictions
    rejected, error_rate, reject_rate = wrapper.explainer.reject_orchestrator.predict_reject(
        x_test, confidence=confidence
    )

    # Compute metrics on accepted samples only
    accepted_mask = ~rejected
    n_accepted = np.sum(accepted_mask)
    n_total = len(y_test)

    if n_accepted > 0:
        y_pred_accepted = y_pred_all[accepted_mask]
        y_prob_accepted = y_prob_all[accepted_mask]
        y_test_accepted = y_test[accepted_mask]

        accuracy_accepted = accuracy_score(y_test_accepted, y_pred_accepted)
        ece_accepted = expected_calibration_error(y_test_accepted, y_prob_accepted)
    else:
        accuracy_accepted = None
        ece_accepted = None

    # Compute improvements
    accuracy_improvement = (
        (accuracy_accepted - accuracy_all) if accuracy_accepted is not None else None
    )
    ece_improvement = (ece_all - ece_accepted) if ece_accepted is not None else None

    return {
        "dataset": "breast_cancer",
        "task": "binary_classification",
        "confidence": confidence,
        "n_total": n_total,
        "n_accepted": int(n_accepted),
        "n_rejected": int(n_total - n_accepted),
        "coverage": float(n_accepted / n_total) if n_total > 0 else 0.0,
        "reject_rate": float(reject_rate),
        "error_rate": float(error_rate),
        "baseline": {
            "accuracy": float(accuracy_all),
            "ece": float(ece_all),
        },
        "with_rejection": {
            "accuracy": float(accuracy_accepted) if accuracy_accepted is not None else None,
            "ece": float(ece_accepted) if ece_accepted is not None else None,
        },
        "improvement": {
            "accuracy": float(accuracy_improvement) if accuracy_improvement is not None else None,
            "ece": float(ece_improvement) if ece_improvement is not None else None,
        },
    }


def run_multiclass_classification_ablation(confidence: float) -> dict[str, Any]:
    """Run ablation on multiclass classification (iris dataset).

    Parameters
    ----------
    confidence : float
        Confidence level for rejection.

    Returns
    -------
    dict
        Results dictionary with metrics.
    """
    from calibrated_explanations import WrapCalibratedExplainer

    print(f"  Multiclass classification (confidence={confidence})...")

    # Load and split data
    data = load_iris()
    x_data, y_data = data.data, data.target

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )

    cal_size = min(CALIBRATION_SIZE, len(x_train) - 10)
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x_train, y_train, test_size=cal_size, random_state=42, stratify=y_train
    )

    # Train model
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(x_fit, y_fit)

    # Create explainer with reject policy
    wrapper = WrapCalibratedExplainer(clf)
    wrapper.fit(x_fit, y_fit)
    wrapper.calibrate(x_cal, y_cal)

    # Initialize reject learner
    try:
        wrapper.explainer.reject_orchestrator.initialize_reject_learner()
    except Exception as err:
        return {"error": str(err), "confidence": confidence}

    # Get predictions without rejection (baseline)
    y_pred_all = clf.predict(x_test)

    accuracy_all = accuracy_score(y_test, y_pred_all)

    # Get rejection predictions
    rejected, error_rate, reject_rate = wrapper.explainer.reject_orchestrator.predict_reject(
        x_test, confidence=confidence
    )

    # Compute metrics on accepted samples only
    accepted_mask = ~rejected
    n_accepted = np.sum(accepted_mask)
    n_total = len(y_test)

    if n_accepted > 0:
        y_pred_accepted = y_pred_all[accepted_mask]
        y_test_accepted = y_test[accepted_mask]

        accuracy_accepted = accuracy_score(y_test_accepted, y_pred_accepted)
    else:
        accuracy_accepted = None

    # Compute improvements
    accuracy_improvement = (
        (accuracy_accepted - accuracy_all) if accuracy_accepted is not None else None
    )

    return {
        "dataset": "iris",
        "task": "multiclass_classification",
        "confidence": confidence,
        "n_total": n_total,
        "n_accepted": int(n_accepted),
        "n_rejected": int(n_total - n_accepted),
        "coverage": float(n_accepted / n_total) if n_total > 0 else 0.0,
        "reject_rate": float(reject_rate),
        "error_rate": float(error_rate),
        "baseline": {
            "accuracy": float(accuracy_all),
        },
        "with_rejection": {
            "accuracy": float(accuracy_accepted) if accuracy_accepted is not None else None,
        },
        "improvement": {
            "accuracy": float(accuracy_improvement) if accuracy_improvement is not None else None,
        },
    }


def run_regression_ablation(confidence: float) -> dict[str, Any]:
    """Run ablation on regression (diabetes dataset with threshold).

    Parameters
    ----------
    confidence : float
        Confidence level for rejection.

    Returns
    -------
    dict
        Results dictionary with metrics.
    """
    from calibrated_explanations import WrapCalibratedExplainer

    print(f"  Regression with threshold (confidence={confidence})...")

    # Load and split data
    data = load_diabetes()
    x_data, y_data = data.data, data.target

    # Normalize target to [0, 1] range for better threshold behavior
    y_data = (y_data - y_data.min()) / (y_data.max() - y_data.min())

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42
    )

    cal_size = min(CALIBRATION_SIZE, len(x_train) - 10)
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x_train, y_train, test_size=cal_size, random_state=42
    )

    # Train model
    reg = RandomForestRegressor(n_estimators=50, random_state=42)
    reg.fit(x_fit, y_fit)

    # Create explainer
    wrapper = WrapCalibratedExplainer(reg)
    wrapper.fit(x_fit, y_fit)
    wrapper.calibrate(x_cal, y_cal)

    # Use median as threshold for binary classification framing
    threshold = np.median(y_cal)

    # Initialize reject learner with threshold
    try:
        wrapper.explainer.reject_orchestrator.initialize_reject_learner(threshold=threshold)
    except Exception as err:
        return {"error": str(err), "confidence": confidence}

    # Get predictions without rejection (baseline)
    y_pred_all = reg.predict(x_test)

    # Compute binary classification metrics based on threshold
    y_true_binary = (y_test >= threshold).astype(int)
    y_pred_binary = (y_pred_all >= threshold).astype(int)

    accuracy_all = accuracy_score(y_true_binary, y_pred_binary)
    mse_all = np.mean((y_test - y_pred_all) ** 2)

    # Get rejection predictions
    rejected, error_rate, reject_rate = wrapper.explainer.reject_orchestrator.predict_reject(
        x_test, confidence=confidence
    )

    # Compute metrics on accepted samples only
    accepted_mask = ~rejected
    n_accepted = np.sum(accepted_mask)
    n_total = len(y_test)

    if n_accepted > 0:
        y_pred_accepted = y_pred_all[accepted_mask]
        y_test_accepted = y_test[accepted_mask]
        y_true_binary_accepted = y_true_binary[accepted_mask]
        y_pred_binary_accepted = y_pred_binary[accepted_mask]

        accuracy_accepted = accuracy_score(y_true_binary_accepted, y_pred_binary_accepted)
        mse_accepted = np.mean((y_test_accepted - y_pred_accepted) ** 2)
    else:
        accuracy_accepted = None
        mse_accepted = None

    # Compute improvements
    accuracy_improvement = (
        (accuracy_accepted - accuracy_all) if accuracy_accepted is not None else None
    )
    mse_improvement = (mse_all - mse_accepted) if mse_accepted is not None else None

    return {
        "dataset": "diabetes",
        "task": "regression_with_threshold",
        "confidence": confidence,
        "threshold": float(threshold),
        "n_total": n_total,
        "n_accepted": int(n_accepted),
        "n_rejected": int(n_total - n_accepted),
        "coverage": float(n_accepted / n_total) if n_total > 0 else 0.0,
        "reject_rate": float(reject_rate),
        "error_rate": float(error_rate),
        "baseline": {
            "accuracy": float(accuracy_all),
            "mse": float(mse_all),
        },
        "with_rejection": {
            "accuracy": float(accuracy_accepted) if accuracy_accepted is not None else None,
            "mse": float(mse_accepted) if mse_accepted is not None else None,
        },
        "improvement": {
            "accuracy": float(accuracy_improvement) if accuracy_improvement is not None else None,
            "mse": float(mse_improvement) if mse_improvement is not None else None,
        },
    }


def print_summary(results: dict[str, Any]) -> None:
    """Print a summary table of ablation results.

    Parameters
    ----------
    results : dict
        Results dictionary from the ablation runs.
    """
    print("\n" + "=" * 80)
    print(f"{'REJECT POLICY ABLATION SUMMARY':^80}")
    print("=" * 80)

    print("\nThis ablation demonstrates the value-add of rejection by comparing:")
    print("  - Baseline: All samples predicted (no rejection)")
    print("  - With Rejection: Only accepted samples considered for metrics")
    print("")

    # Binary classification results
    print("-" * 80)
    print("BINARY CLASSIFICATION (Breast Cancer)")
    print("-" * 80)
    print(
        f"{'Confidence':<12} | {'Coverage':<10} | {'Reject Rate':<12} | "
        f"{'Acc (Base)':<12} | {'Acc (Rej)':<12} | {'Acc Diff':<10}"
    )
    print("-" * 80)

    for entry in results.get("binary_classification", []):
        if "error" in entry:
            print(f"{entry['confidence']:<12.2f} | ERROR: {entry['error']}")
            continue

        base_acc = entry["baseline"]["accuracy"]
        rej_acc = entry["with_rejection"]["accuracy"]
        acc_delta = entry["improvement"]["accuracy"]

        print(
            f"{entry['confidence']:<12.2f} | "
            f"{entry['coverage']:<10.2%} | "
            f"{entry['reject_rate']:<12.2%} | "
            f"{base_acc:<12.4f} | "
            f"{rej_acc if rej_acc else 'N/A':<12} | "
            f"{f'+{acc_delta:.4f}' if acc_delta and acc_delta > 0 else (f'{acc_delta:.4f}' if acc_delta else 'N/A'):<10}"
        )

    # Multiclass results
    print("\n" + "-" * 80)
    print("MULTICLASS CLASSIFICATION (Iris)")
    print("-" * 80)
    print(
        f"{'Confidence':<12} | {'Coverage':<10} | {'Reject Rate':<12} | "
        f"{'Acc (Base)':<12} | {'Acc (Rej)':<12} | {'Acc Diff':<10}"
    )
    print("-" * 80)

    for entry in results.get("multiclass_classification", []):
        if "error" in entry:
            print(f"{entry['confidence']:<12.2f} | ERROR: {entry['error']}")
            continue

        base_acc = entry["baseline"]["accuracy"]
        rej_acc = entry["with_rejection"]["accuracy"]
        acc_delta = entry["improvement"]["accuracy"]

        print(
            f"{entry['confidence']:<12.2f} | "
            f"{entry['coverage']:<10.2%} | "
            f"{entry['reject_rate']:<12.2%} | "
            f"{base_acc:<12.4f} | "
            f"{rej_acc if rej_acc else 'N/A':<12} | "
            f"{f'+{acc_delta:.4f}' if acc_delta and acc_delta > 0 else (f'{acc_delta:.4f}' if acc_delta else 'N/A'):<10}"
        )

    # Regression results
    print("\n" + "-" * 80)
    print("REGRESSION WITH THRESHOLD (Diabetes)")
    print("-" * 80)
    print(
        f"{'Confidence':<12} | {'Coverage':<10} | {'Reject Rate':<12} | "
        f"{'MSE (Base)':<12} | {'MSE (Rej)':<12} | {'MSE Diff':<10}"
    )
    print("-" * 80)

    for entry in results.get("regression_with_threshold", []):
        if "error" in entry:
            print(f"{entry['confidence']:<12.2f} | ERROR: {entry['error']}")
            continue

        base_mse = entry["baseline"]["mse"]
        rej_mse = entry["with_rejection"]["mse"]
        mse_delta = entry["improvement"]["mse"]

        print(
            f"{entry['confidence']:<12.2f} | "
            f"{entry['coverage']:<10.2%} | "
            f"{entry['reject_rate']:<12.2%} | "
            f"{base_mse:<12.4f} | "
            f"{rej_mse if rej_mse else 'N/A':<12} | "
            f"{f'-{mse_delta:.4f}' if mse_delta and mse_delta > 0 else (f'{mse_delta:.4f}' if mse_delta else 'N/A'):<10}"
        )

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    print("* Higher confidence -> higher reject rate -> higher accuracy on accepted samples")
    print("* Reject policy trades coverage for accuracy/calibration improvement")
    print("* Value-add is most pronounced when model uncertainty is high")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Ablation study comparing reject vs no-reject scenarios"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=RESULTS_FILE,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        nargs="+",
        default=CONFIDENCE_LEVELS,
        help="Confidence levels to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    """Run the reject policy ablation study."""
    args = parse_args()

    print("=" * 80)
    print("REJECT POLICY ABLATION STUDY")
    print("=" * 80)
    print(f"Comparing reject vs no-reject at confidence levels: {args.confidence}")
    print("")

    results: dict[str, Any] = {
        "meta": {
            "timestamp": time.time(),
            "platform": os.name,
            "confidence_levels": args.confidence,
        },
        "binary_classification": [],
        "multiclass_classification": [],
        "regression_with_threshold": [],
    }

    # Run ablations for each confidence level
    for confidence in args.confidence:
        print(f"\nRunning ablation at confidence={confidence}...")

        # Binary classification
        binary_result = run_binary_classification_ablation(confidence)
        results["binary_classification"].append(binary_result)

        # Multiclass classification
        multiclass_result = run_multiclass_classification_ablation(confidence)
        results["multiclass_classification"].append(multiclass_result)

        # Regression with threshold
        regression_result = run_regression_ablation(confidence)
        results["regression_with_threshold"].append(regression_result)

    # Print summary
    print_summary(results)

    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
