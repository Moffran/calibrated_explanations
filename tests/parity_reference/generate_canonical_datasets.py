"""Generate synthetic canonical datasets for parity reference tests.

Produces four fixture JSON files in the same directory:
- canonical_dataset.json (binary classification)
- canonical_dataset_regression.json (regression)
- canonical_dataset_multiclass.json (multiclass classification)
- canonical_dataset_probabilistic_regression.json (regression-like)

Each dataset contains >=20 features (mix of categorical integer-coded and numeric),
1000 instances, and splits for train/cal/test. This script is deterministic (seeded).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.datasets import make_classification, make_regression


ROOT = Path(__file__).resolve().parent


def to_serializable(obj: Any) -> Any:
    """Convert nested structures and numpy types to JSON-serializable values.

    Parameters
    ----------
    obj : Any
        Object to convert (may be dict, list, tuple, numpy array, or scalar).

    Returns
    -------
    Any
        A JSON-serializable representation of ``obj`` (lists, dicts, and
        Python scalars).
    """
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_serializable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def make_splits(x: np.ndarray, y: np.ndarray, *, train_frac=0.8, cal_frac=0.1):
    """Shuffle and split feature and label arrays into train/cal/test sets.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix with shape (n_samples, n_features).
    y : np.ndarray
        Target array with length n_samples.
    train_frac : float, optional
        Fraction of samples to use for training (default 0.8).
    cal_frac : float, optional
        Fraction of samples to use for calibration (default 0.1).

    Returns
    -------
    tuple
        A 5-tuple of lists: (x_train, y_train, x_cal, y_cal, x_test).
    """
    n = x.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng()
    rng.shuffle(idx)
    train_end = int(n * train_frac)
    cal_end = train_end + int(n * cal_frac)
    train_idx = idx[:train_end]
    cal_idx = idx[train_end:cal_end]
    test_idx = idx[cal_end:]
    return (
        x[train_idx].tolist(),
        y[train_idx].tolist(),
        x[cal_idx].tolist(),
        y[cal_idx].tolist(),
        x[test_idx].tolist(),
    )


def build_feature_names(n_features: int) -> List[str]:
    """Return a list of sequential feature names.

    Parameters
    ----------
    n_features : int
        Number of feature names to generate.

    Returns
    -------
    List[str]
        Feature names 'f0' .. 'f{n_features-1}'.
    """
    return [f"f{i}" for i in range(n_features)]


def generate_classification(
    path: Path,
    *,
    n_samples=1000,
    n_features=25,
    n_categorical=5,
    n_informative=10,
    n_classes=2,
    seed=42,
):
    """Generate and write a canonical classification dataset JSON file.

    The produced JSON contains train/cal/test splits, feature names, and
    metadata suitable for parity/reference tests.

    Parameters
    ----------
    path : Path
        Output file path to write the dataset JSON.
    n_samples : int, optional
        Number of samples to generate (default 1000).
    n_features : int, optional
        Total number of features including categorical ones (default 25).
    n_categorical : int, optional
        Number of integer-coded categorical features prepended to X.
    n_informative : int, optional
        Number of informative features for sklearn's generator.
    n_classes : int, optional
        Number of target classes.
    seed : int, optional
        Random seed for deterministic generation.
    """
    rng = np.random.RandomState(seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features - n_categorical,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        random_state=seed,
    )
    # prepend categorical integer-coded features
    cats = rng.randint(0, 4, size=(n_samples, n_categorical))

    X_full = np.hstack([cats.astype(float), X])

    x_train, y_train, x_cal, y_cal, x_test = make_splits(X_full, y, train_frac=0.8, cal_frac=0.1)

    data: Dict[str, Any] = {
        "mode": "classification",
        "feature_names": build_feature_names(n_features),
        "categorical_features": list(range(n_categorical)),
        "class_labels": [f"C{i}" for i in range(n_classes)],
        "x_train": x_train,
        "y_train": y_train,
        "x_cal": x_cal,
        "y_cal": y_cal,
        "x_test": x_test,
    }
    path.write_text(json.dumps(to_serializable(data), indent=2, sort_keys=True))


def generate_regression(path: Path, *, n_samples=1000, n_features=25, n_categorical=5, seed=42):
    """Generate and write a canonical regression dataset JSON file.

    Parameters
    ----------
    path : Path
        Output file path to write the dataset JSON.
    n_samples : int, optional
        Number of samples to generate (default 1000).
    n_features : int, optional
        Total number of features including categorical ones (default 25).
    n_categorical : int, optional
        Number of integer-coded categorical features prepended to X.
    seed : int, optional
        Random seed for deterministic generation.
    """
    rng = np.random.RandomState(seed)
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features - n_categorical, noise=0.1, random_state=seed
    )
    cats = rng.randint(0, 4, size=(n_samples, n_categorical))
    X_full = np.hstack([cats.astype(float), X])

    x_train, y_train, x_cal, y_cal, x_test = make_splits(X_full, y, train_frac=0.8, cal_frac=0.1)

    data: Dict[str, Any] = {
        "mode": "regression",
        "feature_names": build_feature_names(n_features),
        "categorical_features": list(range(n_categorical)),
        "x_train": x_train,
        "y_train": y_train,
        "x_cal": x_cal,
        "y_cal": y_cal,
        "x_test": x_test,
    }
    path.write_text(json.dumps(to_serializable(data), indent=2, sort_keys=True))


def main() -> None:
    """Generate all canonical dataset JSON fixtures in the module directory.

    Writes four files covering classification, regression, multiclass, and
    probabilistic-regression variants. This function is intended for
    development/test fixture regeneration.
    """
    generate_classification(
        ROOT / "canonical_dataset.json",
        n_samples=1000,
        n_features=25,
        n_categorical=5,
        n_informative=12,
        n_classes=2,
        seed=42,
    )
    generate_regression(
        ROOT / "canonical_dataset_regression.json",
        n_samples=1000,
        n_features=25,
        n_categorical=5,
        seed=42,
    )
    generate_classification(
        ROOT / "canonical_dataset_multiclass.json",
        n_samples=1000,
        n_features=25,
        n_categorical=5,
        n_informative=12,
        n_classes=5,
        seed=42,
    )
    # probabilistic_regression: use regression-like dataset but keep mode 'regression'
    generate_regression(
        ROOT / "canonical_dataset_probabilistic_regression.json",
        n_samples=1000,
        n_features=25,
        n_categorical=5,
        seed=43,
    )
    print("Canonical datasets generated in", ROOT)


if __name__ == "__main__":
    main()
