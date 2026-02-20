"""Dataset registry and loaders for the ensured evaluation.

This module intentionally lives under `evaluation/` to respect ADR-010
(core vs evaluation split). It provides a stable definition of what
"all datasets" means for the ensured paper evaluation.

The dataset universes are derived from existing evaluation scripts:
- Binary classification: the 25 datasets enumerated in
  `evaluation/Classification_Experiment_*.py`.
- Multiclass: the datasets enumerated in `evaluation/multiclass/Experiment_Multiclass.py`.
- Regression: the `.txt` datasets in `data/reg/` used by fast regression
  evaluation scripts.

All loaders are deterministic and avoid network dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


BINARY_CLASSIFICATION_DATASETS: tuple[str, ...] = (
    "pc1req",
    "haberman",
    "hepati",
    "transfusion",
    "spect",
    "heartS",
    "heartH",
    "heartC",
    "je4243",
    "vote",
    "kc2",
    "wbc",
    "kc3",
    "creditA",
    "diabetes",
    "iono",
    "liver",
    "je4042",
    "sonar",
    "spectf",
    "german",
    "ttt",
    "colic",
    "pc4",
    "kc1",
)


MULTICLASS_DATASETS: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class LoadedDataset:
    """In-memory representation used by ensure experiments."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    categorical_features: list[int]
    name: str
    task: Literal["binary", "multiclass", "regression"]


def infer_categorical_features(X_df: pd.DataFrame, max_unique: int = 10) -> list[int]:
    """Infer categorical columns by unique-count heuristic.

    Matches the heuristic used throughout the repo's evaluation scripts.
    """

    categorical: list[int] = []
    for idx, col in enumerate(X_df.columns):
        # Treat low-cardinality numeric columns as categorical for discretization.
        # Note: `nunique(dropna=False)` to treat missing as its own category.
        if X_df[col].nunique(dropna=False) < max_unique:
            categorical.append(idx)
    return categorical


def load_binary_dataset(name: str) -> LoadedDataset:
    """Load a binary classification dataset from `data/{name}.csv`.

    Expected format: semicolon-separated CSV with target column `Y`.
    """

    file_path = REPO_ROOT / "data" / f"{name}.csv"
    df = pd.read_csv(file_path, sep=";")
    if "Y" not in df.columns:
        raise ValueError(f"Binary dataset {name!r} missing required target column 'Y'.")

    y = df["Y"].to_numpy()
    X_df = df.drop(columns=["Y"])

    categorical = infer_categorical_features(X_df)

    return LoadedDataset(
        X=X_df.to_numpy(dtype=float, copy=False),
        y=y,
        feature_names=list(X_df.columns),
        categorical_features=categorical,
        name=name,
        task="binary",
    )


def load_multiclass_dataset(name: str) -> LoadedDataset:
    """Load a multiclass dataset from `data/Multiclass/multi/{name}.csv`.

    Expected format: semicolon-separated CSV, target = last column.
    """

    file_path = REPO_ROOT / "data" / "Multiclass" / "multi" / f"{name}.csv"
    df = pd.read_csv(file_path, sep=";")
    X_df = df.iloc[:, :-1]
    y = df.iloc[:, -1].to_numpy()

    # Existing evaluation scripts normalize labels to 0..K-1 if they start at 1.
    # Keep that convention to reduce surprises.
    if np.min(y) == 1:
        y = y - 1

    categorical = infer_categorical_features(X_df)

    return LoadedDataset(
        X=X_df.to_numpy(dtype=float, copy=False),
        y=y,
        feature_names=list(X_df.columns),
        categorical_features=categorical,
        name=name,
        task="multiclass",
    )


def list_regression_txt_datasets() -> list[str]:
    """Return base names for all regression datasets used in ensured eval.

    Historically, regression benchmarks in this repo are stored as:
    - `.txt` files with a `REGRESSION` target column.
    - a small number of `.csv` files (notably California Housing).

    The name of this function is kept for backwards compatibility with the
    ensure runner scripts.
    """

    reg_dir = REPO_ROOT / "data" / "reg"
    names = sorted([p.stem for p in reg_dir.glob("*.txt")])

    # CSV-based regression datasets used elsewhere in evaluation.
    if (reg_dir / "housing.csv").exists():
        names.append("housing")
    if (reg_dir / "HousingData.csv").exists():
        names.append("HousingData")

    return sorted(set(names))


def load_regression_dataset_from_txt(name: str) -> LoadedDataset:
    """Load a regression dataset from `data/reg/`.

    Supported inputs:
    - `{name}.txt` with target column `REGRESSION`.
    - `housing.csv` (California Housing) with target `median_house_value` and
      semicolon delimiter.
    - `HousingData.csv` with target `MEDV` and comma delimiter.

    The function name is kept for backwards compatibility with existing
    ensured evaluation scripts.
    """

    reg_dir = REPO_ROOT / "data" / "reg"

    # Special-case CSV datasets.
    if name == "housing":
        file_path = reg_dir / "housing.csv"
        df = pd.read_csv(file_path, sep=";")
        target_col = "median_house_value"
    elif name == "HousingData":
        file_path = reg_dir / "HousingData.csv"
        df = pd.read_csv(file_path, sep=",")
        target_col = "MEDV"
    else:
        file_path = reg_dir / f"{name}.txt"
        df = pd.read_csv(file_path)
        target_col = "REGRESSION"

    if target_col not in df.columns:
        raise ValueError(
            f"Regression dataset {name!r} missing required target column {target_col!r}."
        )

    y = df[target_col].to_numpy(dtype=float, copy=False)
    X_df = df.drop(columns=[target_col])

    categorical = infer_categorical_features(X_df)

    return LoadedDataset(
        X=X_df.to_numpy(dtype=float, copy=False),
        y=y,
        feature_names=list(X_df.columns),
        categorical_features=categorical,
        name=name,
        task="regression",
    )
