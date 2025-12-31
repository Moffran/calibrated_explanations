"""Dataset utilities for fast-filtering evaluations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration describing a dataset on disk."""

    name: str
    path: Path
    task: str
    target: str | int | None = None
    delimiter: str | None = None
    is_multiclass: bool = False


BINARY_CLASSIFICATION = [
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
]

MULTICLASS = [
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

REGRESSION_TXT = [
    "abalone",
    "boston",
    "bank8fh",
    "bank8fm",
    "bank8nh",
    "bank8nm",
    "kin8fh",
    "kin8fm",
    "kin8nh",
    "kin8nm",
    "puma8fh",
    "puma8fm",
    "puma8nh",
    "puma8nm",
    "friedm",
    "deltaA",
    "deltaE",
    "comp",
    "concreate",
    "cooling",
    "heating",
    "wineRed",
    "wineWhite",
]

REGRESSION_CSV = {
    "housing": (DATA_ROOT / "reg" / "housing.csv", "median_house_value", ";"),
    "HousingData": (DATA_ROOT / "reg" / "HousingData.csv", "MEDV", ","),
}


def _infer_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if ";" in sample and "," not in sample:
        return ";"
    return ","


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    return numeric_df.dropna(axis=0)


def load_dataset(spec: DatasetSpec, *, max_samples: int | None = None, random_state: int = 42):
    """Load a dataset to numpy arrays."""
    path = spec.path
    delimiter = spec.delimiter or _infer_delimiter(path)
    df = pd.read_csv(path, sep=delimiter)
    df = _coerce_numeric(df)

    if isinstance(spec.target, int):
        target_col = df.columns[int(spec.target)]
    elif isinstance(spec.target, str):
        target_col = spec.target
    else:
        target_col = df.columns[0]

    y = df[target_col].to_numpy()
    X = df.drop(columns=[target_col])

    if spec.task == "classification" and spec.is_multiclass:
        y = y.astype(int)
        if y.min() == 1:
            y = y - 1

    if max_samples is not None and len(X) > max_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=max_samples, replace=False)
        X = X.iloc[indices]
        y = y[indices]

    return X.to_numpy(), y, list(X.columns)


def resolve_dataset_specs(
    tasks: Iterable[str],
    *,
    limit: int | None = None,
    dataset_names: Sequence[str] | None = None,
) -> list[DatasetSpec]:
    """Return dataset specs for the requested tasks."""
    dataset_names = set(dataset_names or [])
    specs: list[DatasetSpec] = []
    for task in tasks:
        if task == "classification":
            names = BINARY_CLASSIFICATION
            for name in names:
                if dataset_names and name not in dataset_names:
                    continue
                specs.append(
                    DatasetSpec(
                        name=name,
                        path=DATA_ROOT / f"{name}.csv",
                        task="classification",
                        target="Y",
                        delimiter=";",
                        is_multiclass=False,
                    )
                )
        elif task == "multiclass":
            names = MULTICLASS
            for name in names:
                if dataset_names and name not in dataset_names:
                    continue
                specs.append(
                    DatasetSpec(
                        name=name,
                        path=DATA_ROOT / "Multiclass" / "multi" / f"{name}.csv",
                        task="classification",
                        target=-1,
                        delimiter=";",
                        is_multiclass=True,
                    )
                )
        elif task == "regression":
            names = REGRESSION_TXT
            for name in names:
                if dataset_names and name not in dataset_names:
                    continue
                specs.append(
                    DatasetSpec(
                        name=name,
                        path=DATA_ROOT / "reg" / f"{name}.txt",
                        task="regression",
                        target=0,
                        delimiter=",",
                        is_multiclass=False,
                    )
                )
            for name, (path_str, target, delimiter) in REGRESSION_CSV.items():
                if dataset_names and name not in dataset_names:
                    continue
                specs.append(
                    DatasetSpec(
                        name=name,
                        path=Path(path_str),
                        task="regression",
                        target=target,
                        delimiter=delimiter,
                        is_multiclass=False,
                    )
                )
        else:
            raise ValueError(f"Unknown task: {task}")

    if limit is not None:
        specs = specs[:limit]

    return specs
