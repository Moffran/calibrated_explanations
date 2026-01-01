"""Shared test fixtures for the calibrated_explanations test suite."""

import numpy as np
import pytest

from tests.helpers.dataset_utils import read_csv_cached


@pytest.fixture
def regression_dataset(sample_limit):
    """Shared regression dataset fixture.

    Produces the tuple used by many regression tests.
    """
    from sklearn.model_selection import train_test_split

    num_to_test = 2
    dataset = "abalone.txt"

    ds = read_csv_cached(f"data/reg/{dataset}")
    max_rows = sample_limit
    x = ds.drop("REGRESSION", axis=1).values[:max_rows, :]
    y = ds["REGRESSION"].values[:max_rows]
    calibration_size = min(1000, max(2, max_rows - num_to_test - 2))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    no_of_features = x.shape[1]
    categorical_features = [i for i in range(no_of_features) if len(np.unique(x[:, i])) < 10]
    columns = ds.drop("REGRESSION", axis=1).columns

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=num_to_test, random_state=42
    )
    x_prop_train, x_cal, y_prop_train, y_cal = train_test_split(
        x_train, y_train, test_size=calibration_size, random_state=42
    )
    return (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        no_of_features,
        categorical_features,
        columns,
    )


@pytest.fixture
def binary_dataset():
    """Shared binary classification dataset fixture."""
    dataset = "diabetes_full"
    delimiter = ","
    num_to_test = 2
    target_column = "Y"

    filename = f"data/{dataset}.csv"
    df = read_csv_cached(filename, delimiter=delimiter, dtype=np.float64)

    columns = df.drop(target_column, axis=1).columns
    num_classes = len(np.unique(df[target_column]))
    num_features = df.drop(target_column, axis=1).shape[1]

    sorted_indices = np.argsort(df[target_column].values).astype(int)
    x, y = (
        df.drop(target_column, axis=1).values[sorted_indices, :],
        df[target_column].values[sorted_indices],
    )

    categorical_features = [
        i
        for i in range(num_features)
        if len(np.unique(df.drop(target_column, axis=1).iloc[:, i])) < 10
    ]

    test_index = np.array(
        [*range(num_to_test // 2), *range(len(y) - 1, len(y) - num_to_test // 2 - 1, -1)]
    )
    train_index = np.setdiff1d(np.array(range(len(y))), test_index)

    trainx_cal, x_test = x[train_index, :], x[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    from sklearn.model_selection import train_test_split

    x_prop_train, x_cal, y_prop_train, y_cal = train_test_split(
        trainx_cal, y_train, test_size=0.33, random_state=42, stratify=y_train
    )

    return (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        num_classes,
        num_features,
        categorical_features,
        columns,
    )


@pytest.fixture
def multiclass_dataset():
    """Shared multiclass dataset fixture."""
    from calibrated_explanations.utils import transform_to_numeric

    dataset_name = "glass"
    delimiter = ","
    num_test_samples = 6
    file_path = f"data/Multiclass/{dataset_name}.csv"

    df = read_csv_cached(file_path, delimiter=delimiter).dropna()
    target_column = "Type"

    df, categorical_features, categorical_labels, target_labels, _ = transform_to_numeric(
        df, target_column
    )

    columns = df.drop(target_column, axis=1).columns
    num_classes = len(np.unique(df[target_column]))
    num_features = df.drop(target_column, axis=1).shape[1]

    sorted_indices = np.argsort(df[target_column].values).astype(int)
    x, y = (
        df.drop(target_column, axis=1).values[sorted_indices, :],
        df[target_column].values[sorted_indices],
    )

    test_indices = np.hstack(
        [np.where(y == i)[0][: num_test_samples // num_classes] for i in range(num_classes)]
    )
    train_indices = np.setdiff1d(np.arange(len(y)), test_indices)

    x_train_cal, x_test = x[train_indices, :], x[test_indices, :]
    y_train, y_test = y[train_indices], y[test_indices]

    from sklearn.model_selection import train_test_split

    x_prop_train, x_cal, y_prop_train, y_cal = train_test_split(
        x_train_cal, y_train, test_size=0.33, random_state=42, stratify=y_train
    )

    return (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        num_classes,
        num_features,
        categorical_features,
        categorical_labels,
        target_labels,
        columns,
    )
