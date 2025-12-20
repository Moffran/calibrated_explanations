"""Dataset helpers reused across calibration tests."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Simple in-memory CSV read cache to avoid repeated expensive disk reads during tests.
# Keys are (path, sorted kwargs repr) so callers can pass delimiter/dtype when needed.
_CSV_CACHE = {}


def read_csv_cached(path: str, **kwargs) -> pd.DataFrame:
    """Read CSV with simple caching. Returns a new DataFrame copy for safety.

    The cache key stringifies kwargs to keep keys hashable. Returns a copy so
    tests can mutate the returned DataFrame safely without affecting the cache.
    """
    key = (path, tuple((k, repr(v)) for k, v in sorted(kwargs.items())))
    if key not in _CSV_CACHE:
        _CSV_CACHE[key] = pd.read_csv(path, **kwargs)
    # Return a copy to make sure tests don't mutate the cached object
    return _CSV_CACHE[key].copy()


def make_binary_dataset():
    """Return the diabetes slice used for parity tests with train/cal/test splits."""
    dataset = "diabetes_full"
    # Assuming data is in the root data/ folder relative to execution context
    # Adjust path if necessary or make it robust
    try:
        df = pd.read_csv(f"data/{dataset}.csv", dtype=np.float64)
    except FileNotFoundError:
        # Fallback for when running from tests/ directory
        df = pd.read_csv(f"../data/{dataset}.csv", dtype=np.float64)

    df = df.iloc[:500, :]
    target = "Y"
    x, y = df.drop(target, axis=1), df[target]
    no_of_features = x.shape[1]
    columns = x.columns
    categorical_features = [i for i in range(no_of_features) if len(np.unique(x.iloc[:, i])) < 10]
    idx = np.argsort(y.values).astype(int)
    x, y = x.values[idx, :], y.values[idx]
    num_to_test = 2
    test_index = np.array(
        [*range(num_to_test // 2), *range(len(y) - 1, len(y) - num_to_test // 2 - 1, -1)]
    )
    train_index = np.setdiff1d(np.array(range(len(y))), test_index)
    trainx_cal, x_test = x[train_index, :], x[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
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
        None,
        no_of_features,
        categorical_features,
        columns,
    )


def make_regression_dataset():
    """Return the housing slice used for parity tests with train/cal/test splits."""
    dataset = "housing"
    try:
        df = pd.read_csv(f"data/reg/{dataset}.csv", sep=";")
    except FileNotFoundError:
        df = pd.read_csv(f"../data/reg/{dataset}.csv", sep=";")

    df = df.dropna()
    df = df.iloc[:500, :]
    target = "median_house_value"
    x, y = df.drop(target, axis=1), df[target]
    no_of_features = x.shape[1]
    columns = x.columns
    categorical_features = [i for i in range(no_of_features) if len(np.unique(x.iloc[:, i])) < 10]
    
    idx = np.argsort(y.values).astype(int)
    x, y = x.values[idx, :], y.values[idx]
    
    num_to_test = 2
    test_index = np.array(
        [*range(num_to_test // 2), *range(len(y) - 1, len(y) - num_to_test // 2 - 1, -1)]
    )
    train_index = np.setdiff1d(np.array(range(len(y))), test_index)
    trainx_cal, x_test = x[train_index, :], x[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    
    x_prop_train, x_cal, y_prop_train, y_cal = train_test_split(
        trainx_cal, y_train, test_size=0.33, random_state=42
    )
    return (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        None,
        no_of_features,
        categorical_features,
        columns,
    )
