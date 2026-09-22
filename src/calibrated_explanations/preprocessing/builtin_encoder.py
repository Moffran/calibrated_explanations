"""Deterministic mixed-type encoder with JSON-safe mapping snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..utils.exceptions import NotFittedError, ValidationError


class BuiltinEncoder:
    """A tiny deterministic mixed-type encoder.

    Numeric columns are passed through unchanged. Categorical columns are
    mapped to floats starting at 0. Missing categorical values can either be
    encoded as a deterministic sentinel category or fail fast.
    """

    def __init__(
        self,
        unseen_policy: str = "error",
        *,
        categorical_features: Sequence[int] | None = None,
        missing_value_policy: str = "category",
    ) -> None:
        if unseen_policy not in {"error", "ignore"}:
            raise ValidationError(
                "Invalid unseen policy for BuiltinEncoder.",
                details={"allowed": ["error", "ignore"], "received": unseen_policy},
            )
        if missing_value_policy not in {"category", "error"}:
            raise ValidationError(
                "Invalid missing value policy for BuiltinEncoder.",
                details={"allowed": ["category", "error"], "received": missing_value_policy},
            )
        self.unseen_policy = unseen_policy
        self.missing_value_policy = missing_value_policy
        self._explicit_categorical_features = tuple(
            sorted({int(feature) for feature in (categorical_features or ())})
        )
        self.categorical_features_: tuple[int, ...] = self._explicit_categorical_features
        self.mapping_: Dict[str, List[Any]] | None = None

    def fit(self, x: Any) -> "BuiltinEncoder":
        """Learn per-column category mappings from input data."""
        arr = self._as_2d(x)
        if arr.shape[1] == 0:
            self.mapping_ = {}
            self.categorical_features_ = ()
            return self
        self._validate_explicit_feature_indices(arr.shape[1])
        mapping: Dict[str, List[Any]] = {}
        categorical_features: list[int] = []
        for i, col in enumerate(arr.T):
            if self._should_encode_column(i, col):
                cats = self._sorted_categories(col)
                mapping[f"col_{i}"] = cats
                categorical_features.append(i)
        self.mapping_ = mapping
        self.categorical_features_ = tuple(categorical_features)
        return self

    def fit_transform(self, x: Any) -> Any:
        """Fit the encoder and return transformed values."""
        self.fit(x)
        return self.transform(x)

    def transform(self, x: Any) -> Any:
        """Map input categories to learned indices while preserving numeric columns."""
        arr = self._as_2d(x)
        if self.mapping_ is None:
            raise NotFittedError("Encoder not fitted")
        out = np.empty(arr.shape, dtype=float)
        categorical_features = set(self._resolve_categorical_features())
        for i, col in enumerate(arr.T):
            if i in categorical_features:
                cats = self.mapping_.get(f"col_{i}")
                if cats is None:
                    raise ValidationError(
                        "Missing learned mapping for a categorical column.",
                        details={"column_index": i},
                    )
                lookup = {category: float(position) for position, category in enumerate(cats)}
                sentinel = self._resolve_missing_sentinel(cats)
                for j, value in enumerate(col):
                    if self._is_missing(value):
                        if self.missing_value_policy == "error":
                            raise ValidationError(
                                f"Missing value encountered in categorical column {i}",
                                details={"column_index": i, "row_index": j},
                            )
                        value = sentinel
                    else:
                        value = self._safe_val(value)
                    if value in lookup:
                        out[j, i] = lookup[value]
                    elif self.unseen_policy == "ignore":
                        out[j, i] = -1.0
                    else:
                        raise ValidationError(f"Unseen category {value} in column {i}")
            else:
                out[:, i] = self._numeric_column_to_float(col, column_index=i)
        return out

    def get_mapping_snapshot(self) -> Dict[str, Any] | None:
        """Return a shallow copy of the learned mapping."""
        return dict(self.mapping_) if self.mapping_ is not None else None

    def set_mapping(self, mapping: Dict[str, Any]) -> None:
        """Load a mapping produced by ``get_mapping_snapshot``."""
        if mapping is None:
            self.mapping_ = None
            return
        self.mapping_ = {str(k): list(v) for k, v in mapping.items()}
        if not self.categorical_features_:
            inferred = []
            for key in self.mapping_:
                try:
                    inferred.append(int(key.split("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
            self.categorical_features_ = tuple(sorted(inferred))

    @staticmethod
    def _as_2d(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _resolve_categorical_features(self) -> tuple[int, ...]:
        if self.categorical_features_:
            return self.categorical_features_
        if self.mapping_:
            inferred = []
            for key in self.mapping_:
                try:
                    inferred.append(int(key.split("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
            return tuple(sorted(inferred))
        return ()

    def _validate_explicit_feature_indices(self, n_features: int) -> None:
        if not self._explicit_categorical_features:
            return
        out_of_range = [i for i in self._explicit_categorical_features if i < 0 or i >= n_features]
        if out_of_range:
            raise ValidationError(
                "categorical_features contains an out-of-range column index.",
                details={"column_count": n_features, "out_of_range": out_of_range},
            )

    def _should_encode_column(self, index: int, column: np.ndarray) -> bool:
        if index in self._explicit_categorical_features:
            return True
        return not self._is_numeric_column(column)

    def _is_numeric_column(self, column: np.ndarray) -> bool:
        values = [value for value in column if not self._is_missing(value)]
        if not values:
            return False
        if all(isinstance(value, (bool, np.bool_)) for value in values):
            return False
        try:
            for value in values:
                float(value)
        except (TypeError, ValueError):
            return False
        return True

    def _sorted_categories(self, column: np.ndarray) -> List[Any]:
        values: set[Any] = set()
        sentinel = self._missing_sentinel(column)
        for value in column:
            if self._is_missing(value):
                if self.missing_value_policy == "error":
                    raise ValidationError("Missing value encountered in categorical column")
                values.add(sentinel)
            else:
                values.add(self._safe_val(value))
        return sorted(values, key=self._category_sort_key)

    def _numeric_column_to_float(self, column: np.ndarray, *, column_index: int) -> np.ndarray:
        out = np.empty(len(column), dtype=float)
        for row_index, value in enumerate(column):
            if self._is_missing(value):
                out[row_index] = np.nan
                continue
            try:
                out[row_index] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "Numeric column could not be coerced to float during transform.",
                    details={"column_index": column_index, "row_index": row_index},
                ) from exc
        return out

    def _missing_sentinel(self, column: np.ndarray) -> str:
        sentinel = "__missing__"
        present = {self._safe_val(value) for value in column if not self._is_missing(value)}
        suffix = 1
        while sentinel in present:
            sentinel = f"__missing__{suffix}__"
            suffix += 1
        return sentinel

    @staticmethod
    def _resolve_missing_sentinel(categories: Sequence[Any]) -> str:
        # Fit picks the first of __missing__, __missing__1__, ... that is not a real
        # value, so the sentinel is the last consecutive member of that chain present.
        present = set(categories)
        sentinel = "__missing__"
        suffix = 1
        while f"__missing__{suffix}__" in present:
            sentinel = f"__missing__{suffix}__"
            suffix += 1
        return sentinel

    @staticmethod
    def _category_sort_key(value: Any) -> tuple[str, str]:
        return (type(value).__name__, repr(value))

    @staticmethod
    def _is_missing(value: Any) -> bool:
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _safe_val(v: Any) -> Any:
        # Ensure JSON-safe primitive ordering for snapshot determinism
        if v is None:
            return "__none__"
        if isinstance(v, (int, float, str, bool)):
            return v
        try:
            return str(v)
        except (TypeError, ValueError):
            return repr(v)
