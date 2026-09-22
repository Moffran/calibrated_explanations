"""Unit tests for the deterministic builtin categorical encoder."""

from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.preprocessing.builtin_encoder import BuiltinEncoder
from calibrated_explanations.utils.exceptions import NotFittedError, ValidationError


class BadStringValue:
    """Value whose ``__str__`` fails to exercise repr fallback logic."""

    def __str__(self) -> str:
        raise ValueError("cannot stringify")

    def __repr__(self) -> str:
        return "BadStringValue()"


def test_transform_requires_fit() -> None:
    """Transform should fail when no mapping has been learned."""
    enc = BuiltinEncoder()
    with pytest.raises(NotFittedError, match="not fitted"):
        enc.transform(np.array(["a"], dtype=object))


def test_unseen_category_policy_ignore_sets_negative_one() -> None:
    """Unknown categories should map to -1 when ignore policy is selected."""
    enc = BuiltinEncoder(unseen_policy="ignore")
    enc.fit(np.array(["a", "b"], dtype=object))
    out = enc.transform(np.array(["b", "c"], dtype=object))
    np.testing.assert_allclose(out, np.array([[1.0], [-1.0]]))


def test_unseen_category_policy_error_raises_validation_error() -> None:
    """Unknown categories should raise a structured validation error by default."""
    enc = BuiltinEncoder(unseen_policy="error")
    enc.fit(np.array(["a", "b"], dtype=object))
    with pytest.raises(ValidationError, match="Unseen category"):
        enc.transform(np.array(["c"], dtype=object))


def test_mapping_snapshot_none_and_set_mapping_none_roundtrip() -> None:
    """Snapshot helpers should preserve the unfitted state."""
    enc = BuiltinEncoder()
    assert enc.get_mapping_snapshot() is None
    enc.set_mapping(None)
    assert enc.get_mapping_snapshot() is None


def test_fit_handles_none_and_repr_fallback_when_str_fails() -> None:
    """Fit should normalize None and fallback to repr for unstringifiable values."""
    enc = BuiltinEncoder()
    enc.fit(np.array([None, BadStringValue()], dtype=object))
    mapping = enc.get_mapping_snapshot()
    assert mapping is not None
    assert "__missing__" in mapping["col_0"]
    assert "BadStringValue()" in mapping["col_0"]


def test_fit_should_preserve_numeric_columns_when_input_is_mixed_type() -> None:
    """Mixed arrays should leave numeric columns untouched while encoding categories."""
    enc = BuiltinEncoder()
    transformed = enc.fit_transform(np.array([[1, "red"], [2, "blue"], [3, None]], dtype=object))

    mapping = enc.get_mapping_snapshot()
    assert mapping is not None
    assert "col_0" not in mapping
    assert "col_1" in mapping
    np.testing.assert_allclose(transformed[:, 0], np.array([1.0, 2.0, 3.0]))
    assert transformed.shape == (3, 2)


def test_fit_should_respect_explicit_categorical_features_and_missing_policy() -> None:
    """Forced categorical columns should be encoded and missing values should be handled."""
    enc = BuiltinEncoder(categorical_features=(0,), missing_value_policy="category")
    transformed = enc.fit_transform(np.array([[10], [20], [None]], dtype=object))

    assert enc.categorical_features_ == (0,)
    assert transformed.shape == (3, 1)
    assert transformed[2, 0] >= 0.0


def test_fit_should_raise_for_missing_categorical_values_when_policy_is_error() -> None:
    """Missing categorical values should fail fast when category fallback is disabled."""
    enc = BuiltinEncoder(missing_value_policy="error")

    with pytest.raises(ValidationError, match="Missing value encountered"):
        enc.fit_transform(np.array([["red"], [None]], dtype=object))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unseen_policy": "drop"}, "Invalid unseen policy"),
        ({"missing_value_policy": "impute"}, "Invalid missing value policy"),
    ],
)
def test_init_should_reject_unknown_policies(kwargs, message) -> None:
    """Unsupported policy names should fail at construction with the allowed values."""
    with pytest.raises(ValidationError, match=message) as excinfo:
        BuiltinEncoder(**kwargs)

    assert excinfo.value.details["received"] == next(iter(kwargs.values()))


def test_fit_should_learn_empty_mapping_when_input_has_no_columns() -> None:
    """A zero-column input should fit to an empty mapping instead of failing."""
    enc = BuiltinEncoder(categorical_features=(0,))

    enc.fit(np.empty((3, 0), dtype=object))

    assert enc.get_mapping_snapshot() == {}
    assert enc.categorical_features_ == ()


def test_fit_should_reject_out_of_range_explicit_categorical_feature() -> None:
    """Explicit categorical indices beyond the column count should fail fast."""
    enc = BuiltinEncoder(categorical_features=(0, 5))

    with pytest.raises(ValidationError, match="out-of-range") as excinfo:
        enc.fit(np.array([[1, "a"], [2, "b"]], dtype=object))

    assert excinfo.value.details == {"column_count": 2, "out_of_range": [5]}


def test_fit_should_encode_boolean_and_all_missing_columns_as_categorical() -> None:
    """Boolean and all-missing columns are categorical even though floats accept them."""
    enc = BuiltinEncoder()
    data = np.array([[True, None, 1.5], [False, None, 2.5]], dtype=object)

    transformed = enc.fit_transform(data)

    assert enc.categorical_features_ == (0, 1)
    assert enc.get_mapping_snapshot() == {"col_0": [False, True], "col_1": ["__missing__"]}
    np.testing.assert_allclose(transformed, np.array([[1.0, 0.0, 1.5], [0.0, 0.0, 2.5]]))


def test_transform_should_keep_missing_numeric_values_as_nan() -> None:
    """Missing values in a numeric column pass through as NaN, not as a category."""
    enc = BuiltinEncoder()

    transformed = enc.fit_transform(np.array([[1.0], [None], [3.0]], dtype=object))

    assert enc.categorical_features_ == ()
    np.testing.assert_array_equal(np.isnan(transformed[:, 0]), [False, True, False])


def test_transform_should_reject_non_numeric_value_in_numeric_column() -> None:
    """A string arriving in a column fitted as numeric should raise with its position."""
    enc = BuiltinEncoder()
    enc.fit(np.array([[1.0], [2.0]], dtype=object))

    with pytest.raises(ValidationError, match="could not be coerced") as excinfo:
        enc.transform(np.array([[1.0], ["high"]], dtype=object))

    assert excinfo.value.details == {"column_index": 0, "row_index": 1}


def test_transform_should_raise_for_missing_value_when_policy_is_error() -> None:
    """Missing categorical values seen only at transform time also fail under 'error'."""
    enc = BuiltinEncoder(missing_value_policy="error")
    enc.fit(np.array([["red"], ["blue"]], dtype=object))

    with pytest.raises(ValidationError, match="column 0") as excinfo:
        enc.transform(np.array([["red"], [None]], dtype=object))

    assert excinfo.value.details == {"column_index": 0, "row_index": 1}


def test_missing_sentinel_should_not_collide_with_literal_missing_category() -> None:
    """A literal '__missing__' value keeps its own code, distinct from real missing values."""
    enc = BuiltinEncoder()
    data = np.array([["__missing__"], [None], ["x"]], dtype=object)

    transformed = enc.fit_transform(data)

    categories = enc.get_mapping_snapshot()["col_0"]
    assert "__missing__1__" in categories
    assert len(set(transformed[:, 0])) == 3


def test_set_mapping_should_infer_categorical_columns_from_legacy_snapshot() -> None:
    """Snapshots without categorical_features infer them from 'col_N' keys only."""
    enc = BuiltinEncoder()

    enc.set_mapping({"col_1": ["a", "b"], "legacy": ["ignored"], "col_x": ["ignored"]})
    transformed = enc.transform(np.array([[7.0, "b"], [8.0, "a"]], dtype=object))

    assert enc.categorical_features_ == (1,)
    np.testing.assert_allclose(transformed, np.array([[7.0, 1.0], [8.0, 0.0]]))


def test_transform_should_raise_when_categorical_column_has_no_learned_mapping() -> None:
    """A declared categorical column missing from the loaded mapping should fail clearly."""
    enc = BuiltinEncoder(categorical_features=(0, 1))
    enc.set_mapping({"col_0": ["a"]})

    with pytest.raises(ValidationError, match="Missing learned mapping") as excinfo:
        enc.transform(np.array([["a", "b"]], dtype=object))

    assert excinfo.value.details == {"column_index": 1}
