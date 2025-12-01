from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrated_explanations.core import NotFittedError, ValidationError
from calibrated_explanations.utils import helper


class HelperExample:
    """Module-level class so safe_isinstance can resolve by dotted path."""

    pass


def test_make_directory_creates_expected_structure(tmp_path, monkeypatch):
    base = tmp_path / "plots"
    monkeypatch.chdir(tmp_path)

    # Empty save_ext short circuits
    helper.make_directory("ignored", save_ext=[], add_plots_folder=False)

    # Create directory without plots folder
    plain_dir = tmp_path / "plain"
    helper.make_directory(str(plain_dir), add_plots_folder=False)
    assert plain_dir.exists()

    # When add_plots_folder=True we create plots/<path>
    helper.make_directory("subdir", add_plots_folder=True)
    assert (base / "subdir").is_dir()

    # Path already names plots should not duplicate folders
    helper.make_directory("plots", add_plots_folder=True)
    assert base.is_dir()


def test_safe_isinstance_variants(monkeypatch):
    from calibrated_explanations.core import ValidationError
    module_name = HelperExample.__module__
    assert helper.safe_isinstance(HelperExample(), f"{module_name}.HelperExample") is True

    # class path supplied as list with missing module falls back to False
    assert helper.safe_isinstance(HelperExample(), ["nonexistent_mod.Class"]) is False

    # Module present but class missing returns False
    assert helper.safe_isinstance(HelperExample(), f"{module_name}.Missing") is False

    # `None` input produces False without errors
    assert helper.safe_isinstance(HelperExample(), None) is False

    # Missing dot raises ValidationError
    with pytest.raises(ValidationError):
        helper.safe_isinstance(HelperExample(), "notadottedpath")


def test_safe_import_success_and_failures(monkeypatch):
    module = helper.safe_import("math")
    assert module is math

    # Import list of names
    funcs = helper.safe_import("math", ["sin", "cos"])
    assert [f(0) for f in funcs] == [0.0, 1.0]

    # Unknown module bubbles informative ImportError
    with pytest.raises(ImportError) as excinfo:
        helper.safe_import("surely_missing_module")
    assert "required module" in str(excinfo.value)

    # Existing module but missing attribute raises ImportError
    with pytest.raises(ImportError) as excinfo:
        helper.safe_import("math", "not_there")
    assert "does not exist" in str(excinfo.value)


def test_check_is_fitted_paths(tmp_path):
    class DummyFitted:
        fitted = True

    class DummyIsFitted:
        def is_fitted(self):
            return False

    class DummySklearn:
        def fit(self, *_args, **_kwargs):
            return self

        def __sklearn_is_fitted__(self):
            return True

    class DummyAttr:
        coef_ = np.array([1])

        def fit(self):
            return self

    class DummyNeedsFit:
        def fit(self):
            self.some_attr_ = True

    assert helper.check_is_fitted(DummyFitted()) is True
    assert helper.check_is_fitted(DummyIsFitted()) is False
    assert helper.check_is_fitted(DummySklearn()) is None
    assert helper.check_is_fitted(DummyAttr(), attributes=["coef_"]) is None

    estimator = DummyNeedsFit()
    with pytest.raises(NotFittedError) as excinfo:
        helper.check_is_fitted(estimator)
    assert "DummyNeedsFit" in str(excinfo.value)

    class _BadEstimator:
        pass

    with pytest.raises(TypeError):
        helper.check_is_fitted(_BadEstimator())

    with pytest.raises(TypeError):
        helper.check_is_fitted(DummyNeedsFit)


def test_is_notebook_returns_false_in_pytest_environment():
    assert helper.is_notebook() is False


def test_transform_to_numeric_with_and_without_mappings():
    df = pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, 6],
            "cat": ["a", "b", "c", "d", "e", "f"],
            "target": ["yes", "no", "yes", "no", "yes", "no"],
        }
    )
    transformed, categorical_features, categorical_labels, target_labels, mappings = (
        helper.transform_to_numeric(df.copy(), "target")
    )

    assert categorical_features == [1]
    assert set(target_labels.values()) == {"no", "yes"}
    assert transformed["cat"].dtype.kind in {"i", "u"}

    # Reuse mappings to ensure deterministic mapping and branch reuse
    df2 = pd.DataFrame({"num": [7, 8], "cat": ["d", "e"], "target": ["no", "yes"]})
    reused, categorical_features2, categorical_labels2, target_labels2, mappings2 = (
        helper.transform_to_numeric(df2, "target", mappings)
    )
    assert mappings2 == mappings
    assert categorical_features2 == [1]
    assert categorical_labels2[1] == mappings["cat"]
    assert target_labels2 == mappings["target"]
    assert reused["cat"].tolist() == [mappings["cat"]["d"], mappings["cat"]["e"]]


def test_assert_threshold_handles_nested_structures():
    from calibrated_explanations.core import ValidationError
    assert helper.assert_threshold(0.5, [1, 2, 3]) == 0.5
    assert helper.assert_threshold((0.2, 0.8), [1, 2]) == (0.2, 0.8)
    assert helper.assert_threshold([0.1, 0.2], np.array([[1], [2]])) == [0.1, 0.2]

    thresholds = [(0.1, 0.9), (0.2, 0.8)]
    assert helper.assert_threshold(thresholds, np.array([[1], [2]])) == thresholds

    with pytest.raises(ValidationError):
        helper.assert_threshold((0.1,), [1])
    with pytest.raises(AssertionError):
        helper.assert_threshold([0.1, 0.2, 0.3], [1, 2])
    with pytest.raises(ValidationError):
        helper.assert_threshold({1: 2}, [1])


def test_calculate_metrics_behaviour():
    from calibrated_explanations.core import ValidationError
    assert helper.calculate_metrics() == ["ensured"]

    uncertainty = [0.2, 0.8]
    prediction = [0.7, 0.3]
    ensured = helper.calculate_metrics(uncertainty, prediction, w=0.25, metric="ensured")
    expected_ensured = (1 - 0.25) * (1 - np.asarray(uncertainty)) + 0.25 * np.asarray(prediction)
    assert np.allclose(ensured, expected_ensured)

    # Normalize branch and negative weights
    normalized = helper.calculate_metrics(
        uncertainty,
        prediction,
        w=-0.5,
        metric=["ensured"],
        normalize=True,
    )
    expected_normalized = np.zeros_like(expected_ensured)
    assert np.allclose(normalized, expected_normalized)

    with pytest.raises(ValidationError):
        helper.calculate_metrics(uncertainty, None)

    with pytest.raises(ValidationError):
        helper.calculate_metrics(uncertainty, prediction, w=2)


def test_convert_targets_to_numeric_handles_strings_and_numeric():
    numeric, mapping = helper.convert_targets_to_numeric(np.array(["cat", "dog"]))
    assert list(numeric) == [0, 1]
    assert mapping == {"cat": 0, "dog": 1}

    arr, mapping_none = helper.convert_targets_to_numeric(np.array([1, 2, 3]))
    assert mapping_none is None
    assert list(arr) == [1, 2, 3]


def test_concatenate_thresholds_handles_tuples_and_lists():
    perturbed = []
    tuples = [(0.1, 0.9), (0.2, 0.8)]
    result = helper.concatenate_thresholds(perturbed, tuples, np.array([1, 0]))
    assert result == [(0.2, 0.8), (0.1, 0.9)]

    perturbed_array = np.array([0.5])
    appended = helper.concatenate_thresholds(perturbed_array, [0.7, 0.8], np.array([0]))
    assert np.allclose(appended, np.array([0.5, 0.7]))


def test_immutable_array_and_prepare_for_saving(tmp_path, monkeypatch):
    arr = helper.immutable_array([1, 2, 3])
    assert not arr.flags.writeable
    assert arr[1] == 2

    with pytest.raises(ValueError):
        arr[0] = 10

    monkeypatch.chdir(tmp_path)
    path, filename, title, ext = helper.prepare_for_saving("plots/example.png")
    assert Path(path).is_dir()
    assert filename == "example.png"
    assert title == "example"
    assert ext == ".png"

    empty = helper.prepare_for_saving("")
    assert empty == ("", "", "", "")


def test_safe_mean_and_first_element_edge_cases():
    assert helper.safe_mean([], default=3.14) == 3.14
    assert helper.safe_first_element([], default=2.5) == 2.5

    assert helper.safe_first_element(np.array([10])) == 10.0
    assert helper.safe_first_element(np.array([[1, 2], [3, 4]]), col=1) == 2.0
    assert helper.safe_first_element(np.array([[1, 2]]), col=5) == 0.0
    assert helper.safe_first_element(5.5) == 5.5

    # Unhandled objects fall back to default
    class Weird:
        def __array__(self):
            raise ValueError

    assert helper.safe_mean([Weird()], default=1.23) == 1.23
    assert helper.safe_first_element(Weird(), default=4.56) == 4.56
