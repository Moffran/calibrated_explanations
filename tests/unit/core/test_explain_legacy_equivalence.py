"""Regression tests ensuring the optimised explain path matches the legacy implementation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations.core.explain._legacy_explain import explain as legacy_explain
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


def _split(x: np.ndarray, y: np.ndarray, *, calibration: int, test: int):
    x_train = x[: -(calibration + test)]
    y_train = y[: -(calibration + test)]
    x_cal = x[-(calibration + test) : -test]
    y_cal = y[-(calibration + test) : -test]
    x_test = x[-test:]
    y_test = y[-test:]
    return x_train, y_train, x_cal, y_cal, x_test, y_test


def _assert_collection_equal(modern, legacy):
    assert len(modern.explanations) == len(legacy.explanations)
    for modern_exp, legacy_exp in zip(modern.explanations, legacy.explanations):
        for key in ("predict", "low", "high"):
            m_weights = np.asarray(modern_exp.feature_weights.get(key, []), dtype=float)
            l_weights = np.asarray(legacy_exp.feature_weights.get(key, []), dtype=float)
            np.testing.assert_allclose(m_weights, l_weights, rtol=1e-7, atol=1e-8)

            m_predict = np.asarray(modern_exp.feature_predict.get(key, []), dtype=float)
            l_predict = np.asarray(legacy_exp.feature_predict.get(key, []), dtype=float)
            np.testing.assert_allclose(m_predict, l_predict, rtol=1e-7, atol=1e-8)

        for key in ("predict", "low", "high", "counts", "fractions"):
            m_dict = modern_exp.binned[key]
            l_dict = legacy_exp.binned[key]
            assert set(m_dict.keys()) == set(l_dict.keys())
            for feat in m_dict:
                np.testing.assert_allclose(
                    np.asarray(m_dict[feat], dtype=float),
                    np.asarray(l_dict[feat], dtype=float),
                    rtol=1e-7,
                    atol=1e-8,
                )

        m_bins = modern_exp.binned["current_bin"]
        l_bins = legacy_exp.binned["current_bin"]
        assert set(m_bins.keys()) == set(l_bins.keys())
        for feat in m_bins:
            assert int(m_bins[feat]) == int(l_bins[feat])

        assert modern_exp.bin == legacy_exp.bin
        assert modern_exp.conditions == legacy_exp.conditions


def test_classification_matches_legacy():
    x, y = make_classification(
        n_samples=600,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=1,
    )
    x_train, y_train, x_cal, y_cal, x_test, _ = _split(x, y, calibration=150, test=60)
    model = RandomForestClassifier(n_estimators=20, random_state=1)
    model.fit(x_train, y_train)

    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        mode="classification",
        feature_names=[f"f{i}" for i in range(x.shape[1])],
        categorical_features=[],
        suppress_crepes_errors=True,
    )
    explainer.set_discretizer(None)

    subset = x_test[:10]
    modern = explainer.explain(subset, _use_plugin=False)
    legacy = legacy_explain(explainer, subset)

    _assert_collection_equal(modern, legacy)


def test_regression_matches_legacy():
    x, y = make_regression(
        n_samples=600,
        n_features=4,
        n_informative=4,
        noise=0.1,
        random_state=5,
    )
    x_train, y_train, x_cal, y_cal, x_test, _ = _split(x, y, calibration=150, test=60)
    model = RandomForestRegressor(n_estimators=30, random_state=5)
    model.fit(x_train, y_train)

    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        mode="regression",
        feature_names=[f"r{i}" for i in range(x.shape[1])],
        categorical_features=[],
        suppress_crepes_errors=True,
    )
    explainer.set_discretizer(None)

    subset = x_test[:10]
    modern = explainer.explain(subset, _use_plugin=False)
    legacy = legacy_explain(explainer, subset)

    _assert_collection_equal(modern, legacy)


def test_legacy_explain_categorical_paths_and_ignore():
    """Exercise categorical feature branches and feature ignore handling."""

    x_train = np.array(
        [
            [0, 5, -1.0, 0.1],
            [1, 5, -0.5, 0.2],
            [0, 5, 0.0, 0.3],
            [1, 5, 0.5, 0.4],
            [0, 5, 1.0, 0.5],
            [1, 5, 1.5, 0.6],
            [0, 5, -1.5, 0.7],
            [1, 5, 0.2, 0.8],
        ],
        dtype=float,
    )
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    x_cal = np.array(
        [
            [0, 5, -1.2, 0.15],
            [1, 5, 0.2, 0.25],
            [0, 5, 0.7, 0.35],
            [1, 5, 1.2, 0.45],
        ],
        dtype=float,
    )
    y_cal = np.array([0, 1, 0, 1])

    x_test = np.array(
        [
            [1, 5, -0.1, 0.05],
            [0, 5, 0.9, 0.55],
        ],
        dtype=float,
    )

    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(x_train, y_train)

    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        mode="classification",
        feature_names=["cat_toggle", "cat_constant", "cont_signal", "cont_extra"],
        categorical_features=[0, 1],
        suppress_crepes_errors=True,
    )
    explainer.set_discretizer(None)

    legacy = legacy_explain(explainer, x_test, features_to_ignore=[3])

    # The categorical feature (index 0) should contribute a full distribution over
    # its categories that sums to one and matches the calibration categories.
    explanation = legacy.explanations[0]
    fractions = explanation.binned["fractions"][0]
    np.testing.assert_allclose(fractions.sum(), 1.0)
    assert fractions.size > 0

    # The constant categorical feature (index 1) has no uncovered bins, so the
    # legacy path assigns zero contribution weight to it.
    np.testing.assert_allclose(explanation.feature_weights["predict"][1], 0.0)

    # Features marked to be ignored should inherit the baseline prediction.
    assert explanation.binned["predict"][3] == pytest.approx(legacy.predictions[0])


def test_legacy_explain_accepts_threshold_tuples_for_regression():
    """Ensure tuple thresholds are propagated through the legacy explain path."""

    x_train = np.array([[0.0], [0.5], [1.0], [1.5], [2.0], [2.5]])
    y_train = 2 * x_train[:, 0]

    x_cal = np.array([[1.0], [1.0], [1.0], [1.0]])
    y_cal = 2 * x_cal[:, 0]

    x_test = np.array([[1.0], [1.0]])

    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(x_train, y_train)

    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        mode="regression",
        feature_names=["signal"],
        categorical_features=[],
        suppress_crepes_errors=True,
    )
    explainer.set_discretizer(None)
    explainer.features_to_ignore = np.array([], dtype=int)
    explainer.discretizer.names = {0: {0: "constant", 1: "constant"}}

    thresholds = [(0.9, 1.1), (0.9, 1.1)]
    legacy = legacy_explain(explainer, x_test, threshold=thresholds)

    # The legacy implementation should preserve tuple thresholds exactly.
    assert legacy.y_threshold == thresholds

    # With no uncovered perturbations the per-feature weights collapse to zero
    # and the associated perturbation fractions are empty arrays.
    explanation = legacy.explanations[0]
    np.testing.assert_allclose(explanation.feature_weights["predict"][0], 0.0)
    assert explanation.binned["fractions"][0].size == 0


def test_legacy_explain_handles_continuous_bins_and_boundaries():
    """Cover the legacy perturbation logic for continuous features and bins."""

    x_train = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -0.5],
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
        ]
    )
    y_train = x_train[:, 0] - 0.5 * x_train[:, 1]

    x_cal = np.array(
        [
            [-2.5, -1.2],
            [-1.0, -0.4],
            [0.0, 0.2],
            [1.0, 0.6],
            [2.5, 1.4],
        ]
    )
    y_cal = x_cal[:, 0] - 0.5 * x_cal[:, 1]

    x_test = np.array(
        [
            [-0.2, -0.1],
            [1.8, 0.9],
        ]
    )

    model = RandomForestRegressor(n_estimators=40, random_state=1)
    model.fit(x_train, y_train)

    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        mode="regression",
        feature_names=["signal", "offset"],
        categorical_features=[],
        suppress_crepes_errors=True,
    )
    explainer.set_discretizer(None)

    bins = np.array([0, 1])
    thresholds = [(0.0, 0.2), (0.5, 0.7)]
    legacy = legacy_explain(explainer, x_test, threshold=thresholds, bins=bins)

    # Both continuous features should create multiple perturbed bins informed by
    # lesser/greater boundary searches.
    explanation = legacy.explanations[0]
    counts_feature0 = explanation.binned["counts"][0]
    counts_feature1 = explanation.binned["counts"][1]
    assert counts_feature0.size == 2
    assert counts_feature1.size == 2
    assert counts_feature0[0] > 0 and counts_feature0[1] >= 0
    assert counts_feature1[0] > 0 and counts_feature1[1] >= 0

    # Bins propagate through to the final explanation even when the underlying
    # prediction collapses to the degraded fallback.
    np.testing.assert_allclose(legacy.predictions, np.zeros(len(x_test)))
    np.testing.assert_allclose(explanation.binned["fractions"][0].sum(), 1.0)
    assert legacy.y_threshold == thresholds
