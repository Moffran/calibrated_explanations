"""Regression tests ensuring the optimised explain path matches the legacy implementation."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations.core._legacy_explain import explain as legacy_explain
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
