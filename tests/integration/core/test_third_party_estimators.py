"""Skippable integration tests for third-party gradient boosting libraries.

These smoke tests verify that CalibratedExplainer interoperates with popular
scikit-learn compatible learners: XGBoost, LightGBM, and CatBoost.

Each test is skipped automatically if the corresponding library is not
installed in the environment. Tests are also marked as slow so they are
skipped when FAST_TESTS is enabled.
"""

from __future__ import annotations

import numpy as np
import pytest
import warnings
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


# Tests run with `error::FutureWarning` in `pytest.ini` which makes
# scikit-learn's transitory FutureWarning (rename of `force_all_finite`
# to `ensure_all_finite`) fail the suite when LightGBM triggers it.
# Narrowly ignore that specific FutureWarning here so the integration
# smoke test can run in environments with older/newer sklearn.
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
    category=FutureWarning,
)

# Apply a pytest collection-time filter to ensure pytest's global
# `error::FutureWarning` doesn't convert this specific message into
# a failing error for these integration smoke tests.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:'force_all_finite' was renamed to 'ensure_all_finite':FutureWarning"
    )
]


def _tiny_binary_dataset(n_samples: int = 120, n_features: int = 8, random_state: int = 42):
    x_data, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 3),
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=random_state,
    )
    # Split into proper-train, calibration, test
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x_data, y, test_size=0.2, random_state=random_state
    )
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_tmp, y_tmp, test_size=0.25, random_state=random_state
    )  # 0.25 x 0.8 = 0.2
    return x_train, y_train, x_cal, y_cal, x_test, y_test


@pytest.mark.slow
def test_xgboost_classifier_basic_integration():
    try:
        import xgboost as xgb
    except Exception as exc:  # pragma: no cover - environment-specific
        pytest.skip(f"xgboost unavailable or misconfigured: {exc}")
    x_train, y_train, x_cal, y_cal, x_test, _ = _tiny_binary_dataset()

    # Use lightweight params for speed; rely on sklearn-compatible wrapper
    clf = xgb.XGBClassifier(
        n_estimators=15,
        max_depth=2,
        subsample=0.9,
        colsample_bytree=0.9,
        learning_rate=0.2,
        tree_method="hist",
        verbosity=0,
        random_state=42,
    )
    clf.fit(x_train, y_train)

    ce = CalibratedExplainer(clf, x_cal, y_cal, mode="classification")

    # Uncalibrated predictions (from the learner)
    proba_uncal = ce.predict_proba(x_test, calibrated=False)
    assert proba_uncal.shape[0] == x_test.shape[0]
    assert proba_uncal.shape[1] == 2

    # Calibrated predictions with intervals
    proba_cal, (low, high) = ce.predict_proba(x_test, uq_interval=True)
    assert proba_cal.shape == proba_uncal.shape
    assert len(low) == len(high) == x_test.shape[0]
    assert np.all(np.isfinite(proba_cal))

    y_hat, (low_y, high_y) = ce.predict(x_test, uq_interval=True)
    assert len(y_hat) == x_test.shape[0]
    assert len(low_y) == len(high_y) == x_test.shape[0]


@pytest.mark.slow
def test_lightgbm_classifier_basic_integration():
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - environment-specific
        pytest.skip(f"lightgbm unavailable or misconfigured: {exc}")
    x_train, y_train, x_cal, y_cal, x_test, _ = _tiny_binary_dataset()

    clf = lgb.LGBMClassifier(
        n_estimators=25,
        max_depth=-1,
        learning_rate=0.2,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
        force_col_wise=True,
    )
    clf.fit(x_train, y_train)

    ce = CalibratedExplainer(clf, x_cal, y_cal, mode="classification")

    proba_uncal = ce.predict_proba(x_test, calibrated=False)
    assert proba_uncal.shape[0] == x_test.shape[0]
    assert proba_uncal.shape[1] == 2

    proba_cal, (low, high) = ce.predict_proba(x_test, uq_interval=True)
    assert proba_cal.shape == proba_uncal.shape
    assert len(low) == len(high) == x_test.shape[0]
    assert np.all(np.isfinite(proba_cal))

    y_hat, (low_y, high_y) = ce.predict(x_test, uq_interval=True)
    assert len(y_hat) == x_test.shape[0]
    assert len(low_y) == len(high_y) == x_test.shape[0]


@pytest.mark.slow
def test_catboost_classifier_basic_integration():
    try:
        import catboost as cb
    except Exception as exc:  # pragma: no cover - environment-specific
        pytest.skip(f"catboost unavailable or misconfigured: {exc}")
    x_train, y_train, x_cal, y_cal, x_test, _ = _tiny_binary_dataset()

    clf = cb.CatBoostClassifier(
        iterations=40,
        depth=4,
        learning_rate=0.2,
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
    )
    clf.fit(x_train, y_train)

    ce = CalibratedExplainer(clf, x_cal, y_cal, mode="classification")

    proba_uncal = ce.predict_proba(x_test, calibrated=False)
    assert proba_uncal.shape[0] == x_test.shape[0]
    assert proba_uncal.shape[1] == 2

    proba_cal, (low, high) = ce.predict_proba(x_test, uq_interval=True)
    assert proba_cal.shape == proba_uncal.shape
    assert len(low) == len(high) == x_test.shape[0]
    assert np.all(np.isfinite(proba_cal))

    y_hat, (low_y, high_y) = ce.predict(x_test, uq_interval=True)
    assert len(y_hat) == x_test.shape[0]
    assert len(low_y) == len(high_y) == x_test.shape[0]
