"""Probabilistic regression accepts DataFrame inputs without a difficulty estimator.

Regression guard: the split CPS behind thresholded regression is normalized, so
the constant difficulty vector must have one entry per instance even when the
test input is a pandas DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from calibrated_explanations import WrapCalibratedExplainer


@pytest.fixture(scope="module")
def wrapper_and_test():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    y = 3 * x["a"].to_numpy() + rng.normal(scale=0.3, size=300)
    wrapper = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=20, random_state=0))
    wrapper.fit(x[:150], y[:150])
    wrapper.calibrate(x[150:280], y[150:280])
    return wrapper, x[280:]


@pytest.mark.parametrize("threshold", [0.0, (-1.0, 1.0)])
def test_predict_proba_dataframe_matches_ndarray(wrapper_and_test, threshold):
    wrapper, x_test = wrapper_and_test

    proba_df, (low_df, high_df) = wrapper.predict_proba(
        x_test, uq_interval=True, threshold=threshold
    )
    proba_np, (low_np, high_np) = wrapper.predict_proba(
        x_test.to_numpy(), uq_interval=True, threshold=threshold
    )

    assert proba_df.shape == (len(x_test), 2)
    np.testing.assert_array_equal(proba_df, proba_np)
    np.testing.assert_array_equal(low_df, low_np)
    np.testing.assert_array_equal(high_df, high_np)


def test_predict_with_threshold_accepts_dataframe(wrapper_and_test):
    wrapper, x_test = wrapper_and_test

    pred_df, (low_df, high_df) = wrapper.predict(x_test, uq_interval=True, threshold=0.0)
    pred_np, _ = wrapper.predict(x_test.to_numpy(), uq_interval=True, threshold=0.0)

    assert len(pred_df) == len(x_test)
    np.testing.assert_array_equal(pred_df, pred_np)
