"""Memory-oriented tests for reject wrappers."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectCalibratedExplanations


def test_calibration_set_reused_not_duplicated():
    rng_x = np.random.RandomState(0)
    rng_y = np.random.RandomState(1)
    x = rng_x.randn(2000, 20)
    y = rng_y.randint(0, 2, size=len(x))

    model = RandomForestClassifier(n_estimators=10, random_state=0)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x[:1200], y[:1200])
    wrapper.calibrate(x[1200:], y[1200:], seed=0)

    base = wrapper.explain_factual(x[1200:1210], reject_policy=RejectPolicy.FLAG)
    wrapped = RejectCalibratedExplanations.from_collection(
        base,
        base.metadata_full(),
        RejectPolicy.FLAG,
        base.rejected,
    )

    assert wrapped.calibrated_explainer is base.calibrated_explainer
    assert id(wrapped.calibrated_explainer.x_cal) == id(base.calibrated_explainer.x_cal) or np.shares_memory(
        wrapped.calibrated_explainer.x_cal,
        base.calibrated_explainer.x_cal,
    )

    for name in ("x_cal", "_X_cal", "scaled_x_cal", "fast_x_cal", "scaled_y_cal"):
        assert not hasattr(wrapped, name)
