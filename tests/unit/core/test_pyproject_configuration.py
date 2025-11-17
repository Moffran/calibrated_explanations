from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from calibrated_explanations.core import calibrated_explainer as explainer_module
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.plugins.registry import ensure_builtin_plugins


def _make_simple_model():
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression(random_state=0)
    model.fit(x, y)
    return model, x, y



