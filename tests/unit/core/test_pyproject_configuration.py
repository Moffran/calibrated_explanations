from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def make_simple_model_helper():
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression(random_state=0, solver="liblinear")
    model.fit(x, y)
    return model, x, y
