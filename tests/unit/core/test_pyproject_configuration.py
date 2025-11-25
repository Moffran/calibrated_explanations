from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


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
