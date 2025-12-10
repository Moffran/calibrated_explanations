"""Minimal model helpers used throughout tests."""

import os
from typing import Optional, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class DummyLearner:
    """Minimal learner implementation for calibration-focused tests."""

    def __init__(
        self,
        *,
        mode: str = "classification",
        oob_decision_function: Optional[np.ndarray] = None,
        oob_prediction: Optional[np.ndarray] = None,
    ) -> None:
        self.mode = mode
        self.fitted_ = True  # ensures check_is_fitted succeeds
        self.oob_decision_function_ = oob_decision_function
        self.oob_prediction_ = oob_prediction

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DummyLearner":  # pragma: no cover - unused
        """Return self without modifying the learner."""
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return zeros for every provided sample."""
        return np.zeros(len(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return constant probabilities that sum to one for classification."""
        x = np.atleast_2d(x)
        if self.mode == "classification":
            probs = np.zeros((len(x), 2))
            probs[:, 0] = 0.4
            probs[:, 1] = 0.6
            return probs
        return np.zeros((len(x), 1))


class DummyIntervalLearner:
    """Interval learner returning deterministic zero arrays."""

    def predict_uncertainty(
        self, x: np.ndarray, *_args: Any, **_kwargs: Any
    ) -> tuple[np.ndarray, ...]:
        """Return zero uncertainty bands for every input."""
        n = np.atleast_2d(x).shape[0]
        zeros = np.zeros(n)
        return zeros, zeros, zeros, None

    def predict_probability(
        self, x: np.ndarray, *_args: Any, **_kwargs: Any
    ) -> tuple[np.ndarray, ...]:
        """Return zero probability bands while remaining API-compatible."""
        n = np.atleast_2d(x).shape[0]
        zeros = np.zeros(n)
        return zeros, zeros, zeros, None

    def predict_proba(self, x: np.ndarray, *_args: Any, **_kwargs: Any) -> tuple[np.ndarray, ...]:
        """Compatibility shim: some code paths call predict_proba.

        Return three zero arrays (predict, low, high) similar to other helpers.
        """
        x = np.atleast_2d(x)
        n = x.shape[0]
        probs = np.zeros((n, 2))
        probs[:, 0] = 0.4
        probs[:, 1] = 0.6
        low = np.zeros((n, 2))
        high = np.zeros((n, 2))
        return probs, low, high


def get_classification_model(model_name, x_prop_train, y_prop_train):
    """Return a fitted classification model (RF or DT)."""
    fast = bool(os.getenv("FAST_TESTS"))
    t1 = DecisionTreeClassifier()
    r1 = RandomForestClassifier(n_estimators=3 if fast else 10)
    model_dict = {"RF": (r1, "RF"), "DT": (t1, "DT")}

    model, model_name = model_dict[model_name]
    model.fit(x_prop_train, y_prop_train)
    return model, model_name


def get_regression_model(model_name, x_prop_train, y_prop_train):
    """Return a fitted regression model (RF or DT)."""
    fast = bool(os.getenv("FAST_TESTS"))
    t1 = DecisionTreeRegressor()
    r1 = RandomForestRegressor(n_estimators=3 if fast else 10)
    model_dict = {"RF": (r1, "RF"), "DT": (t1, "DT")}

    model, model_name = model_dict[model_name]
    model.fit(x_prop_train, y_prop_train)
    return model, model_name
