from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.utils import helper as helper_module


def test_safe_first_element_handles_various_shapes():
    assert helper_module.safe_first_element(0.5) == 0.5
    assert helper_module.safe_first_element([], default=1.23) == 1.23

    matrix = np.array([[0.1, 0.2, 0.3]])
    assert helper_module.safe_first_element(matrix) == pytest.approx(0.1)
    assert helper_module.safe_first_element(matrix, col=1) == pytest.approx(0.2)

    vector = np.array([9.0, 8.0])
    assert helper_module.safe_first_element(vector, col=1) == pytest.approx(8.0)
    assert helper_module.safe_first_element(vector, col=5) == pytest.approx(0.0)


class _DirListsPredictProbaRegressor:
    """Regressor whose ``dir()`` lists ``predict_proba``, as Pipeline did before sklearn 1.8."""

    def __dir__(self):
        return [*object.__dir__(self), "predict_proba"]


def _learners():
    from types import SimpleNamespace

    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    class LeakyPipelineRegressor(RegressorMixin, BaseEstimator, _DirListsPredictProbaRegressor):
        pass

    return [
        ("classifier", RandomForestClassifier(), True),
        ("regressor", RandomForestRegressor(), False),
        ("pipeline-classifier", Pipeline([("est", RandomForestClassifier())]), True),
        ("pipeline-regressor", Pipeline([("est", RandomForestRegressor())]), False),
        ("regressor-with-leaky-dir", LeakyPipelineRegressor(), False),
        ("classifier-without-probability", SVC(probability=False), True),
        ("duck-typed-classifier", SimpleNamespace(predict_proba=lambda x: x), True),
        ("duck-typed-regressor", SimpleNamespace(predict=lambda x: x), False),
    ]


@pytest.mark.parametrize(
    ("learner", "expected"),
    [(learner, expected) for _, learner, expected in _learners()],
    ids=[name for name, _, _ in _learners()],
)
def test_should_classify_learner_by_estimator_type_when_dir_lists_predict_proba(learner, expected):
    assert helper_module.learner_supports_predict_proba(learner) is expected
