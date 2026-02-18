"""Tests for CalibratedExplainer parallel configuration from environment."""

import os
from unittest.mock import patch

import pytest
from sklearn.linear_model import LogisticRegression
import numpy as np

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


@pytest.fixture
def simple_learner_and_data():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((10, 2))
    y = (x[:, 0] > 0).astype(int)
    learner = LogisticRegression(solver="liblinear")
    learner.fit(x, y)
    return learner, x, y


class TestCalibratedExplainerParallelEnv:
    """Tests for ADR-004 environment variable support in CalibratedExplainer."""

    def test_should_enable_parallel_executor_when_env_var_is_set(self, simple_learner_and_data):
        """Verify that CE_PARALLEL enables parallel executor automatically."""
        learner, x, y = simple_learner_and_data

        with patch.dict(os.environ, {"CE_PARALLEL": "enable,threads"}):
            explainer = CalibratedExplainer(learner, x, y)

            assert explainer.parallel_executor is not None
            assert explainer.parallel_executor.config.enabled is True
            assert explainer.parallel_executor.config.strategy == "threads"
