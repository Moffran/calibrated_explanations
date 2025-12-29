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
    learner = LogisticRegression()
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

    def test_should_not_enable_parallel_executor_when_env_var_is_missing(
        self, simple_learner_and_data
    ):
        """Verify that parallel executor is None by default."""
        learner, x, y = simple_learner_and_data

        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            # Restore PATH etc if needed, but for this test clearing might be too aggressive if imports depend on it.
            # Better to just ensure CE_PARALLEL is not there.
            if "CE_PARALLEL" in os.environ:
                del os.environ["CE_PARALLEL"]

            explainer = CalibratedExplainer(learner, x, y)

            assert explainer.parallel_executor is None

    def test_should_respect_explicit_perf_parallel_over_env_var(self, simple_learner_and_data):
        """Verify that explicit perf_parallel argument takes precedence."""
        learner, x, y = simple_learner_and_data

        with patch.dict(os.environ, {"CE_PARALLEL": "enable"}):
            # Explicitly pass None (should result in None, overriding env? No, logic says if None, check env)
            # Wait, logic is: if perf_parallel is None: check env.
            # So passing None explicitly is same as not passing it.

            # If we want to explicitly DISABLE it despite env var, we can't pass None.
            # We would have to pass a disabled executor.

            from calibrated_explanations.parallel import ParallelExecutor, ParallelConfig

            disabled_executor = ParallelExecutor(ParallelConfig(enabled=False))

            explainer = CalibratedExplainer(learner, x, y, perf_parallel=disabled_executor)

            assert explainer.parallel_executor is disabled_executor
            assert explainer.parallel_executor.config.enabled is False
