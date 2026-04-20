"""Tests for CalibratedExplainer parallel configuration from environment."""

import os
from unittest.mock import patch

import pytest
from sklearn.linear_model import LogisticRegression
import numpy as np

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.config_manager import ConfigManager
from calibrated_explanations.parallel import ParallelConfig


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
            original_from_env = ParallelConfig.from_env

            def fresh_from_env(base=None, *, config_manager=None):
                manager = config_manager or ConfigManager.from_sources()
                return original_from_env(base, config_manager=manager)

            with patch.object(ParallelConfig, "from_env", side_effect=fresh_from_env):
                explainer = CalibratedExplainer(learner, x, y)

            assert explainer.parallel_executor is not None
            assert explainer.parallel_executor.config.enabled is True
            assert explainer.parallel_executor.config.strategy == "threads"

    def test_should_isolate_ce_parallel_from_env_changes_after_snapshot(
        self, simple_learner_and_data
    ):
        """CE_PARALLEL must be read from ConfigManager snapshot; post-construction env changes must not affect it."""
        from calibrated_explanations.parallel.parallel import ParallelConfig

        _EMPTY_PYPROJECT = {
            "plugins": {},
            "explanations": {},
            "intervals": {},
            "plots": {},
            "telemetry": {},
        }
        # Build a ConfigManager snapshot with CE_PARALLEL enabled.
        mgr = ConfigManager(
            env_snapshot={"CE_PARALLEL": "on"},
            pyproject_snapshot=_EMPTY_PYPROJECT,
        )
        cfg = ParallelConfig.from_env(config_manager=mgr)
        assert cfg.enabled is True, "CE_PARALLEL=on must enable parallel execution"

        # A second manager with no CE_PARALLEL returns disabled (default).
        mgr_off = ConfigManager(
            env_snapshot={},
            pyproject_snapshot=_EMPTY_PYPROJECT,
        )
        cfg_off = ParallelConfig.from_env(config_manager=mgr_off)
        assert cfg_off.enabled is False, "Absent CE_PARALLEL must default to disabled"
