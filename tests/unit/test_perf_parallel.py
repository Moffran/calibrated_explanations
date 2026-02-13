"""Unit tests for the ParallelExecutor facade."""

import importlib
import os
import warnings
from unittest.mock import MagicMock, patch

import pytest

from calibrated_explanations.parallel import ParallelConfig, ParallelExecutor


@pytest.fixture
def clean_env():
    """Ensure CE_PARALLEL is unset before/after tests."""
    old = os.environ.get("CE_PARALLEL")
    if "CE_PARALLEL" in os.environ:
        del os.environ["CE_PARALLEL"]
    yield
    if old is not None:
        os.environ["CE_PARALLEL"] = old
    elif "CE_PARALLEL" in os.environ:
        del os.environ["CE_PARALLEL"]


class TestParallelConfig:
    """Tests for configuration loading and environment overrides."""

    def test_perf_parallel_shim_warns_and_forwards(self, clean_env):
        """Deprecated perf shim should forward to canonical parallel module."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            shim = importlib.reload(importlib.import_module("calibrated_explanations.perf.parallel"))

        assert shim.ParallelExecutor.__name__ == ParallelExecutor.__name__
        assert shim.ParallelExecutor.__module__ == ParallelExecutor.__module__
        assert shim.ParallelConfig.__name__ == ParallelConfig.__name__
        assert shim.ParallelConfig.__module__ == ParallelConfig.__module__

    def test_defaults(self):
        """Test default configuration values."""
        cfg = ParallelConfig()
        assert not cfg.enabled
        assert cfg.strategy == "auto"
        assert cfg.min_batch_size == 8
        assert cfg.tiny_workload_threshold is None
        assert cfg.granularity == "instance"

    def test_from_env_enable_flag(self, clean_env):
        """Test enabling via simple flag."""
        os.environ["CE_PARALLEL"] = "1"
        cfg = ParallelConfig.from_env()
        assert cfg.enabled

        os.environ["CE_PARALLEL"] = "true"
        cfg = ParallelConfig.from_env()
        assert cfg.enabled

        os.environ["CE_PARALLEL"] = "on"
        cfg = ParallelConfig.from_env()
        assert cfg.enabled

    def test_from_env_disable_flag(self, clean_env):
        """Test disabling via simple flag."""
        os.environ["CE_PARALLEL"] = "0"
        # Start with enabled base
        base = ParallelConfig(enabled=True)
        cfg = ParallelConfig.from_env(base)
        assert not cfg.enabled

    def test_from_env_complex_string(self, clean_env):
        """Test parsing complex config strings."""
        os.environ["CE_PARALLEL"] = "enable,workers=4,min_batch=100,tiny=24,joblib"
        cfg = ParallelConfig.from_env()
        assert cfg.enabled
        assert cfg.max_workers == 4
        assert cfg.min_batch_size == 100
        assert cfg.tiny_workload_threshold == 24
        assert cfg.strategy == "joblib"

    def test_from_env_granularity(self, clean_env):
        """Test granularity parsing."""
        os.environ["CE_PARALLEL"] = "granularity=instance"
        cfg = ParallelConfig.from_env()
        assert cfg.granularity == "instance"


class TestParallelExecutor:
    """Tests for the executor facade."""


    def test_strategy_auto_selection_joblib(self):
        """Test auto strategy prefers joblib when available and CPUs > 2."""
        cfg = ParallelConfig(enabled=True, strategy="auto")
        executor = ParallelExecutor(cfg)
        # Mock joblib presence by patching the module attribute in the parallel module
        with (
            patch("os.name", "posix"),
            patch("os.cpu_count", return_value=4),
            patch("calibrated_explanations.parallel.parallel._JoblibParallel", new=MagicMock()),
            patch.object(ParallelExecutor, "_is_ci_environment", return_value=False),
        ):
            strategy = executor.resolve_strategy()
            assert strategy.func == executor.joblib_strategy

    def test_joblib_missing_fallback(self, enable_fallbacks):
        """Test fallback to threads if joblib is requested but missing."""
        cfg = ParallelConfig(enabled=True, strategy="joblib")
        executor = ParallelExecutor(cfg)
        # Force joblib to be None
        with (
            patch("calibrated_explanations.parallel.parallel._JoblibParallel", None),
            patch.object(executor, "thread_strategy") as mock_thread,
        ):
            with pytest.warns(UserWarning, match=r"fall.*back"):
                executor.joblib_strategy(lambda x: x, [1])
            mock_thread.assert_called_once()



    def test_metrics_tracking_with_failures(self):
        """Test that metrics track failures correctly when strategy raises."""
        mock_telemetry = MagicMock()
        cfg = ParallelConfig(
            enabled=True,
            strategy="threads",
            min_batch_size=1,
            min_instances_for_parallel=1,
            telemetry=mock_telemetry,
        )
        executor = ParallelExecutor(cfg)

        def failing_fn(x):
            raise RuntimeError("Intentional failure")

        with pytest.raises(RuntimeError):
            executor.map(failing_fn, [1])

        assert executor.metrics.failures == 1


    def test_serial_strategy_execution(self):
        """Test serial strategy execution produces correct order."""
        cfg = ParallelConfig(enabled=True, strategy="serial")
        executor = ParallelExecutor(cfg)

        call_order = []

        def tracking_fn(x):
            call_order.append(x)
            return x * 2

        items = [1, 2, 3]
        results = executor.serial_strategy(tracking_fn, items)

        assert results == [2, 4, 6]
        assert call_order == [1, 2, 3]



