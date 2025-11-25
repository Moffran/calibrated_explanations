"""Unit tests for the ParallelExecutor facade."""
import os
from unittest.mock import MagicMock, patch

import pytest

from calibrated_explanations.perf.parallel import (
    ParallelConfig,
    ParallelExecutor,
    ParallelMetrics,
)


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

    def test_defaults(self):
        """Test default configuration values."""
        cfg = ParallelConfig()
        assert not cfg.enabled
        assert cfg.strategy == "auto"
        assert cfg.min_batch_size == 32
        assert cfg.granularity == "feature"

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
        os.environ["CE_PARALLEL"] = "enable,workers=4,min_batch=100,joblib"
        cfg = ParallelConfig.from_env()
        assert cfg.enabled
        assert cfg.max_workers == 4
        assert cfg.min_batch_size == 100
        assert cfg.strategy == "joblib"

    def test_from_env_granularity(self, clean_env):
        """Test granularity parsing."""
        os.environ["CE_PARALLEL"] = "granularity=instance"
        cfg = ParallelConfig.from_env()
        assert cfg.granularity == "instance"


class TestParallelExecutor:
    """Tests for the executor facade."""

    def test_map_serial_fallback_disabled(self):
        """Test that map runs serially when disabled."""
        cfg = ParallelConfig(enabled=False)
        executor = ParallelExecutor(cfg)
        items = [1, 2, 3]
        results = executor.map(lambda x: x * 2, items)
        assert results == [2, 4, 6]
        assert executor.metrics.submitted == 0  # Not counted as parallel submission

    def test_map_serial_fallback_small_batch(self):
        """Test fallback for small batches."""
        cfg = ParallelConfig(enabled=True, min_batch_size=100)
        executor = ParallelExecutor(cfg)
        items = [1, 2, 3]
        results = executor.map(lambda x: x * 2, items)
        assert results == [2, 4, 6]
        assert executor.metrics.submitted == 0

    def test_map_parallel_execution(self):
        """Test actual parallel execution (using threads for simplicity)."""
        cfg = ParallelConfig(enabled=True, strategy="threads", min_batch_size=1)
        executor = ParallelExecutor(cfg)
        items = [1, 2, 3]
        results = executor.map(lambda x: x * 2, items)
        assert results == [2, 4, 6]
        assert executor.metrics.submitted == 3
        assert executor.metrics.completed == 3

    def test_strategy_auto_selection_windows(self):
        """Test auto strategy selects threads on Windows."""
        cfg = ParallelConfig(enabled=True, strategy="auto")
        executor = ParallelExecutor(cfg)
        with patch("os.name", "nt"):
            strategy = executor._resolve_strategy()
            # Should resolve to thread strategy partial
            assert strategy.func == executor._thread_strategy

    def test_strategy_auto_selection_low_cpu(self):
        """Test auto strategy selects threads on low CPU count."""
        cfg = ParallelConfig(enabled=True, strategy="auto")
        executor = ParallelExecutor(cfg)
        with patch("os.name", "posix"), patch("os.cpu_count", return_value=2):
            strategy = executor._resolve_strategy()
            assert strategy.func == executor._thread_strategy

    def test_strategy_auto_selection_joblib(self):
        """Test auto strategy prefers joblib when available and CPUs > 2."""
        cfg = ParallelConfig(enabled=True, strategy="auto")
        executor = ParallelExecutor(cfg)
        # Mock joblib presence by patching the module attribute in the parallel module
        with patch("os.name", "posix"), patch("os.cpu_count", return_value=4):
            with patch("calibrated_explanations.perf.parallel._JoblibParallel", new=MagicMock()):
                strategy = executor._resolve_strategy()
                assert strategy.func == executor._joblib_strategy

    def test_joblib_missing_fallback(self):
        """Test fallback to threads if joblib is requested but missing."""
        cfg = ParallelConfig(enabled=True, strategy="joblib")
        executor = ParallelExecutor(cfg)
        # Force joblib to be None
        with patch("calibrated_explanations.perf.parallel._JoblibParallel", None):
            strategy = executor._resolve_strategy()
            # The _joblib_strategy method itself handles the fallback check
            # so we invoke it to verify it calls _thread_strategy
            with patch.object(executor, "_thread_strategy") as mock_thread:
                executor._joblib_strategy(lambda x: x, [1])
                mock_thread.assert_called_once()

    def test_telemetry_emission(self):
        """Test that telemetry callback is invoked on fallback."""
        mock_telemetry = MagicMock()
        cfg = ParallelConfig(
            enabled=True,
            strategy="threads",
            min_batch_size=1,
            telemetry=mock_telemetry
        )
        executor = ParallelExecutor(cfg)
        
        # Force an exception during execution
        def failing_fn(x):
            raise ValueError("Boom")
            
        # map should catch the exception and fall back to serial
        # but since the serial execution will also fail (same function), 
        # we need to be careful. The map implementation catches exception during STRATEGY execution.
        # If we make the strategy raise, it falls back to serial.
        
        with patch.object(executor, "_resolve_strategy", side_effect=ValueError("Strategy failed")):
            items = [1]
            # The serial fallback will raise ValueError("Boom") when it runs failing_fn
            # So we expect the map call to raise eventually, OR if map catches everything?
            # Looking at code: map catches Exception during strategy execution, emits telemetry, then runs serial.
            # Serial run will raise ValueError("Boom") which is NOT caught by map.
            with pytest.raises(ValueError, match="Boom"):
                executor.map(failing_fn, items)
            
            # Verify telemetry was called
            mock_telemetry.assert_called_with("parallel_fallback", {"error": "ValueError('Strategy failed')"})
            assert executor.metrics.fallbacks == 1
            assert executor.metrics.failures == 1
