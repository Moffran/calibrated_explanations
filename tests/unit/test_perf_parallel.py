"""Unit tests for the ParallelExecutor facade."""

import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

from calibrated_explanations.parallel import ParallelConfig, ParallelExecutor
from tests.helpers.deprecation import warns_or_raises


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

        from tests.helpers.deprecation import deprecations_error_enabled

        if deprecations_error_enabled():
            with pytest.raises(DeprecationWarning):
                importlib.reload(importlib.import_module("calibrated_explanations.perf.parallel"))
        else:
            with warns_or_raises():
                shim = importlib.reload(
                    importlib.import_module("calibrated_explanations.perf.parallel")
                )

            assert shim.ParallelExecutor is ParallelExecutor
            assert shim.ParallelConfig is ParallelConfig

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
        cfg = ParallelConfig(
            enabled=True, strategy="threads", min_batch_size=1, min_instances_for_parallel=1
        )
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
        # Mock joblib as missing so we test the OS fallback
        with (
            patch("os.name", "nt"),
            patch("calibrated_explanations.parallel.parallel._JoblibParallel", None),
            patch.object(ParallelExecutor, "_is_ci_environment", return_value=False),
        ):
            strategy = executor.resolve_strategy()
            # Should resolve to thread strategy partial
            assert strategy.func == executor.thread_strategy

    def test_strategy_auto_selection_low_cpu(self):
        """Test auto strategy selects threads on low CPU count."""
        cfg = ParallelConfig(enabled=True, strategy="auto")
        executor = ParallelExecutor(cfg)
        with (
            patch("os.name", "posix"),
            patch("os.cpu_count", return_value=2),
            patch.object(ParallelExecutor, "_is_ci_environment", return_value=False),
        ):
            strategy = executor.resolve_strategy()
            assert strategy.func == executor.thread_strategy

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

    def test_telemetry_emission(self, enable_fallbacks):
        """Test that telemetry callback is invoked on fallback."""
        mock_telemetry = MagicMock()
        cfg = ParallelConfig(
            enabled=True,
            strategy="threads",
            min_batch_size=1,
            min_instances_for_parallel=1,
            telemetry=mock_telemetry,
            force_serial_on_failure=True,
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
            with (
                pytest.warns(UserWarning, match=r"fall.*back"),
                pytest.raises(ValueError, match="Boom"),
            ):
                executor.map(failing_fn, items)

            # Verify telemetry was called
            mock_telemetry.assert_called_with(
                "parallel_fallback", {"error": "ValueError('Strategy failed')"}
            )
            assert executor.metrics.fallbacks == 1
            assert executor.metrics.failures == 1

    def test_metrics_tracking_submitted_and_completed(self):
        """Test that metrics accurately track submitted and completed items."""
        cfg = ParallelConfig(
            enabled=True, strategy="threads", min_batch_size=1, min_instances_for_parallel=1
        )
        executor = ParallelExecutor(cfg)

        items = [1, 2, 3, 4, 5]
        results = executor.map(lambda x: x**2, items)

        assert results == [1, 4, 9, 16, 25]
        assert executor.metrics.submitted == 5
        assert executor.metrics.completed == 5
        assert executor.metrics.fallbacks == 0
        assert executor.metrics.failures == 0

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

    def test_thread_strategy_execution(self):
        """Test thread strategy directly executes items in parallel."""
        cfg = ParallelConfig(enabled=True, strategy="threads")
        executor = ParallelExecutor(cfg)

        items = list(range(10))
        results = executor.thread_strategy(lambda x: x * 3, items)
        assert results == [x * 3 for x in items]

    def test_serial_strategy_execution(self):
        """Test serial strategy execution produces correct order."""
        cfg = ParallelConfig(enabled=True, strategy="serial")
        executor = ParallelExecutor(cfg)

        call_order = []

        def tracking_fn(x):
            call_order.append(x)
            return x * 2

        items = [1, 2, 3]
        results = executor._serial_strategy(tracking_fn, items)

        assert results == [2, 4, 6]
        assert call_order == [1, 2, 3]

    def test_auto_strategy_with_environment_override(self, clean_env):
        """Test that auto strategy respects forced overrides via environment."""
        os.environ["CE_PARALLEL"] = "enable,workers=4"
        cfg = ParallelConfig.from_env()
        assert cfg.enabled
        assert cfg.max_workers == 4

    def test_max_workers_environment_override(self, clean_env):
        """Test max_workers can be set via environment variable."""
        os.environ["CE_PARALLEL"] = "workers=6"
        cfg = ParallelConfig.from_env()
        assert cfg.max_workers == 6

    def test_empty_items_handling(self):
        """Test map handles empty item list gracefully."""
        cfg = ParallelConfig(enabled=True, min_batch_size=1)
        executor = ParallelExecutor(cfg)

        results = executor.map(lambda x: x, [])
        assert results == []
        assert executor.metrics.submitted == 0

    def test_single_item_exceeds_min_batch_fallback(self):
        """Test single item with large min_batch falls back to serial."""
        cfg = ParallelConfig(enabled=True, strategy="threads", min_batch_size=10)
        executor = ParallelExecutor(cfg)

        results = executor.map(lambda x: x, [5])
        assert results == [5]
        assert executor.metrics.submitted == 0  # No parallel submission
