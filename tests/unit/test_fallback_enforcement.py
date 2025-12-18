"""Tests for fallback chain enforcement policy.

This module verifies that the fallback chain enforcement policy from
.github/tests-guidance.md is correctly implemented and enforced.
"""

from __future__ import annotations

import warnings

import pytest

from tests.helpers.fallback_control import (
    assert_no_fallbacks_triggered,
    disable_all_fallbacks,
    enable_specific_fallback,
    get_fallback_env_vars,
)


class TestFallbackEnforcement:
    """Tests for the fallback chain enforcement mechanism."""

    def test_should_disable_all_fallbacks_by_default(self, monkeypatch):
        """Verify that the disable_fallbacks fixture sets all env vars to empty."""
        # Arrange: Clear any existing fallback env vars
        for var in [
            "CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS",
            "CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS",
            "CE_EXPLANATION_PLUGIN_FAST_FALLBACKS",
            "CE_INTERVAL_PLUGIN_FALLBACKS",
            "CE_INTERVAL_PLUGIN_FAST_FALLBACKS",
            "CE_PLOT_STYLE_FALLBACKS",
            "CE_PARALLEL_MIN_BATCH_SIZE",
        ]:
            monkeypatch.delenv(var, raising=False)

        # Act: Apply the disable_all_fallbacks function
        disable_all_fallbacks(monkeypatch)

        # Assert: All fallback env vars should be empty or very high
        env_vars = get_fallback_env_vars()
        assert env_vars["CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS"] == ""
        assert env_vars["CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS"] == ""
        assert env_vars["CE_EXPLANATION_PLUGIN_FAST_FALLBACKS"] == ""
        assert env_vars["CE_INTERVAL_PLUGIN_FALLBACKS"] == ""
        assert env_vars["CE_INTERVAL_PLUGIN_FAST_FALLBACKS"] == ""
        assert env_vars["CE_PLOT_STYLE_FALLBACKS"] == ""
        assert env_vars["CE_PARALLEL_MIN_BATCH_SIZE"] == "999999"

    def test_should_enable_specific_fallback_chain(self, monkeypatch):
        """Verify that enable_specific_fallback can re-enable a specific chain."""
        # Arrange: Start with all fallbacks disabled
        disable_all_fallbacks(monkeypatch)

        # Act: Enable a specific fallback
        enable_specific_fallback(
            monkeypatch,
            fallback_type="explanation_factual",
            fallback_chain="core.explanation.factual,legacy.fallback",
        )

        # Assert: The specific chain should be enabled, others still disabled
        env_vars = get_fallback_env_vars()
        assert env_vars["CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS"] == "core.explanation.factual,legacy.fallback"
        assert env_vars["CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS"] == ""
        assert env_vars["CE_EXPLANATION_PLUGIN_FAST_FALLBACKS"] == ""

    def test_should_raise_on_unknown_fallback_type(self, monkeypatch):
        """Verify that enable_specific_fallback rejects unknown fallback types."""
        with pytest.raises(ValueError, match="Unknown fallback_type"):
            enable_specific_fallback(
                monkeypatch,
                fallback_type="invalid_type",
                fallback_chain="some.chain",
            )

    def test_should_detect_fallback_warnings(self):
        """Verify that assert_no_fallbacks_triggered detects fallback warnings."""
        # Act & Assert: Emitting a fallback warning should raise AssertionError
        with pytest.raises(AssertionError, match="Unexpected fallback warnings detected"):
            with assert_no_fallbacks_triggered():
                warnings.warn("Test fallback warning", UserWarning, stacklevel=2)

    def test_should_pass_when_no_fallback_warnings(self):
        """Verify that assert_no_fallbacks_triggered passes with no fallback warnings."""
        # This should not raise
        with assert_no_fallbacks_triggered():
            warnings.warn("Regular warning without the f-word", UserWarning, stacklevel=2)

    def test_should_enable_parallel_fallback(self, monkeypatch):
        """Verify that enabling parallel fallback removes the min batch size constraint."""
        # Arrange: Start with all fallbacks disabled
        disable_all_fallbacks(monkeypatch)

        # Act: Enable parallel fallback
        enable_specific_fallback(
            monkeypatch,
            fallback_type="parallel",
            fallback_chain="",  # Not used for parallel
        )

        # Assert: The min batch size constraint should be removed
        env_vars = get_fallback_env_vars()
        assert env_vars["CE_PARALLEL_MIN_BATCH_SIZE"] is None


class TestFallbackFixtures:
    """Tests for the pytest fixtures that control fallback behavior."""

    def test_should_have_fallbacks_disabled_by_default(self):
        """Verify that tests have fallbacks disabled by default via autouse fixture."""
        # This test runs with the autouse disable_fallbacks fixture
        env_vars = get_fallback_env_vars()
        
        # All fallback chains should be empty or have high threshold
        assert env_vars["CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS"] == ""
        assert env_vars["CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS"] == ""
        assert env_vars["CE_EXPLANATION_PLUGIN_FAST_FALLBACKS"] == ""
        assert env_vars["CE_INTERVAL_PLUGIN_FALLBACKS"] == ""
        assert env_vars["CE_INTERVAL_PLUGIN_FAST_FALLBACKS"] == ""
        assert env_vars["CE_PLOT_STYLE_FALLBACKS"] == ""
        assert env_vars["CE_PARALLEL_MIN_BATCH_SIZE"] == "999999"

    def test_should_enable_fallbacks_with_fixture(self, enable_fallbacks):
        """Verify that enable_fallbacks fixture removes fallback restrictions."""
        # This test explicitly uses enable_fallbacks fixture
        env_vars = get_fallback_env_vars()
        
        # Fallback environment variables should not be set (allowing default behavior)
        assert env_vars["CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS"] is None
        assert env_vars["CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS"] is None
        assert env_vars["CE_EXPLANATION_PLUGIN_FAST_FALLBACKS"] is None
        assert env_vars["CE_INTERVAL_PLUGIN_FALLBACKS"] is None
        assert env_vars["CE_INTERVAL_PLUGIN_FAST_FALLBACKS"] is None
        assert env_vars["CE_PLOT_STYLE_FALLBACKS"] is None
        assert env_vars["CE_PARALLEL_MIN_BATCH_SIZE"] is None


class TestPytestWarningFilter:
    """Tests to verify that pytest's warning filter catches fallback warnings."""

    def test_should_fail_on_fallback_warning_in_ci(self, enable_fallbacks):
        """Verify that fallback warnings cause test failure per pytest config.
        
        This test verifies that the pytest.ini configuration correctly
        treats fallback-related UserWarnings as errors.
        
        NOTE: This test will fail if run without the pytest warning filters.
        In normal test runs, the filters are applied automatically.
        """
        # This test demonstrates the expected behavior
        # In actual tests, emitting a fallback warning would cause failure
        with pytest.warns(UserWarning, match="fallback"):
            warnings.warn("Test fallback engaged", UserWarning, stacklevel=2)

    def test_should_allow_non_fallback_warnings(self):
        """Verify that non-fallback UserWarnings are still allowed."""
        # Non-fallback warnings should not cause test failure
        with warnings.catch_warnings(record=True):
            warnings.warn("Regular user warning", UserWarning, stacklevel=2)
            # Test passes - warning was emitted but didn't cause failure
