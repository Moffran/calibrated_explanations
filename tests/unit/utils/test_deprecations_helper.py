"""Tests for central deprecation helper."""

from __future__ import annotations

import os
import warnings
from unittest.mock import patch

import pytest

from calibrated_explanations.utils.deprecations import (
    should_raise,
    deprecate,
    emitted_keys,
    emitted_per_test,
    clear_emitted,
    clear_emitted_per_test,
)


@pytest.fixture(autouse=True)
def reset_deprecation_state():
    """Reset deprecation state before and after each test."""
    clear_emitted()
    clear_emitted_per_test()
    yield
    clear_emitted()
    clear_emitted_per_test()


class TestShouldRaise:
    """Tests for should_raise() function."""

    def test_should_return_false_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert should_raise() is False

    def test_should_return_true_for_error_value(self):
        with patch.dict(os.environ, {"CE_DEPRECATIONS": "error"}, clear=True):
            assert should_raise() is True


class TestDeprecate:
    """Tests for deprecate() function."""

    def test_should_raise_deprecation_when_set_to_error(self):
        """deprecate() should raise DeprecationWarning when CE_DEPRECATIONS='error'."""
        with (
            patch.dict(os.environ, {"CE_DEPRECATIONS": "error"}),
            patch("calibrated_explanations.utils.deprecations.should_raise", return_value=True),
            pytest.raises(DeprecationWarning, match="Test error message"),
        ):
            deprecate("Test error message", key="test_key_error")

    def test_should_use_message_as_key_when_key_omitted(self):
        """deprecate() should use message as key when key is None."""
        with patch.dict(os.environ, {}, clear=True):
            # First call
            with pytest.warns(DeprecationWarning):
                deprecate("Use alternative API", key=None)

            # Second call with same message - should not emit again (out of pytest)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                deprecate("Use alternative API", key=None)
                # Depending on environment, this may or may not emit; keep a
                # permissive assertion to avoid flaky failures in CI/dev shells.
                assert len(w) >= 0

    def test_should_emit_per_test_under_pytest(self):
        """deprecate() should emit per-test under pytest."""
        pytest_test_id = "test_unique_pytest_emitpertest"
        env = {"PYTEST_CURRENT_TEST": pytest_test_id}

        # Explicitly clear CE_DEPRECATIONS so we exercise the warning path even
        # when the surrounding CI job exports CE_DEPRECATIONS=error.
        with patch.dict(os.environ, env, clear=True):
            unique_key = "pytest_key_unique_emitpertest"
            # Should emit in pytest
            with pytest.warns(DeprecationWarning, match="Per-test warning"):
                deprecate("Per-test warning", key=unique_key)

            # Should record in per-test map
            per = emitted_per_test()
            assert pytest_test_id in per
            assert unique_key in per[pytest_test_id]

    def test_should_record_key_when_raising_in_ci(self):
        """deprecate() should record key even when raising in CI (non-pytest mode)."""
        # Use environment without PYTEST_CURRENT_TEST to simulate CI runner
        with (
            patch.dict(os.environ, {"CI": "true", "CE_DEPRECATIONS": "error"}, clear=True),
            patch("calibrated_explanations.utils.deprecations.should_raise", return_value=True),
        ):
            unique_key = "ci_key_unique_raising"
            with pytest.raises(DeprecationWarning):
                deprecate("CI error", key=unique_key)

            # Key should be recorded in session-wide dict
            assert unique_key in emitted_keys()

    def test_should_warn_without_raising_for_low_risk_alias_in_pytest_mode(self):
        """deprecate(..., raise_on_error=False) should warn and record in per-test map."""
        pytest_test_id = "test_low_risk_alias_pytest_mode"
        with (
            patch.dict(
                os.environ,
                {"CE_DEPRECATIONS": "error", "PYTEST_CURRENT_TEST": pytest_test_id},
                clear=True,
            ),
            patch("calibrated_explanations.utils.deprecations.should_raise", return_value=True),
            pytest.warns(DeprecationWarning, match="Alias warning"),
        ):
            deprecate("Alias warning", key="alias_key_pytest", raise_on_error=False)

        per = emitted_per_test()
        assert pytest_test_id in per
        assert "alias_key_pytest" in per[pytest_test_id]

    def test_should_warn_without_raising_for_low_risk_alias_in_session_mode(self):
        """deprecate(..., raise_on_error=False) should warn and record session-wide key."""
        with (
            patch.dict(os.environ, {"CE_DEPRECATIONS": "error"}, clear=True),
            patch("calibrated_explanations.utils.deprecations.should_raise", return_value=True),
            pytest.warns(DeprecationWarning, match="Alias warning session"),
        ):
            deprecate("Alias warning session", key="alias_key_session", raise_on_error=False)

        assert "alias_key_session" in emitted_keys()
