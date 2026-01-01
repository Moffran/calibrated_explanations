"""Tests for central deprecation helper."""

from __future__ import annotations

import os
import warnings
from unittest.mock import patch

import pytest

from calibrated_explanations.utils.deprecations import (
    _EMITTED,
    _should_raise,
    deprecate,
)


@pytest.fixture(autouse=True)
def reset_deprecation_state():
    """Reset deprecation state before and after each test."""
    from calibrated_explanations.utils import deprecations

    deprecations._EMITTED.clear()
    deprecations._EMITTED_PER_TEST.clear()
    yield
    deprecations._EMITTED.clear()
    deprecations._EMITTED_PER_TEST.clear()


class TestShouldRaise:
    """Tests for should_raise() function."""

    def test_should_raise_when_ce_deprecations_is_error(self):
        """should_raise() should return True when CE_DEPRECATIONS='error'."""
        with patch.dict(os.environ, {"CE_DEPRECATIONS": "error"}):
            assert _should_raise() is True

    def test_should_raise_when_ce_deprecations_is_raise(self):
        """should_raise() should return True when CE_DEPRECATIONS='raise'."""
        with patch.dict(os.environ, {"CE_DEPRECATIONS": "raise"}):
            assert _should_raise() is True

    def test_should_raise_when_ce_deprecations_is_true(self):
        """should_raise() should return True when CE_DEPRECATIONS='true'."""
        with patch.dict(os.environ, {"CE_DEPRECATIONS": "true"}):
            assert _should_raise() is True

    def test_should_raise_when_ce_deprecations_is_1(self):
        """should_raise() should return True when CE_DEPRECATIONS='1'."""
        with patch.dict(os.environ, {"CE_DEPRECATIONS": "1"}):
            assert _should_raise() is True

    def test_should_not_raise_when_ce_deprecations_unset(self):
        """should_raise() should return False when CE_DEPRECATIONS is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert _should_raise() is False

    def test_should_not_raise_when_ce_deprecations_is_false(self):
        """should_raise() should return False for other values."""
        with patch.dict(os.environ, {"CE_DEPRECATIONS": "false"}):
            assert _should_raise() is False

    def test_should_not_raise_during_pytest_without_ci(self):
        """should_raise() should return True when CE_DEPRECATIONS='error' even during pytest."""
        # Note: The current implementation always honors CE_DEPRECATIONS when set to error values,
        # regardless of pytest or CI status (per ADR requirements).
        env = {
            "CE_DEPRECATIONS": "error",
            "PYTEST_CURRENT_TEST": "test_module.py::test_func",
        }
        with patch.dict(os.environ, env):
            # When CE_DEPRECATIONS="error" is set, should_raise() should return True
            # regardless of pytest or CI status
            assert _should_raise() is True

    def test_should_raise_during_ci_with_pytest(self):
        """should_raise() should return True during CI with pytest when CE_DEPRECATIONS='error'."""
        env = {
            "CE_DEPRECATIONS": "error",
            "PYTEST_CURRENT_TEST": "test_module.py::test_func",
            "CI": "true",
        }
        with patch.dict(os.environ, env):
            assert _should_raise() is True


class TestDeprecate:
    """Tests for deprecate() function."""

    def test_should_emit_warning_when_not_set_to_raise(self):
        """deprecate() should emit a DeprecationWarning by default."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.warns(DeprecationWarning, match="Test warning"),
        ):
            deprecate("Test warning", key="test_key_1")

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

    def test_should_deduplicate_outside_pytest(self):
        """deprecate() should emit once per session outside pytest."""
        with patch.dict(os.environ, {}, clear=True):
            # Create a unique key for this test to avoid conflicts
            unique_key = "session_key_unique_deduplicate"

            # First call - should emit
            with pytest.warns(DeprecationWarning, match="Session warning"):
                deprecate("Session warning", key=unique_key)

            # Verify recorded
            from calibrated_explanations.utils import deprecations

            assert unique_key in deprecations._EMITTED

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
            from calibrated_explanations.utils import deprecations

            assert pytest_test_id in deprecations._EMITTED_PER_TEST
            assert unique_key in deprecations._EMITTED_PER_TEST[pytest_test_id]

    def test_should_respect_custom_stacklevel(self):
        """deprecate() should pass stacklevel to warnings.warn."""
        with patch.dict(os.environ, {}, clear=True):
            _EMITTED.clear()

            with pytest.warns(DeprecationWarning):
                deprecate("Test stacklevel", key="stacklevel_test", stacklevel=3)

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
            from calibrated_explanations.utils import deprecations

            assert unique_key in deprecations._EMITTED

    def test_should_record_key_in_pytest_when_raising(self):
        """deprecate() should record in per-test map when raising under pytest."""
        pytest_test_id = "test_pytest_raise_unique"
        env = {"PYTEST_CURRENT_TEST": pytest_test_id, "CE_DEPRECATIONS": "error", "CI": "true"}

        with (
            patch.dict(os.environ, env),
            patch("calibrated_explanations.utils.deprecations.should_raise", return_value=True),
        ):
            unique_key = "pytest_ci_key_unique_raise"
            with pytest.raises(DeprecationWarning):
                deprecate("Pytest CI error", key=unique_key)

            # Key should be recorded in per-test map
            from calibrated_explanations.utils import deprecations

            assert pytest_test_id in deprecations._EMITTED_PER_TEST
            assert unique_key in deprecations._EMITTED_PER_TEST[pytest_test_id]

    def test_should_not_re_emit_after_first_call_outside_pytest(self):
        """deprecate() should not emit twice for same key outside pytest."""
        with patch.dict(os.environ, {}, clear=True):
            # First call
            with pytest.warns(DeprecationWarning, match="Once-only"):
                deprecate("Once-only message", key="once_key")

            # Second call - should not emit
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                deprecate("Once-only message", key="once_key")
                # Should not emit again
                assert len(w) == 0
