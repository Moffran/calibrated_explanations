import logging

import pytest

from calibrated_explanations.logging import (
    LoggingContextFilter,
    ensure_logging_context_filter,
    get_logging_context,
    logging_context,
    telemetry_diagnostic_mode,
    update_logging_context,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("CE_TELEMETRY_DIAGNOSTIC_MODE", raising=False)
    yield


def test_should_enable_diagnostic_mode_when_env_true(monkeypatch):
    monkeypatch.setenv("CE_TELEMETRY_DIAGNOSTIC_MODE", "true")
    assert telemetry_diagnostic_mode() is True


def test_should_disable_diagnostic_mode_when_env_false(monkeypatch):
    monkeypatch.setenv("CE_TELEMETRY_DIAGNOSTIC_MODE", "0")
    assert telemetry_diagnostic_mode() is False


def test_should_fall_back_to_pyproject_when_env_missing(monkeypatch):
    def _fake_pyproject(_path):
        return {"diagnostic_mode": True}

    monkeypatch.delenv("CE_TELEMETRY_DIAGNOSTIC_MODE", raising=False)
    monkeypatch.setattr("calibrated_explanations.logging.read_pyproject_section", _fake_pyproject)

    assert telemetry_diagnostic_mode() is True


def test_should_return_false_when_config_absent(monkeypatch):
    monkeypatch.setattr(
        "calibrated_explanations.logging.read_pyproject_section", lambda _path: None
    )
    assert telemetry_diagnostic_mode() is False


def test_update_logging_context_invalid_key():
    """Verify update_logging_context ignores invalid keys."""
    # This should not raise an error
    update_logging_context(invalid_key="test")
    assert "invalid_key" not in get_logging_context()


def test_logging_context_invalid_key():
    """Verify logging_context ignores invalid keys."""
    with logging_context(invalid_key="test"):
        assert "invalid_key" not in get_logging_context()


def test_ensure_logging_context_filter_duplicate():
    """Verify ensure_logging_context_filter does not add duplicate filters."""
    logger_name = "test_duplicate_logger"
    ensure_logging_context_filter(logger_name)
    logger = logging.getLogger(logger_name)
    initial_filter_count = len(logger.filters)
    assert initial_filter_count >= 1

    # Second call should return early
    ensure_logging_context_filter(logger_name)
    assert len(logger.filters) == initial_filter_count


def test_should_coerce_bool_with_various_inputs(monkeypatch):
    from calibrated_explanations.logging import coerce_bool

    assert coerce_bool(True) is True
    assert coerce_bool(False) is False
    assert coerce_bool(None) is False
    assert coerce_bool("1") is True
    assert coerce_bool("yes") is True
    assert coerce_bool("ON") is True
    assert coerce_bool("enable") is True
    assert coerce_bool("0") is False
    assert coerce_bool("random") is False


def test_should_set_none_values_in_logging_context_to_clear():
    assert get_logging_context() == {}
    update_logging_context(plugin_identifier="test")
    assert get_logging_context()["plugin_identifier"] == "test"
    with logging_context(plugin_identifier=None):
        assert get_logging_context() == {}
    assert get_logging_context()["plugin_identifier"] == "test"
    # cleanup
    update_logging_context(plugin_identifier=None)


def test_should_set_none_in_update_logging_context():
    update_logging_context(request_id="req-1")
    assert get_logging_context()["request_id"] == "req-1"
    update_logging_context(request_id=None)
    assert get_logging_context() == {}


def test_should_inject_and_reset_logging_context():
    assert get_logging_context() == {}

    with logging_context(request_id="req-1", plugin_identifier="core.test"):
        ctx = get_logging_context()
        assert ctx["request_id"] == "req-1"
        assert ctx["plugin_identifier"] == "core.test"

    # context should be restored
    assert get_logging_context() == {}


def test_should_update_logging_context_incrementally():
    update_logging_context(request_id="req-2")
    assert get_logging_context()["request_id"] == "req-2"

    update_logging_context(plugin_identifier="core.other")
    ctx = get_logging_context()
    assert ctx["request_id"] == "req-2"
    assert ctx["plugin_identifier"] == "core.other"


def test_should_attach_filter_once_and_inject_context(monkeypatch):
    logger = logging.getLogger("calibrated_explanations.test")
    # clear existing LoggingContextFilter instances
    logger.filters = [f for f in logger.filters if not isinstance(f, LoggingContextFilter)]

    update_logging_context(explainer_id="expl-1")
    ensure_logging_context_filter(logger_name="calibrated_explanations.test")
    ensure_logging_context_filter(logger_name="calibrated_explanations.test")

    filters = [f for f in logger.filters if isinstance(f, LoggingContextFilter)]
    assert len(filters) == 1

    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="msg",
        args=(),
        exc_info=None,
    )
    assert filters[0].filter(record) is True
    assert getattr(record, "explainer_id") == "expl-1"

    # clean up for other tests
    logger.filters = [f for f in logger.filters if not isinstance(f, LoggingContextFilter)]
    update_logging_context(explainer_id=None, plugin_identifier=None, request_id=None)
