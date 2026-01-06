import logging
import os

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
    monkeypatch.setattr(
        "calibrated_explanations.logging.read_pyproject_section", _fake_pyproject
    )

    assert telemetry_diagnostic_mode() is True


def test_should_return_false_when_config_absent(monkeypatch):
    monkeypatch.setattr(
        "calibrated_explanations.logging.read_pyproject_section", lambda _path: None
    )
    assert telemetry_diagnostic_mode() is False


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
