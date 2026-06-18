import logging

import pytest

from calibrated_explanations.logging import (
    LoggingContextFilter,
    configure_logging,
    ensure_logging_context_filter,
    get_logging_context,
    logging_context,
    reset_module_config_manager,
    telemetry_diagnostic_mode,
    update_logging_context,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    logger = logging.getLogger("calibrated_explanations")
    original_level = logger.level
    original_handlers = list(logger.handlers)
    original_filters = list(logger.filters)
    monkeypatch.delenv("CE_TELEMETRY_DIAGNOSTIC_MODE", raising=False)
    reset_module_config_manager()
    yield
    logger.setLevel(original_level)
    logger.handlers[:] = original_handlers
    logger.filters[:] = original_filters
    reset_module_config_manager()


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


def test_update_logging_context_ignores_unknown_keys_and_context_manager_noop_for_unknown():
    update_logging_context(request_id=None, tenant_id=None, explainer_id=None)

    update_logging_context(request_id="req-1", unknown_key="ignored")
    ctx = get_logging_context()
    assert ctx["request_id"] == "req-1"
    assert "unknown_key" not in ctx

    with logging_context(unknown_key="ignored"):
        assert get_logging_context()["request_id"] == "req-1"

    assert get_logging_context()["request_id"] == "req-1"


def test_configure_logging_adds_handler_and_context_filter():
    """configure_logging should configure only the package root logger."""
    logger = logging.getLogger("calibrated_explanations")
    handler = logging.NullHandler()

    configured = configure_logging(level=logging.INFO, handler=handler)

    assert configured is logger
    assert configured.level == logging.INFO
    assert handler in configured.handlers
    assert any(isinstance(item, LoggingContextFilter) for item in configured.filters)


def test_configure_logging_creates_default_handler():
    """configure_logging should install a default stream handler when omitted."""
    logger = logging.getLogger("calibrated_explanations")
    before = set(logger.handlers)

    configured = configure_logging(level="WARNING")

    added = [handler for handler in configured.handlers if handler not in before]
    assert configured is logger
    assert any(isinstance(handler, logging.StreamHandler) for handler in added)


def test_telemetry_diagnostic_mode_reads_env(monkeypatch):
    """telemetry_diagnostic_mode should use ConfigManager-backed env resolution."""
    monkeypatch.setenv("CE_TELEMETRY_DIAGNOSTIC_MODE", "true")
    reset_module_config_manager()

    assert telemetry_diagnostic_mode() is True


def test_ensure_logging_context_filter_is_idempotent():
    """Repeated filter installation should not duplicate context filters."""
    logger = logging.getLogger("calibrated_explanations.test_idempotent")

    ensure_logging_context_filter(logger.name)
    ensure_logging_context_filter(logger.name)

    filters = [item for item in logger.filters if isinstance(item, LoggingContextFilter)]
    assert len(filters) == 1


def test_root_configure_logging_export_is_lazy():
    """The package root must expose configure_logging as a public lazy symbol."""
    import calibrated_explanations

    assert calibrated_explanations.configure_logging is configure_logging
