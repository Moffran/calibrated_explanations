import pytest

from calibrated_explanations.logging import (
    get_logging_context,
    logging_context,
    update_logging_context,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("CE_TELEMETRY_DIAGNOSTIC_MODE", raising=False)
    yield


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
