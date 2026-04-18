from __future__ import annotations

import logging
from collections.abc import Mapping

import pytest
import unittest.mock

from calibrated_explanations.core.config_manager import ConfigManager, _KNOWN_ENV_KEYS
from calibrated_explanations.governance.events import (
    build_config_governance_event,
    validate_config_governance_event,
    validate_governance_event,
)
from calibrated_explanations.utils.exceptions import ConfigurationError, ValidationError


_EMPTY_PYPROJECT: dict[str, dict[str, object]] = {
    "plugins": {},
    "explanations": {},
    "intervals": {},
    "plots": {},
    "telemetry": {},
}
_CONFIG_PAYLOAD_KEYS = {
    "schema_version",
    "event_id",
    "event_name",
    "event_type",
    "profile_id",
    "config_schema_version",
    "strict",
    "source_count",
    "validation_issue_count",
    "timestamp",
    "details",
}


def config_event_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == "calibrated_explanations.governance.config"
    ]


def event_payload(record: logging.LogRecord) -> dict[str, object]:
    return {key: getattr(record, key) for key in _CONFIG_PAYLOAD_KEYS if key in record.__dict__}


def collect_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(collect_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(collect_keys(item))
    return keys


def assert_minimization_constraints(payload: dict[str, object], event_type: str) -> None:
    forbidden_keys = {"env_snapshot", "pyproject_snapshot", "values", "message", "_env_snapshot"}
    payload_keys = collect_keys(payload)
    assert forbidden_keys.isdisjoint(payload_keys)
    details = payload.get("details")
    if event_type == "resolve":
        assert details is None
    if event_type == "export":
        assert details == {"diagnostic_only": True}
    if event_type == "validation_failure":
        assert isinstance(details, dict)
        assert set(details.keys()) == {"location", "issue_count"}


def test_should_emit_resolve_event_for_from_sources_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        "calibrated_explanations.core.config_manager.read_pyproject_section",
        lambda _path: {},
    )
    for key in _KNOWN_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CE_PLOT_STYLE", "legacy")

    with caplog.at_level(logging.INFO, logger="calibrated_explanations.governance.config"):
        ConfigManager.from_sources()
        records = config_event_records(caplog)
        assert len(records) == 1
        record = records[0]
        payload = event_payload(record)
        validate_config_governance_event(payload)
        assert record.event_type == "resolve"
        assert record.validation_issue_count == 0
        assert record.source_count == 1
        assert record.name == "calibrated_explanations.governance.config"

        caplog.clear()
        ConfigManager(env_snapshot={}, pyproject_snapshot=_EMPTY_PYPROJECT)
        assert config_event_records(caplog) == []


def test_should_emit_export_event_with_diagnostic_only_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = ConfigManager(
        env_snapshot={"CE_PLOT_STYLE": "legacy"},
        pyproject_snapshot=_EMPTY_PYPROJECT,
    )

    with caplog.at_level(logging.INFO, logger="calibrated_explanations.governance.config"):
        manager.export_effective()

    records = config_event_records(caplog)
    assert len(records) == 1
    record = records[0]
    payload = event_payload(record)
    validate_config_governance_event(payload)
    assert record.event_type == "export"
    assert record.details == {"diagnostic_only": True}
    assert record.source_count == 1


def test_should_emit_validation_failure_event_for_strict_and_non_strict_paths(
    caplog: pytest.LogCaptureFixture,
) -> None:
    invalid_pyproject = {
        "plugins": {"trusted": 123},
        "explanations": {},
        "intervals": {},
        "plots": {},
        "telemetry": {},
    }

    with (
        caplog.at_level(logging.INFO, logger="calibrated_explanations.governance.config"),
        pytest.raises(ConfigurationError),
    ):
        ConfigManager(
            env_snapshot={"CE_PLOT_STYLE": "legacy"},
            pyproject_snapshot=invalid_pyproject,
            strict=True,
        )

    strict_records = config_event_records(caplog)
    assert len(strict_records) == 1
    strict_record = strict_records[0]
    strict_payload = event_payload(strict_record)
    validate_config_governance_event(strict_payload)
    assert strict_record.event_type == "validation_failure"
    assert strict_record.validation_issue_count == 1
    assert set(strict_record.details.keys()) == {"location", "issue_count"}

    caplog.clear()
    with (
        caplog.at_level(logging.INFO, logger="calibrated_explanations.governance.config"),
        pytest.warns(UserWarning, match="Config validation issues captured with strict=False"),
    ):
        ConfigManager(
            env_snapshot={"CE_PLOT_STYLE": "legacy"},
            pyproject_snapshot=invalid_pyproject,
            strict=False,
        )

    non_strict_records = config_event_records(caplog)
    assert len(non_strict_records) == 1
    non_strict_record = non_strict_records[0]
    non_strict_payload = event_payload(non_strict_record)
    validate_config_governance_event(non_strict_payload)
    assert non_strict_record.event_type == "validation_failure"
    assert non_strict_record.validation_issue_count == 1
    assert set(non_strict_record.details.keys()) == {"location", "issue_count"}


def test_should_validate_all_config_event_types_and_reject_plugin_validator_path() -> None:
    resolve_payload = build_config_governance_event(
        event_type="resolve",
        profile_id="default",
        config_schema_version="1",
        strict=True,
        source_count=0,
        validation_issue_count=0,
    )
    export_payload = build_config_governance_event(
        event_type="export",
        profile_id="default",
        config_schema_version="1",
        strict=True,
        source_count=3,
        validation_issue_count=0,
    )
    failure_payload = build_config_governance_event(
        event_type="validation_failure",
        profile_id="default",
        config_schema_version="1",
        strict=False,
        source_count=3,
        validation_issue_count=2,
        details={"location": "pyproject.plugins", "issue_count": 2, "message": "ignored"},
    )

    validate_config_governance_event(resolve_payload)
    validate_config_governance_event(export_payload)
    validate_config_governance_event(failure_payload)

    with pytest.raises(ValidationError):
        validate_governance_event(resolve_payload)


def test_should_raise_validation_error_when_config_details_shape_is_invalid() -> None:
    resolve_with_details = {
        "schema_version": "1.0",
        "event_id": "25a35f7b-d612-4b6f-8f6e-f6d695f0f8fe",
        "event_name": "config.lifecycle",
        "event_type": "resolve",
        "profile_id": "default",
        "config_schema_version": "1",
        "strict": True,
        "source_count": 0,
        "validation_issue_count": 0,
        "timestamp": "2026-04-08T12:00:00+00:00",
        "details": {"diagnostic_only": True},
    }
    export_without_diagnostic_flag = {
        "schema_version": "1.0",
        "event_id": "6106f0aa-2e5d-4a01-a8e4-68a3d4bcbac8",
        "event_name": "config.lifecycle",
        "event_type": "export",
        "profile_id": "default",
        "config_schema_version": "1",
        "strict": True,
        "source_count": 1,
        "validation_issue_count": 0,
        "timestamp": "2026-04-08T12:00:01+00:00",
        "details": {},
    }
    validation_failure_with_extra_key = {
        "schema_version": "1.0",
        "event_id": "1ea42ce4-fb09-4d40-8ca8-129ca109c6ac",
        "event_name": "config.lifecycle",
        "event_type": "validation_failure",
        "profile_id": "default",
        "config_schema_version": "1",
        "strict": False,
        "source_count": 2,
        "validation_issue_count": 1,
        "timestamp": "2026-04-08T12:00:09+00:00",
        "details": {"location": "pyproject.plugins", "issue_count": 1, "extra": "forbidden"},
    }

    with pytest.raises(ValidationError):
        validate_config_governance_event(resolve_with_details)
    with pytest.raises(ValidationError):
        validate_config_governance_event(export_without_diagnostic_flag)
    with pytest.raises(ValidationError):
        validate_config_governance_event(validation_failure_with_extra_key)


def test_should_enforce_data_minimization_and_details_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = ConfigManager(
        env_snapshot={"CE_PLOT_STYLE": "legacy"},
        pyproject_snapshot=_EMPTY_PYPROJECT,
    )
    with caplog.at_level(logging.INFO, logger="calibrated_explanations.governance.config"):
        ConfigManager.from_sources(strict=False)
        manager.export_effective()
        with pytest.warns(UserWarning):
            ConfigManager(
                env_snapshot={},
                pyproject_snapshot={
                    "plugins": {"trusted": 123},
                    "explanations": {},
                    "intervals": {},
                    "plots": {},
                    "telemetry": {},
                },
                strict=False,
            )

    records = config_event_records(caplog)
    assert records
    for record in records:
        assert_minimization_constraints(event_payload(record), record.event_type)


def test_should_emit_independent_resolve_events_for_distinct_source_snapshots(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        "calibrated_explanations.core.config_manager.read_pyproject_section",
        lambda _path: {},
    )
    for key in _KNOWN_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    with caplog.at_level(logging.INFO, logger="calibrated_explanations.governance.config"):
        monkeypatch.setenv("CE_PLOT_STYLE", "legacy")
        ConfigManager.from_sources()
        monkeypatch.setenv("CE_PLOT_STYLE", "plot_spec.default")
        monkeypatch.setenv("CE_CACHE", "1")
        ConfigManager.from_sources()

    resolve_records = [
        record for record in config_event_records(caplog) if record.event_type == "resolve"
    ]
    assert len(resolve_records) == 2
    assert resolve_records[0].source_count == 1
    assert resolve_records[1].source_count == 2
    assert resolve_records[0].source_count != resolve_records[1].source_count


def test_should_raise_for_unsupported_plugin_governance_decision() -> None:
    from calibrated_explanations.governance.events import build_plugin_governance_event

    with pytest.raises(ValidationError, match="unsupported governance decision"):
        build_plugin_governance_event(
            decision="unsupported",
            identifier="tests.unsupported",
            provider="tests",
            source="manual",
            trusted=True,
            actor="tests",
        )


def test_should_raise_for_unsupported_config_governance_event_type() -> None:
    with pytest.raises(ValidationError, match="unsupported config governance event type"):
        build_config_governance_event(
            event_type="unsupported",
            profile_id="default",
            config_schema_version="1",
            strict=True,
            source_count=0,
            validation_issue_count=0,
        )


def valid_plugin_payload() -> dict:
    from datetime import datetime, timezone
    from uuid import uuid4
    return {
        "schema_version": "1.0",
        "event_id": str(uuid4()),
        "event_name": "plugin.registration.decision",
        "decision": "accepted_registration",
        "identifier": "tests.fallback",
        "provider": "tests",
        "source": "manual",
        "trusted": True,
        "actor": "tests",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def valid_config_payload() -> dict:
    from datetime import datetime, timezone
    from uuid import uuid4
    return {
        "schema_version": "1.0",
        "event_id": str(uuid4()),
        "event_name": "config.lifecycle",
        "event_type": "resolve",
        "profile_id": "default",
        "config_schema_version": "1",
        "strict": True,
        "source_count": 0,
        "validation_issue_count": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": None,
    }


def test_should_validate_plugin_event_via_fallback_when_jsonschema_is_none() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = valid_plugin_payload()
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        validate_governance_event(payload)
    assert payload["event_name"] == "plugin.registration.decision"


def test_should_raise_when_plugin_event_missing_required_key_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = valid_plugin_payload()
    del payload["decision"]
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="missing required field"):
            validate_governance_event(payload)


def test_should_raise_when_plugin_event_schema_version_wrong_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_plugin_payload(), "schema_version": "2.0"}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="schema_version"):
            validate_governance_event(payload)


def test_should_raise_when_plugin_event_decision_invalid_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_plugin_payload(), "decision": "not_a_decision"}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="decision is invalid"):
            validate_governance_event(payload)


def test_should_raise_when_plugin_event_trusted_not_bool_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_plugin_payload(), "trusted": "yes"}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="trusted must be a bool"):
            validate_governance_event(payload)


def test_should_raise_when_plugin_event_timestamp_not_string_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_plugin_payload(), "timestamp": 12345}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="timestamp must be an ISO-8601 string"):
            validate_governance_event(payload)


def test_should_raise_when_plugin_event_timestamp_invalid_iso_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_plugin_payload(), "timestamp": "not-a-date"}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="timestamp must be valid ISO-8601"):
            validate_governance_event(payload)


def test_should_validate_config_event_via_fallback_when_jsonschema_is_none() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = valid_config_payload()
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        validate_config_governance_event(payload)
    assert payload["event_type"] == "resolve"


def test_should_raise_when_config_event_missing_key_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = valid_config_payload()
    del payload["strict"]
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="missing required field"):
            validate_config_governance_event(payload)


def test_should_raise_when_config_event_schema_version_wrong_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_config_payload(), "schema_version": "2.0"}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="schema_version"):
            validate_config_governance_event(payload)


def test_should_raise_when_config_event_name_wrong_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_config_payload(), "event_name": "wrong.event"}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="event_name"):
            validate_config_governance_event(payload)


def test_should_raise_when_config_event_type_invalid_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_config_payload(), "event_type": "invalid_type"}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="event_type is invalid"):
            validate_config_governance_event(payload)


def test_should_raise_when_config_event_strict_not_bool_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_config_payload(), "strict": "yes"}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="strict must be a bool"):
            validate_config_governance_event(payload)


def test_should_raise_when_config_event_source_count_negative_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_config_payload(), "source_count": -1}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="source_count"):
            validate_config_governance_event(payload)


def test_should_raise_when_config_event_resolve_has_details_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_config_payload(), "event_type": "resolve", "details": {"extra": "data"}}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="resolve details must be null"):
            validate_config_governance_event(payload)


def test_should_validate_config_export_event_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_config_payload(), "event_type": "export", "details": {"diagnostic_only": True}}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        validate_config_governance_event(payload)
    assert payload["details"] == {"diagnostic_only": True}


def test_should_raise_when_config_export_details_wrong_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_config_payload(), "event_type": "export", "details": {}}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="export details"):
            validate_config_governance_event(payload)


def test_should_validate_config_validation_failure_event_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {
        **valid_config_payload(),
        "event_type": "validation_failure",
        "details": {"location": "pyproject.plugins", "issue_count": 1},
    }
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        validate_config_governance_event(payload)
    assert payload["details"]["issue_count"] == 1


def test_should_raise_when_validation_failure_details_not_mapping_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {
        **valid_config_payload(),
        "event_type": "validation_failure",
        "details": "not-a-mapping",
    }
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="details must be an object"):
            validate_config_governance_event(payload)


def test_should_raise_when_validation_failure_details_extra_key_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {
        **valid_config_payload(),
        "event_type": "validation_failure",
        "details": {"location": None, "issue_count": 1, "extra": "forbidden"},
    }
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="details keys"):
            validate_config_governance_event(payload)


def test_should_raise_when_validation_failure_issue_count_negative_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {
        **valid_config_payload(),
        "event_type": "validation_failure",
        "details": {"location": None, "issue_count": -1},
    }
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="issue_count"):
            validate_config_governance_event(payload)


def test_should_raise_when_validation_failure_location_not_string_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {
        **valid_config_payload(),
        "event_type": "validation_failure",
        "details": {"location": 42, "issue_count": 1},
    }
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="location must be a string or null"):
            validate_config_governance_event(payload)


def test_should_raise_when_config_event_timestamp_not_string_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_config_payload(), "timestamp": 12345}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="timestamp must be an ISO-8601 string"):
            validate_config_governance_event(payload)


def test_should_raise_when_config_event_timestamp_invalid_iso_in_fallback() -> None:
    import calibrated_explanations.governance.events as evt_module
    payload = {**valid_config_payload(), "timestamp": "not-a-date"}
    with unittest.mock.patch.object(evt_module, "jsonschema", None):
        with pytest.raises(ValidationError, match="timestamp must be valid ISO-8601"):
            validate_config_governance_event(payload)
