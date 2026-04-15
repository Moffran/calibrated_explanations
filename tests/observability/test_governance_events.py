from __future__ import annotations

import logging

import numpy as np
import pytest

from calibrated_explanations.core.explain._feature_filter import (
    FeatureFilterConfig,
    compute_filtered_features_to_ignore,
    emit_feature_filter_governance_event,
)
from calibrated_explanations.governance.events import validate_governance_event
from calibrated_explanations.logging import logging_context
from calibrated_explanations.plugins import registry
from calibrated_explanations.utils.exceptions import ValidationError
from tests.support.registry_helpers import (
    clear_env_trust_cache,
    clear_explanation_plugins,
    clear_interval_plugins,
    clear_plot_plugins,
    clear_trust_warnings,
)


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch: pytest.MonkeyPatch):
    registry.clear()
    clear_explanation_plugins()
    clear_interval_plugins()
    clear_plot_plugins()
    clear_env_trust_cache()
    clear_trust_warnings()
    monkeypatch.setattr(registry, "ensure_builtin_plugins", lambda: None, raising=False)
    monkeypatch.delenv("CE_TRUST_PLUGIN", raising=False)
    monkeypatch.delenv("CE_DENY_PLUGIN", raising=False)
    yield


def base_meta(**extra: object) -> dict[str, object]:
    meta: dict[str, object] = {
        "schema_version": 1,
        "name": "tests.meta",
        "version": "0.0-test",
        "provider": "tests",
        "capabilities": ["explain"],
        "data_modalities": ("tabular",),
    }
    meta.update(extra)
    return meta


def decision_records(caplog: pytest.LogCaptureFixture, decision: str):
    records = [record for record in caplog.records if getattr(record, "decision", None) == decision]
    assert records
    return records


def test_register_emits_schema_valid_accepted_registration_event(caplog):
    class Plugin:
        plugin_meta = base_meta(name="tests.accepted")

    with (
        logging_context(request_id="req-accepted", tenant_id="tenant-a"),
        caplog.at_level("INFO", logger="calibrated_explanations.governance.plugins"),
        pytest.warns(DeprecationWarning, match="register\\(\\) is deprecated"),
    ):
        registry.register(Plugin(), source="manual")

    record = decision_records(caplog, "accepted_registration")[-1]
    payload = {key: getattr(record, key) for key in record.__dict__}
    validate_governance_event(payload)
    assert record.identifier == "tests.accepted"
    assert record.actor == "_log_plugin_registration_event"
    assert record.request_id == "req-accepted"
    assert record.tenant_id == "tenant-a"


def test_entrypoint_discovery_emits_schema_valid_skipped_untrusted_event(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    class Plugin:
        plugin_meta = base_meta(name="tests.entry.untrusted")

    class EntryPoint:
        name = "tests.entry.untrusted"
        module = "tests_plugins_entry_untrusted"
        attr = "Plugin"
        dist = type("Dist", (), {"name": "tests-provider"})()

        def load(self):
            return Plugin()

    class EntryPoints:
        def select(self, *, group):
            return [EntryPoint()] if group == "calibrated_explanations.plugins" else []

    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: EntryPoints())

    with caplog.at_level("INFO", logger="calibrated_explanations.governance.plugins"):
        loaded = registry.load_entrypoint_plugins(include_untrusted=False)

    assert loaded == ()
    record = decision_records(caplog, "skipped_untrusted")[-1]
    payload = {key: getattr(record, key) for key in record.__dict__}
    validate_governance_event(payload)
    assert record.identifier == "tests.entry.untrusted"
    assert record.source == "entrypoint"


def test_entrypoint_discovery_emits_event_on_repeated_skipped_untrusted_decisions(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    class Plugin:
        plugin_meta = base_meta(name="tests.entry.untrusted.repeat")

    class EntryPoint:
        name = "tests.entry.untrusted.repeat"
        module = "tests_plugins_entry_untrusted_repeat"
        attr = "Plugin"
        dist = type("Dist", (), {"name": "tests-provider"})()

        def load(self):
            return Plugin()

    class EntryPoints:
        def select(self, *, group):
            return [EntryPoint()] if group == "calibrated_explanations.plugins" else []

    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: EntryPoints())

    with caplog.at_level("INFO", logger="calibrated_explanations.governance.plugins"):
        first = registry.load_entrypoint_plugins(include_untrusted=False)
        second = registry.load_entrypoint_plugins(include_untrusted=False)

    assert first == ()
    assert second == ()
    records = decision_records(caplog, "skipped_untrusted")
    assert len(records) == 2
    for record in records:
        payload = {key: getattr(record, key) for key in record.__dict__}
        validate_governance_event(payload)
        assert record.identifier == "tests.entry.untrusted.repeat"
        assert record.source == "entrypoint"


def test_entrypoint_discovery_emits_schema_valid_skipped_denied_event(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    class Plugin:
        plugin_meta = base_meta(name="tests.entry.denied")

    class EntryPoint:
        name = "tests.entry.denied.module"
        module = "tests_plugins_entry_denied"
        attr = "Plugin"
        dist = type("Dist", (), {"name": "tests-provider"})()

        def load(self):
            return Plugin()

    class EntryPoints:
        def select(self, *, group):
            return [EntryPoint()] if group == "calibrated_explanations.plugins" else []

    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: EntryPoints())
    monkeypatch.setenv("CE_DENY_PLUGIN", "tests.entry.denied")

    with caplog.at_level("INFO", logger="calibrated_explanations.governance.plugins"):
        loaded = registry.load_entrypoint_plugins(include_untrusted=False)

    assert loaded == ()
    record = decision_records(caplog, "skipped_denied")[-1]
    payload = {key: getattr(record, key) for key in record.__dict__}
    validate_governance_event(payload)
    assert record.identifier == "tests.entry.denied"
    assert record.reason_code == "denylist"


def test_entrypoint_discovery_emits_schema_valid_checksum_failure_event(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    class Plugin:
        plugin_meta = base_meta(
            name="tests.entry.checksum",
            checksum={"sha256": "deadbeef"},
        )

    class EntryPoint:
        name = "tests.entry.checksum"
        module = "tests_plugins_entry_checksum"
        attr = "Plugin"
        dist = type("Dist", (), {"name": "tests-provider"})()

        def load(self):
            return Plugin()

    class EntryPoints:
        def select(self, *, group):
            return [EntryPoint()] if group == "calibrated_explanations.plugins" else []

    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: EntryPoints())
    with caplog.at_level("INFO", logger="calibrated_explanations.governance.plugins"):
        loaded = registry.load_entrypoint_plugins(include_untrusted=True)

    assert loaded == ()
    record = decision_records(caplog, "checksum_failure")[-1]
    payload = {key: getattr(record, key) for key in record.__dict__}
    validate_governance_event(payload)
    assert record.identifier == "tests_plugins_entry_checksum:Plugin"
    assert record.reason_code == "checksum_validation_failed"


def test_register_emits_schema_valid_denied_registration_event(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    class Plugin:
        plugin_meta = base_meta(name="tests.manual.denied")

    monkeypatch.setenv("CE_DENY_PLUGIN", "tests.manual.denied")

    with (
        caplog.at_level("INFO", logger="calibrated_explanations.governance.plugins"),
        pytest.warns(DeprecationWarning, match="register\\(\\) is deprecated"),
        pytest.raises(ValidationError),
    ):
        registry.register(Plugin(), source="register_call")

    record = decision_records(caplog, "denied_registration")[-1]
    payload = {key: getattr(record, key) for key in record.__dict__}
    validate_governance_event(payload)
    assert record.identifier == "tests.manual.denied"
    assert record.source == "register_call"


def test_governance_events_are_side_effect_only_and_payload_safe(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    class Plugin:
        plugin_meta = base_meta(name="tests.side_effect.safe")

    # Baseline behavior without active caplog capture.
    with pytest.warns(DeprecationWarning, match="register\\(\\) is deprecated"):
        registry.register(Plugin(), source="manual")
    baseline_plugins = registry.list_plugins(include_untrusted=True)
    baseline_plugin_names = tuple(
        getattr(plugin, "plugin_meta", {}).get("name") for plugin in baseline_plugins
    )
    assert "tests.side_effect.safe" in baseline_plugin_names

    # Reset and run under governance log capture.
    registry.clear()
    clear_explanation_plugins()
    clear_interval_plugins()
    clear_plot_plugins()
    clear_env_trust_cache()
    clear_trust_warnings()
    monkeypatch.setattr(registry, "ensure_builtin_plugins", lambda: None, raising=False)

    with caplog.at_level("INFO", logger="calibrated_explanations.governance.plugins"):
        with pytest.warns(DeprecationWarning, match="register\\(\\) is deprecated"):
            registry.register(Plugin(), source="manual")

    captured_plugins = registry.list_plugins(include_untrusted=True)
    captured_plugin_names = tuple(
        getattr(plugin, "plugin_meta", {}).get("name") for plugin in captured_plugins
    )
    assert captured_plugin_names == baseline_plugin_names

    record = decision_records(caplog, "accepted_registration")[-1]
    payload = {key: getattr(record, key) for key in record.__dict__}
    validate_governance_event(payload)


def test_should_emit_feature_filter_governance_record_with_context_when_event_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        logging_context(request_id="req-feature-filter", tenant_id="tenant-a", mode="factual"),
        caplog.at_level(logging.INFO, logger="calibrated_explanations.governance.feature_filter"),
    ):
        emit_feature_filter_governance_event(
            decision="filter_skipped",
            reason="unit-test reason",
            strict=True,
            mode="factual",
        )

    record = caplog.records[-1]
    assert record.name == "calibrated_explanations.governance.feature_filter"
    assert record.decision == "filter_skipped"
    assert record.reason == "unit-test reason"
    assert record.strict_observability is True
    assert record.mode == "factual"
    assert record.request_id == "req-feature-filter"
    assert record.tenant_id == "tenant-a"


def test_should_emit_operational_and_governance_feature_filter_records_when_strict_path_triggers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ExplanationStub:
        def __init__(self, feature_weights: object) -> None:
            self.feature_weights = feature_weights

    class CollectionStub:
        def __init__(self, explanations: list[object]) -> None:
            self.explanations = explanations

    collection = CollectionStub([ExplanationStub({"predict": None})])
    config = FeatureFilterConfig(enabled=True, per_instance_top_k=2, strict_observability=True)

    with (
        logging_context(request_id="req-strict", tenant_id="tenant-b"),
        caplog.at_level(logging.DEBUG),
    ):
        result = compute_filtered_features_to_ignore(
            collection,
            num_features=None,
            base_ignore=np.asarray([0], dtype=int),
            config=config,
        )

    assert result.global_ignore.tolist() == [0]

    operational_records = [
        record
        for record in caplog.records
        if record.name == "calibrated_explanations.core.explain.feature_filter"
    ]
    assert operational_records
    operational_record = operational_records[-1]
    assert operational_record.levelno == logging.WARNING
    assert operational_record.inferred == 0
    assert operational_record.provided == 0
    assert "unable to infer feature count" in operational_record.getMessage()

    governance_records = [
        record
        for record in caplog.records
        if record.name == "calibrated_explanations.governance.feature_filter"
    ]
    assert governance_records
    governance_record = governance_records[-1]
    assert governance_record.levelno == logging.WARNING
    assert governance_record.decision == "feature_filter_missing_feature_count"
    assert governance_record.strict_observability is True
    assert governance_record.request_id == "req-strict"
    assert governance_record.tenant_id == "tenant-b"
    assert governance_record.num_instances == 1

    # Ensure event payload is audit-focused and does not include sensitive/raw data blobs.
    payload = {key: getattr(governance_record, key) for key in governance_record.__dict__}
    forbidden_keys = {"features", "labels", "prediction", "plugin_meta", "raw_input", "X", "y"}
    assert forbidden_keys.isdisjoint(payload.keys())
