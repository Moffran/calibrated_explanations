from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from calibrated_explanations.plugins import registry
from calibrated_explanations.plugins._trust import update_trusted_identifier
from calibrated_explanations.utils.exceptions import ValidationError
from tests.support.registry_helpers import (
    clear_explanation_plugins,
    clear_interval_plugins,
    clear_plot_plugins,
    set_plot_builder,
    set_plot_renderer,
)


class ExplanationPluginStub:
    plugin_meta = {
        "schema_version": 1,
        "name": "tests.trust.atomic.explanation",
        "version": "0.1",
        "provider": "tests",
        "capabilities": ("explain", "task:classification"),
        "modes": ("factual",),
        "tasks": ("classification",),
        "trust": {"trusted": False},
        "dependencies": (),
    }

    def supports(self, _model):
        return True

    def explain(self, *_args, **_kwargs):
        return {}


class IntervalPluginStub:
    plugin_meta = {
        "schema_version": 1,
        "name": "tests.trust.atomic.interval",
        "version": "0.1",
        "provider": "tests",
        "capabilities": ("interval",),
        "modes": ("classification",),
        "dependencies": (),
        "fast_compatible": True,
        "requires_bins": False,
        "confidence_source": "posterior",
        "trust": {"trusted": False},
    }

    def supports(self, _model):
        return True

    def calibrate(self, *_args, **_kwargs):
        return {}


def assert_catalog_invariants() -> None:
    for descriptor in registry.list_explanation_descriptors(trusted_only=False):
        trusted_ids = {
            item.identifier for item in registry.list_explanation_descriptors(trusted_only=True)
        }
        assert descriptor.trusted is (descriptor.identifier in trusted_ids)
    for descriptor in registry.list_interval_descriptors(trusted_only=False):
        trusted_ids = {
            item.identifier for item in registry.list_interval_descriptors(trusted_only=True)
        }
        assert descriptor.trusted is (descriptor.identifier in trusted_ids)
    for descriptor in registry.list_plot_builder_descriptors(trusted_only=False):
        trusted_ids = {
            item.identifier for item in registry.list_plot_builder_descriptors(trusted_only=True)
        }
        assert descriptor.trusted is (descriptor.identifier in trusted_ids)
    for descriptor in registry.list_plot_renderer_descriptors(trusted_only=False):
        trusted_ids = {
            item.identifier for item in registry.list_plot_renderer_descriptors(trusted_only=True)
        }
        assert descriptor.trusted is (descriptor.identifier in trusted_ids)


def assert_renderer_invariants() -> None:
    descriptors = registry.list_plot_renderer_descriptors(trusted_only=False)
    trusted_ids = {
        descriptor.identifier
        for descriptor in registry.list_plot_renderer_descriptors(trusted_only=True)
    }
    for descriptor in descriptors:
        assert descriptor.trusted is (descriptor.identifier in trusted_ids)


@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch):
    registry.clear()
    clear_explanation_plugins()
    clear_interval_plugins()
    clear_plot_plugins()
    monkeypatch.setattr(registry, "ensure_builtin_plugins", lambda: None)
    yield
    registry.clear()
    clear_explanation_plugins()
    clear_interval_plugins()
    clear_plot_plugins()


def test_should_keep_renderer_trust_state_consistent_when_toggled_concurrently() -> None:
    descriptor = registry.PlotRendererDescriptor(
        identifier="tests.renderer.atomic",
        renderer=object(),
        metadata={"name": "tests.renderer.atomic", "trust": {"trusted": False}},
        trusted=False,
        source="manual",
    )
    set_plot_renderer("tests.renderer.atomic", descriptor, trusted=False)

    def _toggle_many() -> None:
        for _ in range(60):
            registry.mark_plot_renderer_trusted("tests.renderer.atomic")
            registry.mark_plot_renderer_untrusted("tests.renderer.atomic")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_toggle_many) for _ in range(8)]
        for future in futures:
            future.result()

    registry.mark_plot_renderer_trusted("tests.renderer.atomic")
    final_descriptor = registry.find_plot_renderer_descriptor("tests.renderer.atomic")
    assert final_descriptor is not None
    assert final_descriptor.trusted is True
    assert_renderer_invariants()


def test_should_preserve_idempotent_state_on_repeated_trust_calls() -> None:
    descriptor = registry.PlotRendererDescriptor(
        identifier="tests.renderer.idempotent",
        renderer=object(),
        metadata={"name": "tests.renderer.idempotent", "trust": {"trusted": False}},
        trusted=False,
        source="manual",
    )
    set_plot_renderer("tests.renderer.idempotent", descriptor, trusted=False)

    first = registry.mark_plot_renderer_trusted("tests.renderer.idempotent")
    second = registry.mark_plot_renderer_trusted("tests.renderer.idempotent")

    assert first.trusted is True
    assert second.trusted is True
    trusted_ids = {
        item.identifier for item in registry.list_plot_renderer_descriptors(trusted_only=True)
    }
    assert trusted_ids == {"tests.renderer.idempotent"}


def test_should_hold_invariants_when_two_identifiers_toggled_concurrently() -> None:
    ids = ["tests.renderer.race.alpha", "tests.renderer.race.beta"]
    for identifier in ids:
        descriptor = registry.PlotRendererDescriptor(
            identifier=identifier,
            renderer=object(),
            metadata={"name": identifier, "trust": {"trusted": False}},
            trusted=False,
            source="manual",
        )
        set_plot_renderer(identifier, descriptor, trusted=False)

    def _toggle_alpha() -> None:
        for _ in range(60):
            registry.mark_plot_renderer_trusted("tests.renderer.race.alpha")
            registry.mark_plot_renderer_untrusted("tests.renderer.race.alpha")

    def _toggle_beta() -> None:
        for _ in range(60):
            registry.mark_plot_renderer_trusted("tests.renderer.race.beta")
            registry.mark_plot_renderer_untrusted("tests.renderer.race.beta")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_toggle_alpha if i % 2 == 0 else _toggle_beta) for i in range(8)]
        for future in futures:
            future.result()

    for identifier in ids:
        registry.mark_plot_renderer_trusted(identifier)

    assert_renderer_invariants()
    for identifier in ids:
        final = registry.find_plot_renderer_descriptor(identifier)
        assert final is not None
        assert final.trusted is True


def test_should_sync_descriptor_trust_state_when_using_legacy_trust_api(caplog) -> None:
    plugin = ExplanationPluginStub()
    registry.register_explanation_plugin("tests.trust.atomic.explanation", plugin)

    descriptor = registry.find_explanation_descriptor("tests.trust.atomic.explanation")
    assert descriptor is not None
    assert descriptor.trusted is False

    with caplog.at_level("INFO", logger="calibrated_explanations.governance.registry"):
        with pytest.warns(DeprecationWarning, match="trust_plugin\\(\\) is deprecated"):
            registry.trust_plugin(plugin)
        registry.untrust_plugin(plugin)

    descriptor_after = registry.find_explanation_descriptor("tests.trust.atomic.explanation")
    assert descriptor_after is not None
    assert descriptor_after.trusted is False
    assert any(getattr(record, "event_name", None) == "trust.mutation" for record in caplog.records)


def test_should_machine_check_invariants_for_all_plugin_kinds() -> None:
    explanation = ExplanationPluginStub()
    interval = IntervalPluginStub()
    registry.register_explanation_plugin("tests.trust.atomic.explanation", explanation)
    registry.register_interval_plugin("tests.trust.atomic.interval", interval)

    builder_descriptor = registry.PlotBuilderDescriptor(
        identifier="tests.builder.atomic",
        builder=object(),
        metadata={"name": "tests.builder.atomic", "trust": {"trusted": False}},
        trusted=False,
        source="manual",
    )
    renderer_descriptor = registry.PlotRendererDescriptor(
        identifier="tests.renderer.atomic.catalog",
        renderer=object(),
        metadata={"name": "tests.renderer.atomic.catalog", "trust": {"trusted": False}},
        trusted=False,
        source="manual",
    )
    set_plot_builder("tests.builder.atomic", builder_descriptor, trusted=False)
    set_plot_renderer("tests.renderer.atomic.catalog", renderer_descriptor, trusted=False)

    registry.mark_explanation_trusted("tests.trust.atomic.explanation")
    registry.mark_interval_trusted("tests.trust.atomic.interval")
    registry.mark_plot_builder_trusted("tests.builder.atomic")
    registry.mark_plot_renderer_trusted("tests.renderer.atomic.catalog")
    assert_catalog_invariants()

    registry.mark_explanation_untrusted("tests.trust.atomic.explanation")
    registry.mark_interval_untrusted("tests.trust.atomic.interval")
    registry.mark_plot_builder_untrusted("tests.builder.atomic")
    registry.mark_plot_renderer_untrusted("tests.renderer.atomic.catalog")
    assert_catalog_invariants()


def test_should_emit_structured_trust_mutation_event_for_each_mutation_path(caplog) -> None:
    explanation = ExplanationPluginStub()
    interval = IntervalPluginStub()
    registry.register_explanation_plugin("tests.trust.mutation.explanation", explanation)
    registry.register_interval_plugin("tests.trust.mutation.interval", interval)

    builder_descriptor = registry.PlotBuilderDescriptor(
        identifier="tests.builder.mutation",
        builder=object(),
        metadata={"name": "tests.builder.mutation", "trust": {"trusted": False}},
        trusted=False,
        source="manual",
    )
    renderer_descriptor = registry.PlotRendererDescriptor(
        identifier="tests.renderer.mutation",
        renderer=object(),
        metadata={"name": "tests.renderer.mutation", "trust": {"trusted": False}},
        trusted=False,
        source="manual",
    )
    set_plot_builder("tests.builder.mutation", builder_descriptor, trusted=False)
    set_plot_renderer("tests.renderer.mutation", renderer_descriptor, trusted=False)

    with caplog.at_level("INFO", logger="calibrated_explanations.governance.registry"):
        registry.mark_explanation_trusted("tests.trust.mutation.explanation")
        registry.mark_interval_trusted("tests.trust.mutation.interval")
        registry.mark_plot_builder_trusted("tests.builder.mutation")
        registry.mark_plot_renderer_trusted("tests.renderer.mutation")
        registry.reset_plugin_catalog(kind="plot")

    records = [
        record
        for record in caplog.records
        if getattr(record, "event_name", None) == "trust.mutation"
    ]
    assert len(records) >= 5
    for record in records:
        assert isinstance(getattr(record, "identifier", None), str)
        assert isinstance(getattr(record, "trusted", None), bool)
        assert isinstance(getattr(record, "kind", None), str)
        assert isinstance(getattr(record, "source", None), str)
        assert isinstance(getattr(record, "actor", None), str)


def test_should_raise_when_trusted_identifier_helper_called_outside_atomic_context() -> None:
    with pytest.raises(ValidationError, match="outside mutate_trust_atomic context"):
        update_trusted_identifier(set(), "tests.illegal.mutation", True)
