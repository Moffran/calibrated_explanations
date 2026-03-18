from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from calibrated_explanations.plugins import registry
from tests.support.registry_helpers import (
    clear_explanation_plugins,
    clear_interval_plugins,
    clear_plot_plugins,
)


class _ExplanationPlugin:
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


def _assert_renderer_invariants() -> None:
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
    registry.set_plot_renderer("tests.renderer.atomic", descriptor, trusted=False)

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
    _assert_renderer_invariants()


def test_should_preserve_idempotent_state_on_repeated_trust_calls() -> None:
    descriptor = registry.PlotRendererDescriptor(
        identifier="tests.renderer.idempotent",
        renderer=object(),
        metadata={"name": "tests.renderer.idempotent", "trust": {"trusted": False}},
        trusted=False,
        source="manual",
    )
    registry.set_plot_renderer("tests.renderer.idempotent", descriptor, trusted=False)

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
        registry.set_plot_renderer(identifier, descriptor, trusted=False)

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

    _assert_renderer_invariants()
    for identifier in ids:
        final = registry.find_plot_renderer_descriptor(identifier)
        assert final is not None
        assert final.trusted is True


def test_should_sync_descriptor_trust_state_when_using_legacy_trust_api(caplog) -> None:
    plugin = _ExplanationPlugin()
    registry.register_explanation_plugin("tests.trust.atomic.explanation", plugin)

    descriptor = registry.find_explanation_descriptor("tests.trust.atomic.explanation")
    assert descriptor is not None
    assert descriptor.trusted is False

    with caplog.at_level("INFO", logger="calibrated_explanations.governance.registry"):
        registry.trust_plugin(plugin)
        registry.untrust_plugin(plugin)

    descriptor_after = registry.find_explanation_descriptor("tests.trust.atomic.explanation")
    assert descriptor_after is not None
    assert descriptor_after.trusted is False
    assert any(
        getattr(record, "event_name", None) == "trust.mutation"
        for record in caplog.records
    )
