"""Tests for the built-in FAST explanation plugin registration."""

from __future__ import annotations

from calibrated_explanations.plugins import explanations_fast
from calibrated_explanations.plugins.explanations_fast import register_fast_explanation_plugin

def test_should_skip_registration_when_descriptor_exists(monkeypatch):
    """Should not register when a descriptor already exists."""
    monkeypatch.setattr(explanations_fast, "find_explanation_descriptor", lambda _identifier: object())

    captured = []

    def fake_register(*args, **kwargs):
        captured.append((args, kwargs))

    monkeypatch.setattr(explanations_fast, "register_explanation_plugin", fake_register)

    register_fast_explanation_plugin()

    assert captured == []


def test_should_register_builtin_when_missing(monkeypatch):
    """Registers the built-in plugin when none is present."""
    monkeypatch.setattr(explanations_fast, "find_explanation_descriptor", lambda _identifier: None)

    captured: dict[str, object] = {}

    def fake_register(identifier, plugin, source):
        captured["identifier"] = identifier
        captured["plugin"] = plugin
        captured["source"] = source

    monkeypatch.setattr(explanations_fast, "register_explanation_plugin", fake_register)

    register_fast_explanation_plugin()

    assert captured["identifier"] == "core.explanation.fast"
    assert captured["source"] == "builtin"
    plugin = captured["plugin"]
    assert plugin.plugin_meta["name"] == "core.explanation.fast"
    assert plugin.explanation_attr == "explain_fast"
    assert getattr(plugin, "plugin_meta")["interval_dependency"] == "core.interval.fast"
