from __future__ import annotations

from calibrated_explanations.plugins import builtins, explanations_fast


def test_register_fast_explanation_skips_when_descriptor_present(monkeypatch):
    monkeypatch.setattr(explanations_fast, "find_explanation_descriptor", lambda name: object())
    monkeypatch.setattr(explanations_fast, "register_explanation_plugin", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not register")))

    explanations_fast.register_fast_explanation_plugin()


def test_register_fast_explanation_builds_plugin(monkeypatch):
    monkeypatch.setattr(explanations_fast, "find_explanation_descriptor", lambda name: None)
    captured = []

    def stub_register(name, plugin, source=""):
        captured.append((name, plugin, source))

    monkeypatch.setattr(explanations_fast, "register_explanation_plugin", stub_register)

    class DummyLegacy:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(builtins, "_LegacyExplanationBase", DummyLegacy)

    explanations_fast.register_fast_explanation_plugin()

    assert captured
    assert captured[0][0] == "core.explanation.fast"
