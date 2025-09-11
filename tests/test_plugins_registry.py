import types

import pytest

from calibrated_explanations.plugins import registry


class DummyPlugin:
    plugin_meta = {"schema_version": 1, "capabilities": ["explain"], "name": "dummy"}

    def supports(self, model):
        return getattr(model, "is_dummy", False)

    def explain(self, model, X, **kwargs):
        return {"explained": True}


def test_register_and_trust_flow(tmp_path):
    p = DummyPlugin()
    # ensure clean start
    registry.clear()
    registry.register(p)
    assert p in registry.list_plugins()

    # trusting unregistered plugin raises
    with pytest.raises(ValueError):
        registry.trust_plugin(object())

    # trust and find
    registry.trust_plugin(p)
    trusted = registry.find_for_trusted(types.SimpleNamespace(is_dummy=True))
    assert p in trusted

    # untrust works
    registry.untrust_plugin(p)
    trusted2 = registry.find_for_trusted(types.SimpleNamespace(is_dummy=True))
    assert p not in trusted2

    # cleanup
    registry.unregister(p)
