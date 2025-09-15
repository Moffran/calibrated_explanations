from typing import Any, Dict

from calibrated_explanations.plugins import registry


class DummyPlugin:
    plugin_meta: Dict[str, Any] = {
        "schema_version": 1,
        "capabilities": ["explain"],
        "name": "dummy",
    }

    def supports(self, model: Any) -> bool:
        return getattr(model, "is_dummy", False)

    def explain(self, model: Any, X: Any, **kwargs: Any) -> Any:
        return {"ok": True}


def test_register_and_find_for():
    registry.clear()
    p = DummyPlugin()
    registry.register(p)
    assert p in registry.list_plugins()

    class M:
        is_dummy = True

    found = registry.find_for(M())
    assert p in found


def test_register_validation():
    registry.clear()

    class Bad:
        plugin_meta = {"name": "bad", "capabilities": ["explain"]}  # missing schema_version

        def supports(self, model: Any) -> bool:
            return False

        def explain(self, model: Any, X: Any, **kwargs: Any) -> Any:
            return None

    try:
        registry.register(Bad())
    except ValueError as e:
        assert "schema_version" in str(e)
    else:
        assert False, "ValueError expected for missing schema_version"


def test_unregister():
    registry.clear()

    class P(DummyPlugin):
        pass

    p = P()
    registry.register(p)
    assert p in registry.list_plugins()
    registry.unregister(p)
    assert p not in registry.list_plugins()
