from calibrated_explanations.perf.cache import LRUCache, make_key
from calibrated_explanations.plugins import registry
from calibrated_explanations.viz.plotspec import (
    PlotSpec,
    IntervalHeaderSpec,
    BarHPanelSpec,
    BarItem,
)


def test_lru_cache_eviction_and_get():
    cache = LRUCache(max_items=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    # access a to make it recently used
    assert cache.get("a") == 1
    cache.set("d", 4)
    # 'b' should have been evicted (oldest)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4


def test_make_key_is_deterministic():
    k1 = make_key([1, "a", 3])
    k2 = make_key([1, "a", 3])
    assert k1 == k2
    assert isinstance(k1, tuple)


class _FakePlugin:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["explain"],
        "name": "fake",
        "version": "0.0-test",
        "provider": "tests",
        "trusted": False,
        "trust": False,
    }

    def supports(self, model):
        return True

    def explain(self, model, X, **kwargs):
        return {"explained": True}


def test_plugin_registry_register_and_find():
    registry.clear()
    p = _FakePlugin()
    registry.register(p)
    assert p in registry.list_plugins()
    # find_for should return our plugin for any model
    found = registry.find_for(object())
    assert any(isinstance(x, _FakePlugin.__class__) or x is p for x in found)
    registry.untrust_plugin(p)  # safe if not trusted
    registry.unregister(p)
    assert p not in registry.list_plugins()


def test_plotspec_dataclasses_roundtrip():
    header = IntervalHeaderSpec(pred=0.5, low=0.1, high=0.9)
    bars = [BarItem(label="f1", value=0.2), BarItem(label="f2", value=-0.1)]
    body = BarHPanelSpec(bars=bars)
    spec = PlotSpec(title="t", header=header, body=body)
    assert spec.title == "t"
    assert spec.header.pred == 0.5
    assert len(spec.body.bars) == 2
