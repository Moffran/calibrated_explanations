from calibrated_explanations.cache import LRUCache, make_key
from calibrated_explanations.plugins import registry
from calibrated_explanations.viz import (
    PlotSpec,
    IntervalHeaderSpec,
    BarHPanelSpec,
    BarItem,
)


def test_lru_cache_eviction_and_get():
    cache = LRUCache(
        namespace="unit",
        version="v1",
        max_items=3,
        max_bytes=None,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=lambda _: 1,
    )
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
    k1 = make_key("unit", "v1", [1, "a", 3])
    k2 = make_key("unit", "v1", [1, "a", 3])
    assert k1 == k2
    assert isinstance(k1, tuple)


class FakePlugin:
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

    def explain(self, model, x, **kwargs):
        return {"explained": True}


def test_plugin_registry_register_and_find():
    registry.clear()
    p = FakePlugin()
    registry.register(p)
    assert p in registry.list_plugins()
    # find_for should return our plugin for any model
    found = registry.find_for(object())
    assert any(isinstance(x, FakePlugin.__class__) or x is p for x in found)
    registry.untrust_plugin(p)  # safe if not trusted
    registry.unregister(p)
    assert p not in registry.list_plugins()


def test_plotspec_dataclasses_roundtrip__should_preserve_structure_and_values():
    """Verify that PlotSpec correctly assembles header, body, and metadata.

    Domain Invariants:
    - Title must be preserved exactly (string identity)
    - Header prediction must be within bounds (invariant: low ≤ pred ≤ high)
    - Body bars must be indexed consistently
    Ref: ADR-005 Explanation Envelope, ADR-007 PlotSpec Abstraction
    """
    header = IntervalHeaderSpec(pred=0.5, low=0.1, high=0.9)
    bars = [BarItem(label="f1", value=0.2), BarItem(label="f2", value=-0.1)]
    body = BarHPanelSpec(bars=bars)
    spec = PlotSpec(title="t", header=header, body=body)

    # Domain invariant: title is preserved exactly
    assert spec.title == "t", "Title must be preserved as-is"

    # Domain invariants: header bounds ordering (low ≤ pred ≤ high)
    assert spec.header.pred == 0.5
    assert spec.header.low <= spec.header.pred <= spec.header.high, (
        f"Point estimate ({spec.header.pred}) must lie within "
        f"bounds [{spec.header.low}, {spec.header.high}]"
    )

    # Domain invariant: body contains correct number of bars
    assert len(spec.body.bars) == 2, "Body must contain exactly 2 bars"

    # Domain invariants: each bar has required fields and valid structure
    for bar_idx, item in enumerate(spec.body.bars):
        assert item.label is not None, f"Bar {bar_idx} must have a label"
        assert isinstance(item.value, (int, float)), f"Bar {bar_idx} value must be numeric"
