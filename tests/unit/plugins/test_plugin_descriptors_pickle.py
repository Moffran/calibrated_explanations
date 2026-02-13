import pickle
from types import MappingProxyType

from calibrated_explanations.plugins.registry import (
    ExplanationPluginDescriptor,
    IntervalPluginDescriptor,
    PlotBuilderDescriptor,
    PlotRendererDescriptor,
)


# Module-level dummy plugin class so instances are picklable
class DummyPlugin:
    pass


def _make_dummy_plugin():
    return DummyPlugin()


def test_plugin_descriptors_pickle_roundtrip():
    meta = {"a": 1, "b": [1, 2]}
    p = _make_dummy_plugin()
    desc = ExplanationPluginDescriptor(identifier="ex1", plugin=p, metadata=meta, trusted=True)
    dumped = pickle.dumps(desc)
    loaded = pickle.loads(dumped)
    assert loaded.identifier == "ex1"
    # After pickling the frozen dataclass metadata should be a plain dict
    assert isinstance(loaded.metadata, dict)

    desc2 = IntervalPluginDescriptor(identifier="int1", plugin=p, metadata=meta)
    assert isinstance(pickle.loads(pickle.dumps(desc2)).metadata, dict)

    desc3 = PlotBuilderDescriptor(identifier="pb1", builder=p, metadata=meta)
    assert isinstance(pickle.loads(pickle.dumps(desc3)).metadata, dict)

    desc4 = PlotRendererDescriptor(identifier="pr1", renderer=p, metadata=meta)
    assert isinstance(pickle.loads(pickle.dumps(desc4)).metadata, dict)
