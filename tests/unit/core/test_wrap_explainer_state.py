from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer


class DummyLearner:
    def fit(self, x, y=None, **_):
        return self


def test___getstate_converts_mappingproxy_nested() -> None:
    """Ensure __getstate__ converts MappingProxyType instances to plain dicts.

    This covers the recursive conversion branch in `WrapCalibratedExplainer.__getstate__`.
    """
    w = WrapCalibratedExplainer(DummyLearner())
    # Attach mappingproxy attributes that should be converted
    w.some_mapping = MappingProxyType({"a": 1})
    w.nested = MappingProxyType({"inner": MappingProxyType({"b": 2})})
    # Also include a list/tuple/set container of mappingproxy to hit collection branch
    w.container = [MappingProxyType({"x": 3}), (MappingProxyType({"y": 4}),)]

    state = w.__getstate__()

    assert isinstance(state["some_mapping"], dict)
    assert state["some_mapping"]["a"] == 1

    assert isinstance(state["nested"], dict)
    assert isinstance(state["nested"]["inner"], Mapping)
    assert state["nested"]["inner"]["b"] == 2

    # container should preserve structure and convert mapping proxies inside
    assert isinstance(state["container"], list)
    assert isinstance(state["container"][0], dict)
    assert state["container"][0]["x"] == 3
