import numpy as np
from types import SimpleNamespace

import pytest

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer


def test_serialise_preprocessor_value_various_types():
    from types import SimpleNamespace

    # Provide a minimal 'fitted' learner so wrapper initializer proceeds
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    assert w.serialise_preprocessor_value(None) is None
    assert w.serialise_preprocessor_value({"a": 1}) == {"a": 1}
    # sets become lists
    out = w.serialise_preprocessor_value({"s": {1, 2}})
    assert isinstance(out["s"], list)
    arr = np.array([1, 2, 3])
    assert w.serialise_preprocessor_value(arr) == [1, 2, 3]


def test_extract_preprocessor_snapshot_and_build_metadata():
    class DummyTransformer:
        pass

    class Pre:
        def get_mapping_snapshot(self):
            return {"m": 1}

        categories_ = [["a"]]

        transformers_ = [("t", DummyTransformer(), [0])]

        def get_feature_names_out(self):
            return ["f0"]

        mapping_ = {"x": 1}

    pre = Pre()
    from types import SimpleNamespace
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    w.auto_encode = True
    w.preprocessor = pre
    snap = w.extract_preprocessor_snapshot(pre)
    assert "custom" in snap or "categories" in snap
    meta = w.build_preprocessor_metadata()
    assert meta is not None
    assert "transformer_id" in meta


def test_format_proba_output_variants():
    from types import SimpleNamespace
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    multiclass = np.ones((2, 3)) * 0.3
    out = w.format_proba_output(multiclass, uq_interval=True)
    assert isinstance(out, tuple) and len(out) == 2

    binary = np.array([[0.2, 0.8], [0.4, 0.6]])
    outb = w.format_proba_output(binary, uq_interval=True)
    assert isinstance(outb[1][0], np.ndarray)


def test_normalize_public_kwargs_and_import_mapping_stash(monkeypatch):
    from types import SimpleNamespace
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    with pytest.warns(DeprecationWarning):
        res = w.normalize_public_kwargs({"alpha": 0.1, "foo": 2})
    # alias 'alpha' should be removed by the normaliser
    assert "alpha" not in res
    assert res.get("foo") == 2

    # import_preprocessor_mapping should warn when mapping cannot be applied
    mapping = {"a": 1}
    with pytest.warns(UserWarning):
        w.import_preprocessor_mapping(mapping)
    # We do not introspect private stash attributes; only ensure a warning was raised.
