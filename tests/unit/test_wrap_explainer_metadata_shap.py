import pytest

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from tests.helpers.deprecation import deprecations_error_enabled, warns_or_raises


class DummyExplainer:
    def __init__(self, initial_meta=None):
        self.preprocessor_metadata = initial_meta

    def explain_shap(self, x, **kwargs):
        return {"called": True, "x": list(x) if hasattr(x, "__iter__") else x, "kwargs": kwargs}

    def set_preprocessor_metadata(self, metadata):
        self.preprocessor_metadata = metadata


def make_wrapper_with_dummy(dummy=None):
    class FakeLearner:
        def __init__(self):
            self.fitted = True

        def fit(self, *a, **k):
            self.fitted = True

    w = WrapCalibratedExplainer(FakeLearner())
    # mark as fitted & calibrated and inject dummy explainer
    w.fitted = True
    w.calibrated = True
    if dummy is None:
        dummy = DummyExplainer()
    w.explainer = dummy
    return w, dummy


def test_explain_shap_delegates_to_explainer():
    w, dummy = make_wrapper_with_dummy()
    with warns_or_raises(match="WrapCalibratedExplainer.explain_shap is deprecated"):
        result = w.explain_shap([[1, 2, 3]], foo="bar")
    if deprecations_error_enabled():
        assert dummy.preprocessor_metadata is None
    else:
        assert isinstance(result, dict)
        assert result["called"] is True
        assert result["x"] == [[1, 2, 3]]
        assert result["kwargs"]["foo"] == "bar"


def test_preprocessor_metadata_property_and_setter():
    initial = {"auto_encode": "true"}
    w, dummy = make_wrapper_with_dummy(DummyExplainer(initial_meta=initial))
    # property should reflect explainer attribute
    assert w.preprocessor_metadata == initial

    new_meta = {"transformer_id": "mod:Cls", "mapping_snapshot": {"a": 1}}
    w.set_preprocessor_metadata(new_meta)
    assert dummy.preprocessor_metadata == new_meta
    assert w.preprocessor_metadata == new_meta
