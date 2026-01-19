import sys

import types

from calibrated_explanations.plugins.builtins import (
    _supports_calibrated_explainer,
    collection_to_batch,
)


def test_supports_calibrated_explainer_true():
    # Create a fake module and class at the expected import path so
    # safe_isinstance can discover it.
    mod_name = "calibrated_explanations.core.calibrated_explainer"
    fake_mod = types.ModuleType(mod_name)

    class CalibratedExplainer:
        pass

    fake_mod.CalibratedExplainer = CalibratedExplainer
    sys.modules[mod_name] = fake_mod

    inst = CalibratedExplainer()
    assert _supports_calibrated_explainer(inst) is True


def test_collection_to_batch_minimal():
    class DummyCollection:
        def __init__(self):
            self.explanations = [object()]
            self.mode = "classification"
            self.calibrated_explainer = None
            self.x_test = None
            self.y_threshold = None
            self.bins = None
            self.features_to_ignore = None
            self.low_high_percentiles = None
            self.feature_filter_per_instance_ignore = None
            self.filter_telemetry = None

    c = DummyCollection()
    batch = collection_to_batch(c)
    assert hasattr(batch, "instances")
    assert batch.collection_metadata["container"] is c
