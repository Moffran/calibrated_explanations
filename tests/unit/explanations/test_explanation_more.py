import types
import sys

import numpy as np
import pandas as pd
import pytest

from calibrated_explanations.explanations import explanation as explanation_module
from calibrated_explanations.explanations.explanation import CalibratedExplanation


def make_container_and_explainer():
    class ExplainerStub:
        def __init__(self):
            self.mode = "regression"
            self.class_labels = None
            self.feature_names = ["f0"]
            self.categorical_features = []
            self.categorical_labels = None
            self.sample_percentiles = [25, 75]
            self.num_features = 1
            self.x_cal = np.array([[0.1]])
            self.y_cal = np.array([0.0, 1.0])

        def is_multiclass(self):
            return False

        def predict(self, data, **_kwargs):
            rows = data.shape[0]
            predict = np.ones((rows, 1)) * 0.5
            low = np.ones((rows, 1)) * 0.4
            high = np.ones((rows, 1)) * 0.6
            return predict, low, high, np.zeros(rows, dtype=int)

    class ContainerStub:
        def __init__(self):
            self.low_high_percentiles = (5, 95)
            self.explainer = ExplainerStub()
            self.features_to_ignore = []
            self.explanations = []

        def _get_explainer(self):
            return self.explainer

        def get_explainer(self):
            return self.explainer

    return ContainerStub(), ContainerStub().get_explainer()


def test_to_narrative_handles_output_formats(monkeypatch):
    # Provide a fake narrative plugin module
    mod = types.ModuleType("calibrated_explanations.viz.narrative_plugin")

    class FakePlugin:
        def __init__(self, template_path=None):
            self.template_path = template_path

        def plot(self, wrapper, template_path=None, expertise_level=None, output=None, **_kw):
            # simple behavior depending on output
            if output == "dict":
                return [{"narrative": "ok"}]
            if output == "text":
                return "narrative text"
            if output == "dataframe":
                return pd.DataFrame([{"narrative": "ok"}])
            return []

    mod.NarrativePlotPlugin = FakePlugin
    monkeypatch.setitem(sys.modules, "calibrated_explanations.viz.narrative_plugin", mod)

    # Minimal Dummy explanation subclass (do not call super())
    class Dummy(CalibratedExplanation):
        def __init__(self):
            self.calibrated_explanations = types.SimpleNamespace(calibrated_explainer=None)
            self.y_threshold = None

        def get_explainer_helper(self):
            return types.SimpleNamespace()

        def _check_preconditions(self):
            return None

        def get_rules(self):
            return {"rule": []}

        def _is_lesser(self, a, b):
            return a < b

        def __repr__(self):
            return "dummy"

        def plot(self, *a, **k):
            return None

        def add_conjunctions(self, *a, **k):
            return self

        def build_rules_payload(self):
            return {"core": {}, "metadata": {}}

        @property
        def has_rules(self):
            return False

        @property
        def has_conjunctive_rules(self):
            return False

    d = Dummy()
    # dict
    out = d.to_narrative(output_format="dict")
    assert isinstance(out, (dict, list))
    # text
    out_text = d.to_narrative(output_format="text")
    assert isinstance(out_text, str) and "narrative" in out_text
    # dataframe
    out_df = d.to_narrative(output_format="dataframe")
    assert isinstance(out_df, pd.DataFrame)


def test_plot_runtimeerror_agg_raises_configuration_error(monkeypatch):
    container, explainer = make_container_and_explainer()

    # Ensure explanation module's discretizer classes are present as simple dummies
    class DummyDiscretizer:
        def __init__(self):
            # simple names for one feature
            self.names = [["f0 <= 0.2", "f0 > 0.2"]]

        def discretize(self, x):
            # return a single-bin index per column
            return np.asarray(x).reshape(-1)[0:1] * 0 + 0

    from calibrated_explanations import utils

    monkeypatch.setattr(utils, "BinaryRegressorDiscretizer", DummyDiscretizer)
    monkeypatch.setattr(utils, "BinaryEntropyDiscretizer", DummyDiscretizer)
    # ensure explainer has a discretizer instance to satisfy isinstance checks
    container.explainer.discretizer = DummyDiscretizer()
    # build small arrays for constructor of FactualExplanation
    x = np.array([[0.1]])
    rule_values = np.empty((1, 1), dtype=object)
    rule_values[0, 0] = np.array([[0.1]])
    binned = {
        "predict": np.array([[0.2]]),
        "low": np.array([[0.1]]),
        "high": np.array([[0.3]]),
        "rule_values": rule_values,
    }
    feature_weights = {
        "predict": np.array([[0.1]]),
        "low": np.array([[0.05]]),
        "high": np.array([[0.15]]),
    }
    feature_predict = {
        "predict": np.array([[0.5]]),
        "low": np.array([[0.4]]),
        "high": np.array([[0.6]]),
    }
    prediction = {
        "predict": np.array([0.2]),
        "low": np.array([0.1]),
        "high": np.array([0.4]),
        "classes": np.array([1]),
    }

    # instantiate FactualExplanation
    factual = explanation_module.FactualExplanation(
        container,
        index=0,
        x=x,
        binned=binned,
        feature_weights=feature_weights,
        feature_predict=feature_predict,
        prediction=prediction,
    )

    # monkeypatch plotting functions to raise RuntimeError with 'Agg' in message
    def raise_agg(*args, **kwargs):
        raise RuntimeError("Backend 'Agg' not available")

    from calibrated_explanations import plotting

    monkeypatch.setattr(plotting, "plot_regression", raise_agg)
    monkeypatch.setattr(plotting, "plot_probabilistic", raise_agg)
    # Also monkeypatch the imported names in the explanation module
    monkeypatch.setattr(explanation_module, "plot_regression", raise_agg)
    monkeypatch.setattr(explanation_module, "plot_probabilistic", raise_agg)

    from calibrated_explanations.utils.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        factual.plot(show=True)


def test_fast_explanation_repr_and_build_payload():
    container, explainer = make_container_and_explainer()
    x = np.array([[1.0]])
    rule_values = np.empty((1, 1), dtype=object)
    rule_values[0, 0] = np.array([[1.0]])
    feature_weights = {
        "predict": np.array([[0.1]]),
        "low": np.array([[0.05]]),
        "high": np.array([[0.15]]),
    }
    feature_predict = {
        "predict": np.array([[0.5]]),
        "low": np.array([[0.4]]),
        "high": np.array([[0.6]]),
    }
    prediction = {
        "predict": np.array([0.25]),
        "low": np.array([0.2]),
        "high": np.array([0.3]),
        "classes": np.array([1]),
    }

    fast = explanation_module.FastExplanation(
        container,
        index=0,
        x=x,
        feature_weights=feature_weights,
        feature_predict=feature_predict,
        prediction=prediction,
    )

    # inject a minimal rules payload so repr has something to format
    mock_rules = {
        "base_predict": [0.25],
        "base_predict_low": [0.2],
        "base_predict_high": [0.3],
        "rule": ["f0 <= 0.2"],
        "value": ["1.00"],
        "weight": [0.1],
        "weight_low": [0.05],
        "weight_high": [0.15],
        "feature": [0],
        "is_conjunctive": [False],
        "feature_value": [1.0],
        "predict": [0.25],
        "predict_low": [0.2],
        "predict_high": [0.3],
    }
    fast.rules = mock_rules
    fast.has_rules = True

    r = repr(fast)
    assert "Prediction" in r
    payload = fast.build_rules_payload()
    assert "core" in payload and "metadata" in payload
