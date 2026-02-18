import sys
import types
import json

import numpy as np
import pytest

from calibrated_explanations.explanations import explanations as explanations_mod
from calibrated_explanations.utils.exceptions import ValidationError
from calibrated_explanations.explanations import (
    AlternativeExplanations,
    CalibratedExplanations,
    FrozenCalibratedExplainer,
)
from calibrated_explanations.plugins.manager import PluginManager


class DummyDomainMapper:
    def __init__(self):
        self.discretized_feature_names = []
        self.feature_values = None


class DummyLimeExplanation:
    def __init__(self, num_features):
        self.intercept = {1: None}
        self.local_pred = None
        self.predict_proba = [0.0, 0.0]
        self.local_exp = {1: [(None, None) for _ in range(num_features)]}
        self.domain_mapper = DummyDomainMapper()


class DummyShapExplanation:
    def __init__(self, num_features):
        self.base_values = np.zeros(1)
        self.values = np.zeros((1, num_features))
        self.data = None


class DummyCalibratedExplainer:
    def __init__(self, num_features=3, mode="classification"):
        self.feature_names = [f"feature_{i}" for i in range(num_features)]
        self.class_labels = {1: "class_one", 0: "class_zero"}
        self.num_features = num_features
        self.mode = mode
        self.y_cal = np.array([0.0, 1.0])
        self.x_cal = np.zeros((2, num_features))
        self.categorical_features = []
        self.categorical_labels = []
        self.feature_values = []
        self.sample_percentiles = [5.0, 95.0]
        self.is_multiclass = False
        self.discretizer = object()
        self.predict_func = lambda data: np.ones(len(data))
        self.rule_boundaries = []
        self.learner = "dummy"
        self.difficulty_estimator = "difficulty"
        self.plugin_manager = PluginManager(self)

        class PredOrch:
            def __init__(self, outer):
                self.outer = outer

            def predict_internal(self, data, **_kw):
                return self.outer.predict_func(data)

        self.prediction_orchestrator = PredOrch(self)

    def discretize(self, values):
        # Mock discretization for testing
        return values

    def preload_lime(self, x_cal=None):
        return None, DummyLimeExplanation(self.num_features)

    def preload_shap(self, x_cal=None):
        return None, DummyShapExplanation(self.num_features)

    def predict(self, x, **kwargs):
        return self.prediction_orchestrator.predict_internal(x, **kwargs)

    def infer_explanation_mode(self):
        return "factual"

    @property
    def is_mondrian(self):
        return False

    def runtime_telemetry(self):
        return {"explanations": self.num_features}


class DummyExplanation:
    def __init__(self, index, x_row, predict, interval, feature_weights):
        self.index = index
        self.x_test = x_row
        self.predict = predict
        self.prediction_interval = interval
        self.prediction_probabilities = None
        self.prediction = {"predict": predict}
        self.feature_weights = {"predict": np.array(feature_weights, dtype=float)}
        self.calls = []
        self.rules = None
        self.conjunctive_rules = None

    def __str__(self):
        return f"dummy-{self.index}"

    def plot(self, **kwargs):
        self.calls.append(("plot", kwargs))

    def remove_conjunctions(self):
        self.calls.append(("remove", None))

    def add_conjunctions(self, n_top_features, max_rule_size):
        self.calls.append(("add", (n_top_features, max_rule_size)))

    def reset(self):
        self.calls.append(("reset", None))

    def super_explanations(self, only_ensured=False, include_potential=True, copy=True):
        self.calls.append(
            (
                "super",
                {
                    "only_ensured": only_ensured,
                    "include_potential": include_potential,
                    "copy": copy,
                },
            )
        )
        return self

    def semi_explanations(self, only_ensured=False, include_potential=True, copy=True):
        self.calls.append(
            (
                "semi",
                {
                    "only_ensured": only_ensured,
                    "include_potential": include_potential,
                    "copy": copy,
                },
            )
        )
        return self

    def counter_explanations(self, only_ensured=False, include_potential=True, copy=True):
        self.calls.append(
            (
                "counter",
                {
                    "only_ensured": only_ensured,
                    "include_potential": include_potential,
                    "copy": copy,
                },
            )
        )
        return self

    def ensured_explanations(self, include_potential=True, copy=True):
        self.calls.append(("ensured", {"include_potential": include_potential, "copy": copy}))
        return self

    def pareto_explanations(self, include_potential=True, copy=True):
        self.calls.append(("pareto", {"include_potential": include_potential, "copy": copy}))
        return self

    def filter_rule_sizes(self, *, rule_sizes=None, size_range=None, copy=True):
        self.calls.append(
            (
                "filter_rule_sizes",
                {"rule_sizes": rule_sizes, "size_range": size_range, "copy": copy},
            )
        )
        return self

    def rank_features(self, feature_weights, num_to_show=None):
        order = list(range(len(feature_weights)))
        return order[: num_to_show if num_to_show is not None else len(order)]

    def define_conditions(self):
        return [f"rule_{i}" for i in range(len(self.feature_weights["predict"]))]

    def get_rules(self):
        return {
            "rule": [f"rule_{self.index}"],
            "base_predict": [0.1],
            "base_predict_low": [0.0],
            "base_predict_high": [0.2],
            "weight": [0.3],
            "weight_low": [0.1],
            "weight_high": [0.5],
            "value": [f"value_{self.index}"],
            "feature": [self.index],
            "feature_value": [self.x_test[0] if len(self.x_test) else None],
            "predict": [self.predict],
            "predict_low": [self.prediction_interval[0]],
            "predict_high": [self.prediction_interval[1]],
            "is_conjunctive": [False],
        }


@pytest.fixture
def calibrated_collection():
    dummy_explainer = DummyCalibratedExplainer(num_features=3)
    x = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    )
    bins = ["bin0", "bin1", "bin2"]
    collection = CalibratedExplanations(dummy_explainer, x, 0.5, bins)
    collection.explanations = [
        DummyExplanation(
            i, x[i], predict=0.1 * (i + 1), interval=(i, i + 1), feature_weights=[i, i + 1, i + 2]
        )
        for i in range(len(x))
    ]
    collection.low_high_percentiles = (10.0, 90.0)
    return collection


def test_iteration_and_indexing_behaviours(calibrated_collection):
    collection = calibrated_collection
    assert len(collection) == 3
    first = collection[0]
    assert isinstance(first, DummyExplanation)
    subset = collection[1:3]
    assert isinstance(subset, CalibratedExplanations)
    assert [exp.index for exp in subset] == [0, 1]
    bool_subset = collection[[True, False, True]]
    assert isinstance(bool_subset, CalibratedExplanations)
    assert [exp.index for exp in bool_subset] == [0, 1]
    single = collection[[2]]
    assert isinstance(single, DummyExplanation)
    assert "CalibratedExplanations" in repr(collection)

    with pytest.raises(ValidationError):
        _ = collection["invalid"]


def test_collection_init_validation_and_misc_helpers():
    x = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValidationError):
        CalibratedExplanations(
            DummyCalibratedExplainer(),
            x,
            None,
            bins=None,
            condition_source="invalid",
        )

    exported = explanations_mod.ExportedExplanationCollection(metadata={"m": 1}, explanations=())
    assert exported.__getstate__()["metadata"] == {"m": 1}

    collection = CalibratedExplanations(DummyCalibratedExplainer(), x, None, bins=None)
    collection.explanations = [DummyExplanation(0, x[0], 0.2, (0.0, 1.0), [1, 2, 3])]
    collection.explanations[0].build_rules_payload = lambda: {"ok": True}
    assert collection.build_rules_payload() == [{"ok": True}]
    assert collection.copy(deep=True) is not collection


def test_getitem_preserves_threshold_variants_and_str():
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    bins = ["b0", "b1", "b2"]
    tuple_threshold = (0.25, 0.75)
    tuple_collection = CalibratedExplanations(DummyCalibratedExplainer(), x, tuple_threshold, bins)
    tuple_collection.explanations = [
        DummyExplanation(
            i, x[i], predict=0.1 * (i + 1), interval=(i, i + 1), feature_weights=[1, 2, 3]
        )
        for i in range(len(x))
    ]
    assert str(tuple_collection).startswith("CalibratedExplanations(")
    subset = tuple_collection[1:]
    assert isinstance(subset, CalibratedExplanations)
    assert subset.y_threshold == tuple_threshold
    assert subset.bins == ["b1", "b2"]

    list_thresholds = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    list_collection = CalibratedExplanations(DummyCalibratedExplainer(), x, list_thresholds, bins)
    list_collection.explanations = [
        DummyExplanation(
            i, x[i], predict=0.2 * (i + 1), interval=(i, i + 1), feature_weights=[2, 3, 4]
        )
        for i in range(len(x))
    ]
    subset_by_index = list_collection[[0, 2]]
    assert isinstance(subset_by_index, CalibratedExplanations)
    assert subset_by_index.y_threshold == [list_thresholds[i] for i in [0, 2]]

    none_collection = CalibratedExplanations(DummyCalibratedExplainer(), x, None, bins)
    none_collection.explanations = [
        DummyExplanation(
            i, x[i], predict=0.3 * (i + 1), interval=(i, i + 1), feature_weights=[3, 4, 5]
        )
        for i in range(len(x))
    ]
    sliced_none = none_collection[1:]
    assert isinstance(sliced_none, CalibratedExplanations)
    assert sliced_none.y_threshold is None
    none_collection.low_high_percentiles = None
    assert not none_collection.is_one_sided


def test_getitem_handles_short_feature_filter_mask_without_crashing(calibrated_collection):
    calibrated_collection.feature_filter_per_instance_ignore = [[0]]
    subset = calibrated_collection[[0, 1]]
    assert subset.feature_filter_per_instance_ignore is None


def test_prediction_related_properties(calibrated_collection):
    collection = calibrated_collection
    assert collection.predict == [exp.predict for exp in collection.explanations]
    preds_first = collection.predictions
    assert np.allclose(preds_first, [0.1, 0.2, 0.3])
    collection.explanations[0].predict = 999
    preds_second = collection.predictions
    assert np.allclose(preds_second, preds_first)

    matrices = [
        np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]),
        np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]),
        np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]),
    ]
    for exp, value in zip(collection.explanations, matrices):
        exp.prediction_probabilities = value
    prob_matrix = collection.probabilities
    assert prob_matrix.shape == (3, 2)
    assert np.allclose(collection.lower, [0.0, 1.0, 2.0])
    assert np.allclose(collection.upper, [1.0, 2.0, 3.0])
    assert collection.prediction_interval == [(0, 1), (1, 2), (2, 3)]


def test_probabilities_stack_vectors():
    dummy_explainer = DummyCalibratedExplainer(num_features=2)
    x = np.array([[0.0, 0.1], [0.2, 0.3]])
    collection = CalibratedExplanations(dummy_explainer, x, None, bins=None)
    collection.explanations = [
        DummyExplanation(0, x[0], 0.4, (0.0, 1.0), [1.0, 2.0]),
        DummyExplanation(1, x[1], 0.6, (1.0, 2.0), [2.0, 3.0]),
    ]
    collection.explanations[0].prediction_probabilities = np.array([0.4, 0.6])
    collection.explanations[1].prediction_probabilities = np.array([0.3, 0.7])
    stacked = collection.probabilities
    assert stacked.shape == (2, 2)
    assert np.allclose(stacked[0], [0.4, 0.6])


class FakeFactual(DummyExplanation):
    def __init__(
        self,
        calibrated_explanations,
        index,
        instance,
        binned,
        feature_weights,
        feature_predict,
        prediction,
        y_threshold,
        instance_bin=None,
        condition_source="observed",
    ):
        super().__init__(
            index,
            instance,
            predict=prediction["predict"],
            interval=(0.0, 1.0),
            feature_weights=[0.0, 1.0, 2.0],
        )
        self.metadata = {
            "parent": calibrated_explanations,
            "binned": binned,
            "feature_weights": feature_weights,
            "feature_predict": feature_predict,
            "threshold": y_threshold,
            "instance_bin": instance_bin,
            "condition_source": condition_source,
        }


class FakeAlternative(FakeFactual):
    pass


class FakeFast(DummyExplanation):
    def __init__(
        self,
        calibrated_explanations,
        index,
        instance,
        feature_weights,
        feature_predict,
        prediction,
        y_threshold,
        instance_bin=None,
        condition_source="observed",
    ):
        super().__init__(
            index,
            instance,
            predict=prediction["predict"],
            interval=(0.0, 1.0),
            feature_weights=[0.0, 1.0, 2.0],
        )
        self.metadata = {
            "parent": calibrated_explanations,
            "feature_weights": feature_weights,
            "feature_predict": feature_predict,
            "threshold": y_threshold,
            "instance_bin": instance_bin,
            "condition_source": condition_source,
        }


def test_to_batch_and_from_batch(monkeypatch, calibrated_collection):
    from calibrated_explanations.core import SerializationError, ValidationError

    called = {}

    def fakecollection_to_batch(collection):
        called["collection"] = collection
        return {"batch": "ok"}

    fake_module = types.ModuleType("calibrated_explanations.plugins.builtins")
    fake_module.collection_to_batch = fakecollection_to_batch
    monkeypatch.setitem(sys.modules, "calibrated_explanations.plugins.builtins", fake_module)
    batch = calibrated_collection.to_batch()
    assert batch == {"batch": "ok"}
    assert called["collection"] is calibrated_collection

    class DummyBatch:
        def __init__(self, container):
            self.collection_metadata = {"container": container}
            if hasattr(container, "explanations"):
                self.instances = [{"explanation": exp} for exp in container.explanations]
                self.container_cls = type(container)
                self.explanation_cls = type(container.explanations[0])
            else:
                self.instances = []
                self.container_cls = None
                self.explanation_cls = object

    restored = CalibratedExplanations.from_batch(DummyBatch(calibrated_collection))
    # ADR-015: from_batch must reconstruct a new instance even if template is provided
    assert restored is not calibrated_collection
    assert (
        restored.calibrated_explainer.num_features
        == calibrated_collection.calibrated_explainer.num_features
    )
    assert len(restored.explanations) == len(calibrated_collection.explanations)

    with pytest.raises(SerializationError):
        CalibratedExplanations.from_batch(DummyBatch(None))

    with pytest.raises(ValidationError):
        CalibratedExplanations.from_batch(DummyBatch(object()))


def test_plot_routing(monkeypatch, calibrated_collection):
    saved = {}

    def fake_prepare(filename):
        saved["filename"] = filename
        return ("/tmp/", "file", "title", ".png")

    monkeypatch.setattr(explanations_mod, "prepare_for_saving", fake_prepare)
    calibrated_collection.plot(index=1, filename="plot.png", show=False)
    expected_kwargs = {
        "filter_top": 10,
        "show": False,
        "filename": "/tmp/title1.png",
        "uncertainty": False,
        "style": "regular",
        "rnk_metric": None,
        "rnk_weight": 0.5,
        "style_override": None,
    }
    assert saved["filename"] == "plot.png"
    assert ("plot", expected_kwargs) in calibrated_collection.explanations[1].calls

    calibrated_collection.plot(filename="plot_all.png", show=True)
    for exp in calibrated_collection.explanations:
        assert any(call[0] == "plot" for call in exp.calls)


def test_conjunction_management(calibrated_collection):
    calibrated_collection.add_conjunctions(n_top_features=2, max_rule_size=3)
    calibrated_collection.reset()
    calibrated_collection.remove_conjunctions()
    for exp in calibrated_collection.explanations:
        actions = [call[0] for call in exp.calls]
        assert {"remove", "add", "reset"}.issubset(actions)


def test_alternative_specific_filters(calibrated_collection):
    alt = AlternativeExplanations.from_collection(calibrated_collection)
    alt.super_explanations(only_ensured=True, include_potential=False)
    alt.semi_explanations(only_ensured=True, include_potential=False)
    alt.counter_explanations(only_ensured=True, include_potential=False)
    alt.ensured_explanations()
    alt.pareto_explanations(include_potential=False)
    for exp in alt.explanations:
        actions = [
            call[0]
            for call in exp.calls
            if call[0] in {"super", "semi", "counter", "ensured", "pareto"}
        ]
        assert {"super", "semi", "counter", "ensured", "pareto"}.issubset(set(actions))


def test_alternative_specific_filters_inplace(calibrated_collection):
    alt = AlternativeExplanations.from_collection(calibrated_collection)

    out_super = alt.super_explanations(copy=False)
    out_semi = alt.semi_explanations(copy=False)
    out_counter = alt.counter_explanations(copy=False)
    out_ensured = alt.ensured_explanations(copy=False)
    out_pareto = alt.pareto_explanations(copy=False)

    assert out_super is alt
    assert out_semi is alt
    assert out_counter is alt
    assert out_ensured is alt
    assert out_pareto is alt
    for exp in alt.explanations:
        actions = [call[0] for call in exp.calls]
        assert "super" in actions
        assert "semi" in actions
        assert "counter" in actions
        assert "ensured" in actions
        assert "pareto" in actions


def test_collection_to_json_and_back(calibrated_collection):
    for i, exp in enumerate(calibrated_collection.explanations):
        exp.provenance = {"steps": np.array([i, i + 1])}
        exp.metadata = {"scores": np.array([[i]])}
    payload = calibrated_collection.to_json()
    assert payload["collection"]["size"] == len(calibrated_collection)
    assert payload["schema_version"] == "1.0.0"
    exported = CalibratedExplanations.from_json(payload)
    assert exported.metadata["size"] == len(calibrated_collection)
    assert len(exported.explanations) == len(calibrated_collection)
    assert all(exp.task == "classification" for exp in exported.explanations)


def test_collection_to_json_stream(calibrated_collection):
    for i, exp in enumerate(calibrated_collection.explanations):
        exp.provenance = {"steps": np.array([i, i + 1])}
        exp.metadata = {"scores": np.array([[i]])}

    # Test JSONL format
    chunks = list(calibrated_collection.to_json_stream(format="jsonl"))
    assert len(chunks) == len(calibrated_collection) + 2  # metadata + explanations + telemetry

    # First chunk is metadata
    meta = json.loads(chunks[0])
    assert "collection" in meta
    assert meta["schema_version"] == "1.0.0"

    # Last chunk is telemetry
    telemetry = json.loads(chunks[-1])
    assert "export_telemetry" in telemetry
    assert telemetry["export_telemetry"]["export_rows"] == len(calibrated_collection)
    assert "elapsed_seconds" in telemetry["export_telemetry"]
    assert "peak_memory_mb" in telemetry["export_telemetry"]

    # Middle chunks are explanations
    for chunk in chunks[1:-1]:
        exp_data = json.loads(chunk)
        assert "schema_version" in exp_data
        assert exp_data["schema_version"] == "1.0.0"

    # Test chunked format
    chunks_chunked = list(calibrated_collection.to_json_stream(format="chunked", chunk_size=2))
    assert len(chunks_chunked) >= 2  # at least metadata and one chunk

    # First is metadata
    meta_chunked = json.loads(chunks_chunked[0])
    assert "collection" in meta_chunked

    # Last is telemetry
    telemetry_chunked = json.loads(chunks_chunked[-1])
    assert "export_telemetry" in telemetry_chunked

    # Middle chunks are JSON arrays
    for chunk in chunks_chunked[1:-1]:
        assert chunk.startswith("[") and chunk.endswith("]")
        # Parse as JSON array
        arr = json.loads(chunk)
        assert isinstance(arr, list)
        assert len(arr) <= 2  # chunk_size=2


def test_to_json_stream_handles_underlying_telemetry_attachment_failure(calibrated_collection):
    class BrokenUnderlying:
        def __setattr__(self, name, value):
            if name == "_last_telemetry":
                raise RuntimeError("read-only telemetry")
            return super().__setattr__(name, value)

    class FrozenLike:
        mode = "classification"
        feature_names = ["feature_0", "feature_1", "feature_2"]
        class_labels = {0: "zero", 1: "one"}
        sample_percentiles = [5.0, 95.0]
        _explainer = BrokenUnderlying()

    calibrated_collection.calibrated_explainer = FrozenLike()
    chunks = list(calibrated_collection.to_json_stream(format="jsonl"))
    assert chunks
    telemetry = json.loads(chunks[-1])
    assert telemetry["export_telemetry"]["schema_version"] == "1.0.0"


def test_collection_to_json_stream_rejects_unknown_format(calibrated_collection):
    with pytest.raises(ValidationError):
        list(calibrated_collection.to_json_stream(format="xml"))


def test_collection_to_json_without_version(calibrated_collection):
    payload = calibrated_collection.to_json(include_version=False)
    assert "schema_version" not in payload
    assert all("schema_version" not in item for item in payload["explanations"])


def test_legacy_payload_prefers_available_rules(calibrated_collection):
    exp = calibrated_collection.explanations[0]
    exp.has_conjunctive_rules = True  # pylint: disable=protected-access
    exp.conjunctive_rules = {"ensured": ["rule-a"]}
    payload = calibrated_collection.legacy_payload(exp)
    assert payload["rules"] == exp.conjunctive_rules

    exp.has_conjunctive_rules = False  # pylint: disable=protected-access
    exp.conjunctive_rules = None
    exp.rules = {"ensured": ["rule-b"]}
    payload_rules = calibrated_collection.legacy_payload(exp)
    assert payload_rules["rules"] == exp.rules

    exp.rules = None
    generated = calibrated_collection.legacy_payload(exp)
    assert "rule" in generated["rules"]


def test_legacy_payload_handles_get_rules_exception(calibrated_collection):
    class BrokenRules:
        index = 0
        has_conjunctive_rules = False
        conjunctive_rules = None
        rules = None
        feature_weights = {}
        feature_predict = {}
        prediction = {}

        def get_mode(self):
            return "classification"

        def get_rules(self):
            raise RuntimeError("boom")

    payload = calibrated_collection.legacy_payload(BrokenRules())
    assert payload["rules"] == {}


def test_collection_metadata_includes_runtime(calibrated_collection):
    metadata = calibrated_collection.collection_metadata()
    assert metadata["feature_names"] == calibrated_collection.feature_names
    assert metadata["class_labels"] == {"1": "class_one", "0": "class_zero"}
    assert metadata["sample_percentiles"] == [5.0, 95.0]
    assert metadata["runtime_telemetry"] == {"explanations": 3}


def test_collection_metadata_runtime_telemetry_failure():
    class BrokenUnderlying:
        def runtime_telemetry(self):
            raise RuntimeError("telemetry unavailable")

    class BrokenFrozen:
        mode = "classification"
        class_labels = {0: "zero"}
        sample_percentiles = [5.0, 95.0]
        _explainer = BrokenUnderlying()

        @property
        def feature_names(self):
            raise RuntimeError("feature names unavailable")

    collection = CalibratedExplanations(
        DummyCalibratedExplainer(),
        np.array([[0.1, 0.2, 0.3]]),
        None,
        bins=None,
    )
    collection.explanations = [
        DummyExplanation(0, np.array([0.1, 0.2, 0.3]), 0.1, (0.0, 1.0), [1, 2, 3])
    ]
    collection.calibrated_explainer = BrokenFrozen()
    metadata = collection.collection_metadata()
    assert metadata["mode"] == "classification"
    assert "feature_names" not in metadata
    assert "runtime_telemetry" not in metadata


def test_frozen_explainer_read_only():
    dummy = DummyCalibratedExplainer()
    frozen = FrozenCalibratedExplainer(dummy)
    assert np.array_equal(frozen.x_cal, dummy.x_cal)
    assert np.array_equal(frozen.y_cal, dummy.y_cal)
    assert frozen.num_features == dummy.num_features
    assert frozen.categorical_features == dummy.categorical_features
    assert frozen.categorical_labels == dummy.categorical_labels
    assert frozen.feature_values == dummy.feature_values
    assert frozen.feature_names == dummy.feature_names
    assert frozen.class_labels == dummy.class_labels
    assert not hasattr(frozen, "assign_threshold")
    assert frozen.sample_percentiles == dummy.sample_percentiles
    assert frozen.mode == dummy.mode
    assert frozen.is_multiclass == dummy.is_multiclass
    assert type(frozen.discretizer) is type(dummy.discretizer)
    assert np.array_equal(frozen.discretize(dummy.x_cal), dummy.x_cal)
    assert frozen.rule_boundaries == dummy.rule_boundaries
    assert frozen.learner == dummy.learner
    assert frozen.difficulty_estimator == dummy.difficulty_estimator
    assert frozen.predict(dummy.x_cal).shape[0] == dummy.x_cal.shape[0]
    assert callable(frozen.preload_lime)
    assert callable(frozen.preload_shap)
    with pytest.raises(AttributeError):
        frozen.some_attribute = 5


def test_filter_rule_sizes_inplace_mutates_collection(calibrated_collection):
    out = calibrated_collection.filter_rule_sizes(rule_sizes=1, copy=False)
    assert out is calibrated_collection
    for exp in calibrated_collection.explanations:
        assert any(
            call[0] == "filter_rule_sizes" and call[1]["copy"] is False for call in exp.calls
        )


def test_from_batch_full_probabilities_and_instance_validation(calibrated_collection):
    from calibrated_explanations.core import SerializationError, ValidationError

    class GoodBatch:
        def __init__(self, collection):
            self.collection_metadata = {"container": collection}
            self.container_cls = type(collection)
            self.explanation_cls = type(collection.explanations[0])
            self.instances = [
                {
                    "prediction": {"__full_probabilities__": [[0.1, 0.9], [0.2, 0.8]]},
                    "explanation": collection.explanations[0],
                }
            ]

    restored = CalibratedExplanations.from_batch(GoodBatch(calibrated_collection))
    assert restored.batch_metadata["__full_probabilities__"] is not None
    assert "full_probabilities_shape" in restored.telemetry

    class MissingExplanationBatch(GoodBatch):
        def __init__(self, collection):
            super().__init__(collection)
            self.instances = [{"prediction": {}}]

    with pytest.raises(SerializationError):
        CalibratedExplanations.from_batch(MissingExplanationBatch(calibrated_collection))

    class WrongTypeBatch(GoodBatch):
        def __init__(self, collection):
            super().__init__(collection)
            self.instances = [{"explanation": object()}]

    with pytest.raises(ValidationError):
        CalibratedExplanations.from_batch(WrongTypeBatch(calibrated_collection))


def test_from_batch_requires_explainer_context(calibrated_collection):
    from calibrated_explanations.core import SerializationError, ValidationError

    class IncompleteTemplate:
        calibrated_explainer = None
        x_test = None
        y_threshold = None
        bins = None
        features_to_ignore = []
        condition_source = "prediction"

    class MissingContextBatch:
        container_cls = type(calibrated_collection)
        explanation_cls = type(calibrated_collection.explanations[0])
        instances = []
        collection_metadata = {"container": IncompleteTemplate()}

    with pytest.raises((ValidationError, SerializationError)):
        CalibratedExplanations.from_batch(MissingContextBatch())

    template = CalibratedExplanations(
        DummyCalibratedExplainer(),
        np.array([[0.1, 0.2, 0.3]]),
        None,
        bins=None,
    )
    template.calibrated_explainer = None
    template.x_test = None

    class MissingExplainerContextBatch:
        container_cls = CalibratedExplanations
        explanation_cls = DummyExplanation
        instances = []
        collection_metadata = {"container": template}

    with pytest.raises(SerializationError):
        CalibratedExplanations.from_batch(MissingExplainerContextBatch())


def test_collection_property_fallback_paths_for_cached_fields():
    x = np.array([[0.1, 0.2, 0.3]])
    collection = CalibratedExplanations(DummyCalibratedExplainer(), x, None, bins=None)

    class BrokenLabels:
        @property
        def class_labels(self):
            raise RuntimeError("broken labels")

        @property
        def feature_names(self):
            raise RuntimeError("broken names")

    collection.calibrated_explainer = BrokenLabels()
    assert collection.feature_names is None
    assert collection.class_labels is None

    class BrokenPredict:
        prediction_probabilities = None
        prediction_interval = (None, None)

        @property
        def predict(self):
            raise RuntimeError("broken predict")

    collection.explanations = [BrokenPredict()]
    assert collection.predictions is None


def test_collection_to_narrative_and_plot_style_narrative(monkeypatch, calibrated_collection):
    calls = []

    class FakeNarrativePlugin:
        def __init__(self, template_path=None):
            self.template_path = template_path

        def plot(self, explanations, **kwargs):
            calls.append((explanations, kwargs))
            return [{"ok": True, "kwargs": kwargs}]

    monkeypatch.setattr(
        "calibrated_explanations.viz.narrative_plugin.NarrativePlotPlugin",
        FakeNarrativePlugin,
    )

    out = calibrated_collection.to_narrative(
        template_path="custom.yaml",
        expertise_level=("beginner",),
        output_format="dict",
    )
    assert out[0]["ok"] is True
    assert calls[-1][1]["template_path"] == "custom.yaml"
    assert calls[-1][1]["output"] == "dict"

    class FakeItem:
        def __init__(self):
            self.called = None

        def to_narrative(self, **kwargs):
            self.called = kwargs
            return {"single": True, "kwargs": kwargs}

    single = FakeItem()
    calibrated_collection.explanations = [single]
    result = calibrated_collection.plot(
        style="narrative",
        index=0,
        template_path="single.yaml",
        expertise_level="advanced",
        output="text",
    )
    assert result["single"] is True
    assert single.called["template_path"] == "single.yaml"
    assert single.called["output_format"] == "text"


def test_as_lime_and_shap_transformations(calibrated_collection):
    lime_explanations = calibrated_collection.as_lime(num_features_to_show=2)
    assert len(lime_explanations) == len(calibrated_collection)
    for lime in lime_explanations:
        assert lime.local_pred is not None
        assert len(lime.local_exp[1]) == 2

    shap_exp = calibrated_collection.as_shap()
    assert shap_exp.values.shape[0] == len(calibrated_collection)
    assert shap_exp.data is calibrated_collection.x_test


def test_as_lime_regression_branch():
    dummy_explainer = DummyCalibratedExplainer(mode="regression")
    x = np.array([[0.1, 0.2, 0.3]])
    collection = CalibratedExplanations(dummy_explainer, x, 0.5, bins=["bin0"])
    collection.explanations = [
        DummyExplanation(
            0, x[0], predict=0.42, interval=(0.0, 1.0), feature_weights=[1.0, 2.0, 3.0]
        )
    ]
    lime = collection.as_lime()
    assert lime[0].predicted_value == collection.explanations[0].prediction["predict"]
    assert lime[0].min_value == np.min(dummy_explainer.y_cal)
    assert lime[0].max_value == np.max(dummy_explainer.y_cal)


def test_class_labels_and_feature_names_cache(calibrated_collection):
    labels = calibrated_collection.class_labels
    assert labels == ["class_zero", "class_one"]
    assert calibrated_collection.feature_names == ["feature_0", "feature_1", "feature_2"]
