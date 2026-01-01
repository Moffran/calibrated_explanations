import pickle
from calibrated_explanations.plugins.explanations import ExplanationContext


class MockBridge:
    pass


def test_explanation_context_is_picklable():
    context = ExplanationContext(
        task="classification",
        mode="factual",
        feature_names=("f1", "f2"),
        categorical_features=(0,),
        categorical_labels={0: {0: "a", 1: "b"}},
        discretizer=None,
        helper_handles={"foo": "bar"},
        predict_bridge=MockBridge(),
        interval_settings={"a": 1},
        plot_settings={"b": 2},
    )

    pickled = pickle.dumps(context)
    unpickled = pickle.loads(pickled)

    assert unpickled is not None
    assert unpickled.task == "classification"
    assert unpickled.categorical_labels == {0: {0: "a", 1: "b"}}
