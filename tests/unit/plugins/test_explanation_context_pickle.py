import pickle
from types import MappingProxyType

from calibrated_explanations.plugins.explanations import ExplanationContext


def test_explanation_context_pickle_converts_mappingproxy():
    ctx = ExplanationContext(
        task="task",
        mode="classification",
        feature_names=("f1",),
        categorical_features=(),
        categorical_labels=MappingProxyType({0: MappingProxyType({0: "a"})}),
        discretizer=object(),
        helper_handles={},
        predict_bridge=object(),
        interval_settings=MappingProxyType({"k": 1}),
        plot_settings=MappingProxyType({"p": 2}),
    )

    dumped = pickle.dumps(ctx)
    loaded = pickle.loads(dumped)
    # __getstate__ returns dict(self.__dict__), so loaded state fields should be present
    assert loaded.task == "task"
    assert isinstance(loaded.interval_settings, dict)
