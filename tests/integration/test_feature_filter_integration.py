from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder


def test_explain_factual_respects_per_instance_top_k():
    # Small deterministic dataset to keep the integration test fast
    x, y = make_classification(
        n_samples=400, n_features=50, n_informative=5, n_redundant=0, random_state=0
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10, random_state=0)
    x_train_proper, x_cal, y_train_proper, y_cal = train_test_split(
        x_train, y_train, test_size=50, random_state=0
    )

    learner = RandomForestClassifier(n_estimators=10, random_state=0)
    learner.fit(x_train_proper, y_train_proper)

    builder = ExplainerBuilder(learner)
    config = (
        builder.task("classification")
        .perf_parallel(False)
        .perf_feature_filter(True, per_instance_top_k=5)
        .build_config()
    )

    wrapper = WrapCalibratedExplainer._from_config(config)
    wrapper.calibrate(x_cal, y_cal)

    # Explain a handful of instances and assert each final explanation respects the top-k
    explanations = wrapper.explain_factual(x_test[:5])
    per_k = getattr(getattr(wrapper, "_feature_filter_config", None), "per_instance_top_k", None)
    assert per_k is not None
    for i, e in enumerate(explanations):
        rules = e.get_rules()["rule"]
        kept = len(rules)
        assert kept <= per_k, f"instance {i} kept {kept} rules > top_k {per_k}"
