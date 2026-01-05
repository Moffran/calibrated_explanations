import numpy as np

from calibrated_explanations.core.explain._feature_filter import (
    compute_filtered_features_to_ignore,
    FeatureFilterConfig,
)


def make_dummy_collection(feature_weight_arrays):
    class DummyExp:
        def __init__(self, arr):
            self.feature_weights = {"predict": np.asarray(arr)}

    class DummyCollection:
        def __init__(self, exps):
            self.explanations = exps

    exps = [DummyExp(a) for a in feature_weight_arrays]
    return DummyCollection(exps)


def test_global_top_k_respected_when_aggregating_max_abs():
    # 3 instances, 10 features total
    num_features = 10
    per_instance_top_k = 5

    # Construct weights so each instance has a distinct top-k set
    # Instance 0: features 0-4 strong
    w0 = [10 if i < 5 else 0 for i in range(num_features)]
    # Instance 1: features 5-9 strong
    w1 = [10 if 5 <= i < 10 else 0 for i in range(num_features)]
    # Instance 2: features 2-6 moderate (overlapping)
    w2 = [5 if 2 <= i <= 6 else 0 for i in range(num_features)]

    coll = make_dummy_collection([w0, w1, w2])

    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=per_instance_top_k)
    result = compute_filtered_features_to_ignore(
        coll, num_features=num_features, base_ignore=np.array([], dtype=int), config=cfg
    )

    # Number of globally kept features should be the union of per-instance keeps.
    # In this test, instances have disjoint top-k sets (0-4, 5-9, 2-6).
    # The union of these sets is 0-9 (all 10 features).
    kept = num_features - len(result.global_ignore)
    assert kept == 10, "Global kept features should be the union of per-instance keeps"
    assert result.global_ignore.size == 0, "No features should be globally ignored in this disjoint case"

    # Also assert that the number of features kept per instance is below
    # per_instance_top_k
    for i, ignore_arr in enumerate(result.per_instance_ignore):
        kept_count = num_features - len(ignore_arr)
        assert kept_count <= per_instance_top_k, (
            f"Instance {i} kept {kept_count} features, "
            f"exceeds per_instance_top_k={per_instance_top_k}"
        )
