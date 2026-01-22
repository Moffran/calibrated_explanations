import numpy as np
from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.core.explain._feature_filter import (
    compute_filtered_features_to_ignore,
    FeatureFilterConfig,
)


class DummyFastExplanation:
    def __init__(self, weights: np.ndarray):
        self.feature_weights = {"predict": np.asarray(weights, dtype=float)}


class DummyExplainerForFrozen:
    def __init__(self, n_features: int):
        self.x_cal = np.zeros((1, n_features))
        self.y_cal = np.zeros(1)
        self.num_features = n_features


def should_compute_global_ignore_for_fast_collection():
    # Arrange
    num_features = 4
    fast_collection = CalibratedExplanations(
        DummyExplainerForFrozen(num_features), np.zeros((2, num_features)), None, None
    )
    # instance 0 prefers feature 0, instance 1 prefers feature 1 -> feature 3 never kept
    fast_collection.explanations = [
        DummyFastExplanation(np.array([10.0, 0.0, 0.0, 0.0])),
        DummyFastExplanation(np.array([0.0, 5.0, 0.3, 1.0])),
    ]

    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=1)

    # Act
    result = compute_filtered_features_to_ignore(
        fast_collection, num_features=num_features, base_ignore=np.array([], dtype=int), config=cfg
    )

    # Assert: result has per-instance masks and global ignore includes feature 3
    assert hasattr(result, "per_instance_ignore")
    # global_ignore is a numpy array; check that 3 is included
    assert 3 in {int(x) for x in getattr(result, "global_ignore", [])}
