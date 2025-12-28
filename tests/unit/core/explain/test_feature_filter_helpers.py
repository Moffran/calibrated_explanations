import numpy as np
import pytest

from calibrated_explanations.core.explain._feature_filter import (
    FeatureFilterConfig,
    compute_filtered_features_to_ignore,
)


class FakeExplanation:
    def __init__(self, weights):
        self.feature_weights = {"predict": np.array(weights)}


class FakeCollection:
    def __init__(self, weight_sequences):
        self.explanations = [FakeExplanation(weights) for weights in weight_sequences]


class CustomExplanation:
    def __init__(self, feature_weights):
        self.feature_weights = feature_weights


class CustomCollection:
    def __init__(self, feature_weights_list):
        self.explanations = [CustomExplanation(weights) for weights in feature_weights_list]


def test_feature_filter_config_parsing_thresholds(monkeypatch):
    """FeatureFilterConfig honors simple enabling/disabling tokens and top_k overrides."""
    base = FeatureFilterConfig(enabled=False, per_instance_top_k=5)
    monkeypatch.setenv("CE_FEATURE_FILTER", "enable,top_k=3")
    cfg = FeatureFilterConfig.from_base_and_env(base)
    assert cfg.enabled is True
    assert cfg.per_instance_top_k == 3

    monkeypatch.setenv("CE_FEATURE_FILTER", "false")
    cfg2 = FeatureFilterConfig.from_base_and_env(base)
    assert cfg2.enabled is False
    assert cfg2.per_instance_top_k == 5

    monkeypatch.setenv("CE_FEATURE_FILTER", "top_k=not-a-number")
    cfg3 = FeatureFilterConfig.from_base_and_env(base)
    assert cfg3.per_instance_top_k == 5

    monkeypatch.delenv("CE_FEATURE_FILTER", raising=False)
    cfg4 = FeatureFilterConfig.from_base_and_env(base)
    assert cfg4.enabled is False


def test_feature_filter_config_simple_tokens(monkeypatch):
    base = FeatureFilterConfig(enabled=False, per_instance_top_k=4)
    monkeypatch.setenv("CE_FEATURE_FILTER", "true")
    cfg = FeatureFilterConfig.from_base_and_env(base)
    assert cfg.enabled is True

    monkeypatch.setenv("CE_FEATURE_FILTER", "off,top_k=2")
    cfg2 = FeatureFilterConfig.from_base_and_env(base)
    assert cfg2.enabled is False
    assert cfg2.per_instance_top_k == 2


def test_feature_filter_config_enforces_top_k_bounds(monkeypatch):
    base = FeatureFilterConfig(per_instance_top_k=5)
    monkeypatch.setenv("CE_FEATURE_FILTER", "top_k=-2")
    cfg = FeatureFilterConfig.from_base_and_env(base)
    assert cfg.per_instance_top_k == 1


def test_compute_filtered_features_skips_when_disabled():
    """Disabled feature filtering should only echo the baseline ignore set."""
    collection = FakeCollection([])
    cfg = FeatureFilterConfig(enabled=False)
    baseline = np.array([2, 4], dtype=int)

    result = compute_filtered_features_to_ignore(
        collection,
        num_features=5,
        base_ignore=baseline,
        config=cfg,
    )

    assert np.array_equal(result.global_ignore, baseline)
    assert result.per_instance_ignore == []


@pytest.mark.parametrize(
    "weights_list,expected_per_instance",
    [
        ([[0.1, 0.2, 0.5], [0.0, 0.1, 0.9]], [[0, 1], [0, 1]]),
        ([[1.0, 0.0, 0.0]], [[1, 2]]),
    ],
)
def test_compute_filtered_features_picks_top_k(weights_list, expected_per_instance):
    """Verify that enabled filtering retains up to `per_instance_top_k` features."""
    collection = FakeCollection(weights_list)
    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=1)
    baseline = np.array([1], dtype=int)

    result = compute_filtered_features_to_ignore(
        collection,
        num_features=3,
        base_ignore=baseline,
        config=cfg,
    )

    expected_global = (
        np.array([0, 1], dtype=int) if len(weights_list) > 1 else np.array([1, 2], dtype=int)
    )
    assert np.array_equal(result.global_ignore, expected_global)
    assert [arr.tolist() for arr in result.per_instance_ignore] == expected_per_instance


def test_compute_filtered_features_handles_invalid_weights():
    collection = CustomCollection(
        [
            None,
            {},
            {"predict": []},
        ]
    )
    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=3)
    baseline = np.array([0, 1, 2], dtype=int)

    result = compute_filtered_features_to_ignore(
        collection,
        num_features=3,
        base_ignore=baseline,
        config=cfg,
    )

    assert np.array_equal(result.global_ignore, baseline)
    assert all(np.array_equal(arr, baseline) for arr in result.per_instance_ignore)


def test_compute_filtered_features_covers_padding_truncation():
    collection = CustomCollection(
        [
            {"predict": [1]},
            {"predict": [10, -1, 0, 2]},
        ]
    )
    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=1)
    baseline = np.array([0], dtype=int)

    result = compute_filtered_features_to_ignore(
        collection,
        num_features=3,
        base_ignore=baseline,
        config=cfg,
    )

    assert np.array_equal(result.global_ignore, baseline)
    assert [arr.tolist() for arr in result.per_instance_ignore] == [[0, 1], [0, 2]]


def test_compute_filtered_features_preserves_disjoint_global_keeps():
    """Global mask should keep all features selected across instances."""
    collection = FakeCollection(
        [
            np.array([10.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 5.0, 0.3, 1.0]),
        ]
    )
    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=1)
    baseline = np.array([2], dtype=int)

    result = compute_filtered_features_to_ignore(
        collection,
        num_features=4,
        base_ignore=baseline,
        config=cfg,
    )

    assert np.array_equal(result.global_ignore, np.array([2, 3], dtype=int))
    global_keep = set(range(4)) - set(result.global_ignore.tolist())
    assert global_keep == {0, 1}


def test_compute_filtered_features_drops_unused_candidates():
    """Global mask should remove candidates never kept by FAST."""
    collection = FakeCollection(
        [
            np.array([10.0, 0.5, 0.0]),
            np.array([9.0, 0.1, 0.0]),
        ]
    )
    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=1)
    baseline = np.array([], dtype=int)

    result = compute_filtered_features_to_ignore(
        collection,
        num_features=3,
        base_ignore=baseline,
        config=cfg,
    )

    assert np.array_equal(result.global_ignore, np.array([1, 2], dtype=int))
