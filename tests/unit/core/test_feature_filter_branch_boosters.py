from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.core.explain._feature_filter import (
    FeatureFilterConfig,
    compute_filtered_features_to_ignore,
    safe_len_feature_weights,
)


class ExpStub:
    def __init__(self, feature_weights):
        self.feature_weights = feature_weights


class CollStub:
    def __init__(self, explanations):
        self.explanations = explanations


def test_feature_filter_config_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    base = FeatureFilterConfig(enabled=False, per_instance_top_k=8, strict_observability=False)

    monkeypatch.setenv("CE_STRICT_OBSERVABILITY", "true")
    monkeypatch.setenv("CE_FEATURE_FILTER", "on,top_k=3")
    cfg = FeatureFilterConfig.from_base_and_env(base)
    assert cfg.enabled is True
    assert cfg.per_instance_top_k == 3
    assert cfg.strict_observability is True

    monkeypatch.setenv("CE_FEATURE_FILTER", "off")
    cfg = FeatureFilterConfig.from_base_and_env(base)
    assert cfg.enabled is False

    monkeypatch.setenv("CE_FEATURE_FILTER", "on")
    cfg = FeatureFilterConfig.from_base_and_env(base)
    assert cfg.enabled is True

    monkeypatch.setenv("CE_FEATURE_FILTER", " , ")
    cfg = FeatureFilterConfig.from_base_and_env(base)
    assert cfg.per_instance_top_k == 8

    monkeypatch.setenv("CE_FEATURE_FILTER", "enable,top_k=-0,top_k=xyz")
    cfg = FeatureFilterConfig.from_base_and_env(base)
    assert cfg.enabled is True
    assert cfg.per_instance_top_k == 1


def test_safe_len_feature_weights_empty_and_scalar() -> None:
    empty = CollStub([])
    assert safe_len_feature_weights(empty) == 0

    missing_predict = CollStub([ExpStub({})])
    assert safe_len_feature_weights(missing_predict) == 0

    scalar_predict = CollStub([ExpStub({"predict": 1.2})])
    assert safe_len_feature_weights(scalar_predict) == 1


def test_compute_filtered_features_disabled_and_empty_batch() -> None:
    cfg = FeatureFilterConfig(enabled=False, per_instance_top_k=2)
    coll = CollStub([])
    result = compute_filtered_features_to_ignore(
        coll,
        num_features=4,
        base_ignore=np.asarray([1], dtype=int),
        config=cfg,
    )
    assert result.global_ignore.tolist() == [1]
    assert result.per_instance_ignore == []


def test_compute_filtered_features_strict_missing_feature_count() -> None:
    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=2, strict_observability=True)
    coll = CollStub([ExpStub({"predict": None})])
    result = compute_filtered_features_to_ignore(
        coll,
        num_features=None,
        base_ignore=np.asarray([0], dtype=int),
        config=cfg,
    )
    assert result.global_ignore.tolist() == [0]
    assert len(result.per_instance_ignore) == 1
    assert result.per_instance_ignore[0].tolist() == [0]


def test_compute_filtered_features_missing_mappings_and_sizes() -> None:
    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=1, strict_observability=True)
    coll = CollStub(
        [
            ExpStub({"predict": [1.0, 0.5, 0.2]}),  # ensures feature count inference
            ExpStub(None),  # missing mapping
            ExpStub({}),  # missing predict
            ExpStub({"predict": []}),  # empty array
            ExpStub({"predict": [1.0]}),  # shorter than num_features
            ExpStub({"predict": [0.5, 0.4, 0.3, 0.2, 0.1]}),  # longer than num_features
        ]
    )
    result = compute_filtered_features_to_ignore(
        coll,
        num_features=3,
        base_ignore=np.asarray([0], dtype=int),
        config=cfg,
    )
    assert len(result.per_instance_ignore) == 6
    assert 0 in set(result.global_ignore.tolist())
    for per_instance in result.per_instance_ignore:
        assert 0 in set(per_instance.tolist())


def test_compute_filtered_features_global_ignore_when_all_features_baselined() -> None:
    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=2)
    coll = CollStub([ExpStub({"predict": [0.9, 0.1]}), ExpStub({"predict": [0.1, 0.9]})])
    result = compute_filtered_features_to_ignore(
        coll,
        num_features=2,
        base_ignore=np.asarray([0, 1], dtype=int),
        config=cfg,
    )
    assert sorted(result.global_ignore.tolist()) == [0, 1]


def test_compute_filtered_features_observed_count_edge_cases() -> None:
    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=1)
    coll = CollStub(
        [
            ExpStub({"predict": [1.0]}),  # padding path (< num_features)
            ExpStub({"predict": [1.0, 0.5, 0.2, 0.1]}),  # truncation path (> num_features)
            ExpStub({"predict": []}),  # observed_len == 0
            ExpStub({}),  # missing predict in global observed loop
        ]
    )
    result = compute_filtered_features_to_ignore(
        coll,
        num_features=3,
        base_ignore=np.asarray([], dtype=int),
        config=cfg,
    )
    assert len(result.per_instance_ignore) == 4
    assert isinstance(result.global_ignore, np.ndarray)
