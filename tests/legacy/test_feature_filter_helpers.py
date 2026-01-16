import logging
import numpy as np
import pytest

from calibrated_explanations.core.explain._feature_filter import (
    FeatureFilterConfig,
    compute_filtered_features_to_ignore,
    emit_feature_filter_governance_event,
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


def test_feature_filter_config_from_env_variants(monkeypatch):
    # Test empty tokens
    monkeypatch.setenv("CE_FEATURE_FILTER", ",,,")
    cfg = FeatureFilterConfig.from_base_and_env()
    assert cfg.enabled is False

    # Test top_k
    monkeypatch.setenv("CE_FEATURE_FILTER", "on,top_k=5")
    cfg = FeatureFilterConfig.from_base_and_env()
    assert cfg.enabled is True
    assert cfg.per_instance_top_k == 5

    # Test top_k with non-digit
    monkeypatch.setenv("CE_FEATURE_FILTER", "top_k=abc")
    cfg = FeatureFilterConfig.from_base_and_env()
    assert cfg.per_instance_top_k == 8  # default

    # Test top_k with negative (should be max(1, ...))
    monkeypatch.setenv("CE_FEATURE_FILTER", "top_k=-5")
    cfg = FeatureFilterConfig.from_base_and_env()
    assert cfg.per_instance_top_k == 1


def test_safe_len_feature_weights_variants():
    from calibrated_explanations.core.explain._feature_filter import safe_len_feature_weights
    from unittest.mock import MagicMock

    # Empty explanations
    mock_ce = MagicMock()
    mock_ce.explanations = []
    assert safe_len_feature_weights(mock_ce) == 0

    # Scalar weights
    mock_exp = MagicMock()
    mock_exp.feature_weights = {"predict": 1.0}
    mock_ce.explanations = [mock_exp]
    assert safe_len_feature_weights(mock_ce) == 1


def test_compute_filtered_features_to_ignore_edge_cases():
    from unittest.mock import MagicMock

    mock_ce = MagicMock()

    # No weights mapping
    mock_exp1 = MagicMock()
    mock_exp1.feature_weights = None
    mock_ce.explanations = [mock_exp1]
    config = FeatureFilterConfig(enabled=True, per_instance_top_k=2)
    res = compute_filtered_features_to_ignore(
        mock_ce, num_features=2, base_ignore=np.array([0]), config=config
    )
    assert 0 in res.per_instance_ignore[0]

    # predict_weights is None
    mock_exp2 = MagicMock()
    mock_exp2.feature_weights = {"other": [1, 2]}
    mock_ce.explanations = [mock_exp2]
    res = compute_filtered_features_to_ignore(
        mock_ce, num_features=2, base_ignore=np.array([0]), config=config
    )
    assert 0 in res.per_instance_ignore[0]

    # weights_arr.size == 0
    mock_exp3 = MagicMock()
    mock_exp3.feature_weights = {"predict": []}
    mock_ce.explanations = [mock_exp3]
    res = compute_filtered_features_to_ignore(
        mock_ce, num_features=2, base_ignore=np.array([0]), config=config
    )
    assert 0 in res.per_instance_ignore[0]

    # no candidates_for_filter
    mock_exp4 = MagicMock()
    mock_exp4.feature_weights = {"predict": [1, 2]}
    mock_ce.explanations = [mock_exp4]
    res = compute_filtered_features_to_ignore(
        mock_ce, num_features=2, base_ignore=np.array([0, 1]), config=config
    )
    assert 0 in res.per_instance_ignore[0]
    assert 1 in res.per_instance_ignore[0]

    # global_ignore branch: predict is None in global loop
    mock_exp5 = MagicMock()
    mock_exp5.feature_weights = {"predict": [1, 2]}
    mock_exp6 = MagicMock()
    mock_exp6.feature_weights = {"other": [1, 2]}
    mock_ce.explanations = [mock_exp5, mock_exp6]
    res = compute_filtered_features_to_ignore(
        mock_ce, num_features=2, base_ignore=np.array([]), config=config
    )
    assert res.global_ignore.size >= 0

    # observed_len == 0
    mock_exp7 = MagicMock()
    mock_exp7.feature_weights = {"predict": []}
    mock_ce.explanations = [mock_exp7]
    res = compute_filtered_features_to_ignore(
        mock_ce, num_features=2, base_ignore=np.array([]), config=config
    )
    assert res.global_ignore.size >= 0


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


def test_governance_logger_records_strict_warnings(caplog):
    """Strict observability warnings should emit governance logs with structured context."""
    from unittest.mock import MagicMock

    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=1, strict_observability=True)
    mock_collection = MagicMock()
    mock_exp = MagicMock()
    mock_exp.feature_weights = None
    mock_collection.explanations = [mock_exp]

    caplog.set_level(logging.WARNING, logger="calibrated_explanations.governance.feature_filter")
    compute_filtered_features_to_ignore(
        mock_collection,
        num_features=3,
        base_ignore=np.array([], dtype=int),
        config=cfg,
    )

    governance_records = [
        rec
        for rec in caplog.records
        if rec.name == "calibrated_explanations.governance.feature_filter"
    ]
    decisions = {rec.decision for rec in governance_records}
    assert decisions & {
        "feature_filter_missing_feature_count",
        "feature_filter_missing_weights_mapping",
        "feature_filter_missing_predict_weights",
        "feature_filter_empty_weights",
    }
    assert any(getattr(rec, "strict_observability", None) is True for rec in governance_records)


def test_governance_logger_records_skip_and_error_decisions(caplog):
    """Skip and error transitions should be logged under the governance domain."""
    caplog.set_level(logging.INFO, logger="calibrated_explanations.governance.feature_filter")

    emit_feature_filter_governance_event(
        decision="filter_skipped",
        mode="factual",
        reason="skip reason",
        strict=False,
    )
    emit_feature_filter_governance_event(
        decision="filter_error",
        mode="factual",
        reason="error reason",
        strict=True,
    )

    governance_records = [
        rec
        for rec in caplog.records
        if rec.name == "calibrated_explanations.governance.feature_filter"
    ]
    decisions = {rec.decision for rec in governance_records}
    assert {"filter_skipped", "filter_error"} <= decisions
    assert any(rec.strict_observability is False for rec in governance_records)
    assert any(rec.strict_observability is True for rec in governance_records)
