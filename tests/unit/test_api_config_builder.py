from __future__ import annotations

import pytest

from calibrated_explanations.api import config


def test_perf_cache_optional_params_are_applied():
    """perf_cache with all optional kwargs writes through to ExplainerConfig fields."""
    # Arrange
    builder = config.ExplainerBuilder(model=object())

    # Act
    builder.perf_cache(
        True,
        max_items=50,
        max_bytes=1024,
        namespace="ns",
        version="2",
        ttl=30.0,
    )
    cfg = builder.build_config()

    # Assert
    assert cfg.perf_cache_enabled is True
    assert cfg.perf_cache_max_items == 50
    assert cfg.perf_cache_max_bytes == 1024
    assert cfg.perf_cache_namespace == "ns"
    assert cfg.perf_cache_version == "2"
    assert cfg.perf_cache_ttl == 30.0


def test_perf_feature_filter_top_k_is_applied():
    """perf_feature_filter with per_instance_top_k writes through to config."""
    # Arrange
    builder = config.ExplainerBuilder(model=object())

    # Act
    builder.perf_feature_filter(True, per_instance_top_k=5)
    cfg = builder.build_config()

    # Assert
    assert cfg.perf_feature_filter_enabled is True
    assert cfg.perf_feature_filter_per_instance_top_k == 5


def test_build_config_rethrows_non_exception_errors(monkeypatch):
    builder = config.ExplainerBuilder(model=object())

    def raise_keyboard_interrupt(_cfg):
        raise KeyboardInterrupt("boom")

    monkeypatch.setattr(config, "_perf_from_config", raise_keyboard_interrupt)
    with pytest.raises(KeyboardInterrupt):
        builder.build_config()
