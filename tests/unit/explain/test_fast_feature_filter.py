"""Tests for feature-filter ConfigManager config resolution (ADR-034 Task 1)."""

from __future__ import annotations


from calibrated_explanations.core.config_manager import ConfigManager
from calibrated_explanations.core.explain._feature_filter import (
    FeatureFilterConfig,
)

_EMPTY_PYPROJECT: dict = {
    "plugins": {},
    "explanations": {},
    "intervals": {},
    "plots": {},
    "telemetry": {},
}


def make_mgr(env: dict) -> ConfigManager:
    return ConfigManager(env_snapshot=env, pyproject_snapshot=_EMPTY_PYPROJECT)


class TestFeatureFilterConfigResolution:
    """Config-resolution tests for FeatureFilterConfig.from_base_and_env (ADR-034 §6/§3)."""

    def test_should_enable_feature_filter_when_ce_feature_filter_is_on(self) -> None:
        """CE_FEATURE_FILTER=on must set enabled=True."""
        cfg = FeatureFilterConfig.from_base_and_env(
            config_manager=make_mgr({"CE_FEATURE_FILTER": "on"})
        )
        assert cfg.enabled is True

    def test_should_disable_feature_filter_when_ce_feature_filter_is_off(self) -> None:
        """CE_FEATURE_FILTER=off must set enabled=False."""
        cfg = FeatureFilterConfig.from_base_and_env(
            config_manager=make_mgr({"CE_FEATURE_FILTER": "off"})
        )
        assert cfg.enabled is False

    def test_should_set_top_k_from_ce_feature_filter_token(self) -> None:
        """CE_FEATURE_FILTER=on,top_k=4 must set per_instance_top_k=4."""
        cfg = FeatureFilterConfig.from_base_and_env(
            config_manager=make_mgr({"CE_FEATURE_FILTER": "on,top_k=4"})
        )
        assert cfg.enabled is True
        assert cfg.per_instance_top_k == 4

    def test_should_enable_strict_observability_when_ce_strict_observability_is_1(self) -> None:
        """CE_STRICT_OBSERVABILITY=1 must set strict_observability=True."""
        cfg = FeatureFilterConfig.from_base_and_env(
            config_manager=make_mgr({"CE_STRICT_OBSERVABILITY": "1"})
        )
        assert cfg.strict_observability is True

    def test_should_not_enable_strict_observability_when_env_absent(self) -> None:
        """Absent CE_STRICT_OBSERVABILITY must leave strict_observability=False."""
        cfg = FeatureFilterConfig.from_base_and_env(config_manager=make_mgr({}))
        assert cfg.strict_observability is False

    def test_should_use_default_config_when_no_config_manager_injected(self) -> None:
        """from_base_and_env() with no config_manager must not raise and return a valid FeatureFilterConfig."""
        # Calling without config_manager uses the module singleton transparently via public API.
        cfg = FeatureFilterConfig.from_base_and_env()
        assert isinstance(cfg, FeatureFilterConfig)

    def test_should_isolate_env_reads_from_post_snapshot_env_changes(self, monkeypatch) -> None:
        """Config snapshot must be captured at manager construction; later env changes must not affect existing configs."""
        mgr = make_mgr({"CE_FEATURE_FILTER": "on"})
        # Mutating the process env after snapshot construction must not affect mgr.
        monkeypatch.setenv("CE_FEATURE_FILTER", "off")

        cfg = FeatureFilterConfig.from_base_and_env(config_manager=mgr)
        assert (
            cfg.enabled is True
        ), "Snapshot must reflect the state at ConfigManager construction time"
