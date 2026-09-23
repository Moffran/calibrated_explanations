"""Minimal tests for the config builder scaffolding.

These tests intentionally avoid changing public APIs by exercising only
the internal/private `_from_config` helper and the fluent builder.
"""

from __future__ import annotations

from dataclasses import is_dataclass

import pytest
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations.api.config import ExplainerBuilder, ExplainerConfig
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.utils.exceptions import ConfigurationError


def test_explainer_config_dataclass_and_defaults():
    model = RandomForestClassifier()
    cfg = ExplainerConfig(model=model)
    assert is_dataclass(cfg)
    assert not hasattr(cfg, "task"), "task field was removed in v0.11.3"
    assert not hasattr(cfg, "parallel_workers"), "parallel_workers field was removed in v0.11.3"
    assert cfg.low_high_percentiles == (5, 95)
    assert cfg.threshold is None
    assert cfg.preprocessor is None
    assert cfg.auto_encode == "auto"
    assert cfg.unseen_category_policy == "error"


def test_explainer_builder_fluent_roundtrip():
    model = RandomForestClassifier()
    b = (
        ExplainerBuilder(model)
        .low_high_percentiles((10, 90))
        .threshold(0.7)
        .preprocessor(None)
        .auto_encode("auto")
        .unseen_category_policy("ignore")
    )
    cfg = b.build_config()
    assert isinstance(cfg, ExplainerConfig)
    assert cfg.model is model
    assert cfg.low_high_percentiles == (10, 90)
    assert cfg.threshold == 0.7
    assert cfg.unseen_category_policy == "ignore"


def test_explainer_builder_has_no_task_or_parallel_workers_methods():
    model = RandomForestClassifier()
    b = ExplainerBuilder(model)
    assert not hasattr(b, "task"), "task() method was removed in v0.11.3"
    assert not hasattr(b, "parallel_workers"), "parallel_workers() method was removed in v0.11.3"


def test_wrap_from_config_applies_defaults(monkeypatch):
    # Configure defaults
    model = RandomForestClassifier()
    cfg = ExplainerConfig(model=model, low_high_percentiles=(10, 90), threshold=0.3)
    w = WrapCalibratedExplainer.from_config(cfg)

    # Monkeypatch underlying explainer to capture kwargs passed through
    class DummyExplainer:
        def explain_factual(self, x, **kwargs):  # noqa: D401
            return kwargs

        def explore_alternatives(self, x, **kwargs):  # noqa: D401
            return kwargs

    w.fitted = True
    w.calibrated = True
    w.explainer = DummyExplainer()  # type: ignore[assignment]

    # factual inherits defaults
    out = w.explain_factual([[0.0]])
    assert out["low_high_percentiles"] == (10, 90)
    assert out["threshold"] == 0.3

    # explicit kwargs override config defaults
    out2 = w.explain_factual([[0.0]], low_high_percentiles=(5, 95), threshold=None)
    assert out2["low_high_percentiles"] == (5, 95)
    assert out2["threshold"] is None

    # alternatives also inherit
    out3 = w.explore_alternatives([[0.0]])
    assert out3["low_high_percentiles"] == (10, 90)
    assert out3["threshold"] == 0.3


def test_wrap_from_config_applies_defaults_fast():
    # Configure defaults
    model = RandomForestClassifier()
    cfg = ExplainerConfig(model=model, low_high_percentiles=(20, 80), threshold=0.4)
    w = WrapCalibratedExplainer.from_config(cfg)

    class DummyExplainerFast:
        def explain_fast(self, x, **kwargs):  # noqa: D401
            return kwargs

    w.fitted = True
    w.calibrated = True
    w.explainer = DummyExplainerFast()  # type: ignore[assignment]

    out = w.explain_fast([[0.0]])
    assert out["low_high_percentiles"] == (20, 80)
    assert out["threshold"] == 0.4


def test_explainer_builder_perf_options(monkeypatch: pytest.MonkeyPatch):
    model = RandomForestClassifier()
    builder = ExplainerBuilder(model)
    builder = builder.perf_cache(
        True,
        max_items=10,
        max_bytes=1024,
        namespace="ns",
        version="1.2.3",
        ttl=60.0,
    )
    builder = builder.perf_parallel(
        True,
        backend="threads",
        workers=4,
        min_batch=8,
        min_instances=16,
        tiny_workload=32,
        granularity="instance",
    )
    builder = builder.perf_feature_filter(True, per_instance_top_k=3)
    builder = builder.perf_telemetry(lambda event, payload: None)

    cfg = builder.build_config()
    assert cfg.perf_cache_enabled is True
    assert cfg.perf_cache_max_items == 10
    assert cfg.perf_cache_max_bytes == 1024
    assert cfg.perf_cache_namespace == "ns"
    assert cfg.perf_parallel_backend == "threads"
    assert cfg.perf_parallel_workers == 4
    assert cfg.perf_parallel_min_batch == 8
    assert cfg.perf_parallel_min_instances == 16
    assert cfg.perf_parallel_tiny_workload == 32
    assert cfg.perf_feature_filter_enabled is True
    assert cfg.perf_feature_filter_per_instance_top_k == 3


def test_explainer_builder_rejects_removed_feature_parallel_granularity():
    model = RandomForestClassifier()
    builder = ExplainerBuilder(model)

    with pytest.raises(ConfigurationError):
        builder.perf_parallel(True, granularity="feature")


def test_explainer_builder_perf_factory_failure(monkeypatch: pytest.MonkeyPatch):
    model = RandomForestClassifier()
    builder = ExplainerBuilder(model).perf_cache(True)

    def boom(cfg):
        raise RuntimeError("perf factory broke")

    monkeypatch.setattr("calibrated_explanations.api.config._perf_from_config", boom)
    with pytest.raises(ConfigurationError, match="perf factory broke") as excinfo:
        builder.build_config()

    assert excinfo.value.details == {
        "capability": "perf_primitives",
        "source": "factory",
        "cause": "RuntimeError: perf factory broke",
    }
    assert isinstance(excinfo.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("env_var", "env_value", "enable", "capability", "token"),
    [
        ("CE_CACHE", "enable,max_items=abc", "perf_cache", "cache", "max_items=abc"),
        ("CE_CACHE", "ttl=soon", "perf_cache", "cache", "ttl=soon"),
        ("CE_PARALLEL", "enable,threads,workers=two", "perf_parallel", "parallel", "workers=two"),
    ],
)
def test_build_config_should_fail_closed_when_requested_perf_env_is_malformed(
    monkeypatch: pytest.MonkeyPatch, env_var, env_value, enable, capability, token
):
    monkeypatch.setenv(env_var, env_value)
    builder = ExplainerBuilder(RandomForestClassifier())
    getattr(builder, enable)(True)

    with pytest.raises(ConfigurationError) as excinfo:
        builder.build_config()

    details = excinfo.value.details
    assert details["capability"] == capability
    assert details["source"] == env_var
    assert token in details["cause"]
    assert excinfo.value.__cause__.details["env_var"] == env_var
    assert excinfo.value.__cause__.details["token"] == token


def test_from_config_should_fail_closed_for_hand_built_invalid_parallel_config():
    """A hand-built ExplainerConfig bypasses builder validation but still fails closed."""
    cfg = ExplainerConfig(
        model=RandomForestClassifier(),
        perf_parallel_enabled=True,
        perf_parallel_granularity="feature",  # type: ignore[arg-type]
    )

    with pytest.raises(ConfigurationError, match="parallel executor") as excinfo:
        WrapCalibratedExplainer.from_config(cfg)

    assert excinfo.value.details["capability"] == "parallel"
    assert excinfo.value.details["source"] == "config"
    assert "granularity" in excinfo.value.details["cause"]


@pytest.mark.parametrize(
    ("builder_cache", "builder_parallel", "env", "expected"),
    [
        (False, False, {}, {"cache": None, "parallel": None}),
        (True, True, {}, {"cache": "config", "parallel": "config"}),
        (False, False, {"CE_CACHE": "on"}, {"cache": "env", "parallel": None}),
        (False, False, {"CE_PARALLEL": "enable,threads"}, {"cache": None, "parallel": "env"}),
        (
            True,
            True,
            {"CE_CACHE": "off", "CE_PARALLEL": "off"},
            {"cache": "disabled_by_env", "parallel": "disabled_by_env"},
        ),
    ],
)
def test_from_config_should_apply_env_over_builder_and_record_activation(
    monkeypatch: pytest.MonkeyPatch, caplog, builder_cache, builder_parallel, env, expected
):
    import logging
    import warnings

    # Start from a clean perf environment; other suites may leave these set.
    monkeypatch.delenv("CE_CACHE", raising=False)
    monkeypatch.delenv("CE_PARALLEL", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    builder = ExplainerBuilder(RandomForestClassifier()).perf_cache(builder_cache)
    builder.perf_parallel(builder_parallel, backend="threads")

    with (
        warnings.catch_warnings(record=True) as recorded,
        caplog.at_level(logging.INFO, logger="calibrated_explanations"),
    ):
        warnings.simplefilter("always")
        cfg = builder.build_config()
        wrapper = WrapCalibratedExplainer.from_config(cfg)

    assert cfg.perf_factory.activation == expected
    assert wrapper.perf_cache.enabled is (expected["cache"] in {"config", "env"})
    assert wrapper.parallel_executor.config.enabled is (expected["parallel"] in {"config", "env"})
    assert not [w for w in recorded if issubclass(w.category, UserWarning)]
    activation_logs = [r for r in caplog.records if "Performance primitives" in r.getMessage()]
    assert bool(activation_logs) is any(expected.values())


def test_perf_factory_make_parallel_backend_alias():
    model = RandomForestClassifier()
    cfg = ExplainerBuilder(model).build_config()
    factory = cfg.perf_factory
    cache = factory.make_cache()
    backend = factory.make_parallel_backend(cache)
    assert backend is not None


@pytest.mark.parametrize(
    ("builder_parallel", "env", "source"),
    [
        (True, {}, "config"),
        (False, {"CE_PARALLEL": "1"}, "CE_PARALLEL"),
        (False, {"CE_PARALLEL": "enable,workers=2"}, "CE_PARALLEL"),
        (True, {"CE_PARALLEL": "auto"}, "config"),
    ],
)
def test_build_config_should_fail_fast_when_parallel_enabled_with_auto_strategy(
    monkeypatch: pytest.MonkeyPatch, builder_parallel, env, source
):
    """ADR-004: enabling parallel without an explicit backend fails at build time."""
    monkeypatch.delenv("CE_PARALLEL", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    builder = ExplainerBuilder(RandomForestClassifier()).perf_parallel(builder_parallel)

    with pytest.raises(ConfigurationError, match="strategy='auto'") as excinfo:
        builder.build_config()

    assert excinfo.value.details["capability"] == "parallel"
    assert excinfo.value.details["source"] == source
    assert excinfo.value.__cause__.details["received"] == "auto"


def test_from_config_should_fail_fast_for_hand_built_enabled_auto_parallel_config(
    monkeypatch: pytest.MonkeyPatch,
):
    """A hand-built config with parallel enabled and the default backend is rejected."""
    monkeypatch.delenv("CE_PARALLEL", raising=False)
    cfg = ExplainerConfig(model=RandomForestClassifier(), perf_parallel_enabled=True)

    with pytest.raises(ConfigurationError, match="strategy='auto'"):
        WrapCalibratedExplainer.from_config(cfg)


def test_build_config_should_accept_auto_default_while_parallel_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """The default "auto" backend stays valid as long as parallel execution is off."""
    monkeypatch.delenv("CE_PARALLEL", raising=False)

    cfg = ExplainerBuilder(RandomForestClassifier()).perf_parallel(False).build_config()
    wrapper = WrapCalibratedExplainer.from_config(cfg)

    assert cfg.perf_parallel_backend == "auto"
    assert wrapper.parallel_executor.config.enabled is False
