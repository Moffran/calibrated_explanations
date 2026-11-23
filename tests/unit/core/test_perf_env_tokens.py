"""CE_CACHE / CE_PARALLEL token contract (#219, #225).

Both parsers share the enable/disable labels and honour them in any position.
Since v1.1.0 (#225) an unrecognised token raises ``ConfigurationError`` whose
``details`` name the variable and the offending token; v1.0.2 only warned.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder
from calibrated_explanations.cache import CacheConfig, CalibratorCache
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.config_manager import ConfigManager
from calibrated_explanations.parallel import ParallelConfig
from calibrated_explanations.utils.exceptions import ConfigurationError

PARSERS = [
    pytest.param("CE_CACHE", CacheConfig, id="cache"),
    pytest.param("CE_PARALLEL", ParallelConfig, id="parallel"),
]
ENABLE_LABELS = ["1", "true", "on", "yes", "enable", "ON", "Enable"]
DISABLE_LABELS = ["0", "false", "off", "no", "disable", "OFF", "Disable"]


def _parse(monkeypatch, env_var, config_cls, raw, base=None):
    monkeypatch.setenv(env_var, raw)
    return config_cls.from_env(base, config_manager=ConfigManager.from_sources())


@pytest.mark.parametrize(("env_var", "config_cls"), PARSERS)
@pytest.mark.parametrize("label", ENABLE_LABELS)
def test_enable_label_should_enable_in_any_position(monkeypatch, env_var, config_cls, label):
    other = "threads" if env_var == "CE_PARALLEL" else "max_items=5"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        leading = _parse(monkeypatch, env_var, config_cls, f"{label},{other}")
        trailing = _parse(monkeypatch, env_var, config_cls, f"{other}, {label}")

    assert leading.enabled is True
    assert trailing.enabled is True


@pytest.mark.parametrize(("env_var", "config_cls"), PARSERS)
@pytest.mark.parametrize("label", DISABLE_LABELS)
def test_disable_label_should_disable_in_any_position(monkeypatch, env_var, config_cls, label):
    base = config_cls(enabled=True)
    other = "threads" if env_var == "CE_PARALLEL" else "max_items=5"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = _parse(monkeypatch, env_var, config_cls, f"{other},{label}", base)

    assert cfg.enabled is False


@pytest.mark.parametrize(("env_var", "config_cls"), PARSERS)
def test_last_enable_disable_label_should_win(monkeypatch, env_var, config_cls):
    assert _parse(monkeypatch, env_var, config_cls, "on,off").enabled is False
    assert _parse(monkeypatch, env_var, config_cls, "off,on").enabled is True


def test_ce_parallel_should_enable_when_numeric_label_precedes_strategy(monkeypatch):
    cfg = _parse(monkeypatch, "CE_PARALLEL", ParallelConfig, "1,threads")

    assert cfg.enabled is True
    assert cfg.strategy == "threads"


INVALID_TOKENS = [
    pytest.param("CE_PARALLEL", "enable,thread", "thread", id="parallel-typo"),
    pytest.param("CE_PARALLEL", "enable,threads,worker=3", "worker=3", id="parallel-key-typo"),
    pytest.param(
        "CE_PARALLEL",
        "enable,threads,granularity=row",
        "granularity=row",
        id="parallel-granularity",
    ),
    pytest.param(
        "CE_PARALLEL",
        "enable,threads,force_serial=maybe",
        "force_serial=maybe",
        id="parallel-force-serial",
    ),
    pytest.param("CE_PARALLEL", "off,bogus", "bogus", id="parallel-after-disable"),
    pytest.param(
        "CE_PARALLEL", "threads,max_items=5", "max_items=5", id="parallel-cache-directive"
    ),
    pytest.param("CE_CACHE", "enable,max_item=5", "max_item=5", id="cache-typo"),
    pytest.param("CE_CACHE", "enable,unknown=true", "unknown=true", id="cache-unknown-key"),
    pytest.param("CE_CACHE", "on,threads", "threads", id="cache-parallel-directive"),
    pytest.param("CE_CACHE", "off,bogus", "bogus", id="cache-after-disable"),
]
CONFIG_CLASSES = {"CE_CACHE": CacheConfig, "CE_PARALLEL": ParallelConfig}


def _assert_token_error(error, env_var, token):
    assert error.details["env_var"] == env_var
    assert error.details["token"] == token
    assert env_var in str(error)
    assert repr(token) in str(error)


@pytest.mark.parametrize(("env_var", "raw", "token"), INVALID_TOKENS)
def test_unrecognised_token_should_raise_configuration_error(monkeypatch, env_var, raw, token):
    with pytest.raises(ConfigurationError, match="unrecognised token") as excinfo:
        _parse(monkeypatch, env_var, CONFIG_CLASSES[env_var], raw)

    _assert_token_error(excinfo.value, env_var, token)


@pytest.mark.parametrize(
    ("env_var", "raw", "token"),
    [
        pytest.param("CE_CACHE", "enable,max_items=on", "max_items=on", id="cache-numeric"),
        pytest.param(
            "CE_PARALLEL", "enable,threads,workers=two", "workers=two", id="parallel-numeric"
        ),
    ],
)
def test_malformed_numeric_directive_should_raise_with_token_details(
    monkeypatch, env_var, raw, token
):
    with pytest.raises(ConfigurationError) as excinfo:
        _parse(monkeypatch, env_var, CONFIG_CLASSES[env_var], raw)

    assert excinfo.value.details["env_var"] == env_var
    assert excinfo.value.details["token"] == token


def test_granularity_feature_should_remain_explicitly_rejected(monkeypatch):
    with pytest.raises(ConfigurationError, match="granularity='feature'") as excinfo:
        _parse(monkeypatch, "CE_PARALLEL", ParallelConfig, "enable,threads,granularity=feature")

    assert excinfo.value.details["token"] == "granularity=feature"
    assert excinfo.value.details["received"] == "feature"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("threads,on", {"enabled": True, "strategy": "threads"}),
        ("off,on,threads", {"enabled": True, "strategy": "threads"}),
        ("threads,granularity=instance", {"granularity": "instance"}),
        (
            "enable,processes,workers=3,min_batch=4,min_instances=5,tiny=6,"
            "instance_chunk=7,feature_chunk=8,task_bytes=9",
            {
                "strategy": "processes",
                "max_workers": 3,
                "min_batch_size": 4,
                "min_instances_for_parallel": 5,
                "tiny_workload_threshold": 6,
                "instance_chunk_size": 7,
                "feature_chunk_size": 8,
                "task_size_hint_bytes": 9,
            },
        ),
        ("joblib", {"strategy": "joblib"}),
        ("sequential", {"strategy": "sequential"}),
        ("auto", {"strategy": "auto", "enabled": False}),
    ],
)
def test_valid_parallel_directives_should_parse_without_warning(monkeypatch, raw, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = _parse(monkeypatch, "CE_PARALLEL", ParallelConfig, raw)

    for field, value in expected.items():
        assert getattr(cfg, field) == value


def test_valid_cache_directives_should_parse_without_warning(monkeypatch):
    raw = "max_items=9,ON,namespace=svc,version=v2,max_bytes=1024,ttl=1.5"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = _parse(monkeypatch, "CE_CACHE", CacheConfig, raw)

    assert (cfg.enabled, cfg.max_items, cfg.namespace, cfg.version) == (True, 9, "svc", "v2")
    assert (cfg.max_bytes, cfg.ttl_seconds) == (1024, 1.5)


@pytest.mark.parametrize(
    ("env_var", "raw", "token", "enable"),
    [
        pytest.param("CE_CACHE", "enable,max_item=5", "max_item=5", "perf_cache", id="cache"),
        pytest.param("CE_CACHE", "off,bogus", "bogus", None, id="cache-not-requested"),
        pytest.param(
            "CE_PARALLEL",
            "enable,threads,granularity=row",
            "granularity=row",
            "perf_parallel",
            id="parallel",
        ),
    ],
)
def test_build_config_should_expose_token_details_on_outer_error(
    monkeypatch, env_var, raw, token, enable
):
    monkeypatch.delenv("CE_CACHE", raising=False)
    monkeypatch.delenv("CE_PARALLEL", raising=False)
    monkeypatch.setenv(env_var, raw)
    builder = ExplainerBuilder(LogisticRegression())
    if enable == "perf_parallel":
        builder.perf_parallel(True, backend="threads")
    elif enable == "perf_cache":
        builder.perf_cache(True)

    with pytest.raises(ConfigurationError) as excinfo:
        builder.build_config()

    details = excinfo.value.details
    assert details["capability"] == ("cache" if env_var == "CE_CACHE" else "parallel")
    assert details["source"] == env_var
    assert token in details["cause"]
    assert details["env_var"] == env_var
    assert details["token"] == token
    _assert_token_error(excinfo.value.__cause__, env_var, token)


@pytest.fixture
def learner_and_data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 2))
    y = (x[:, 0] > 0).astype(int)
    return LogisticRegression(solver="liblinear").fit(x, y), x, y


@pytest.mark.parametrize(
    ("env_var", "raw", "token"),
    [
        pytest.param("CE_CACHE", "on,max_item=5", "max_item=5", id="cache"),
        pytest.param("CE_PARALLEL", "enable,thread", "thread", id="parallel"),
    ],
)
def test_plain_constructors_should_reject_unrecognised_token(
    monkeypatch, learner_and_data, env_var, raw, token
):
    learner, x, y = learner_and_data
    monkeypatch.setenv(env_var, raw)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.fit(x, y)  # perf env is consumed at calibration, not at fit

    with pytest.raises(ConfigurationError) as wrapped:
        wrapper.calibrate(x, y)
    with pytest.raises(ConfigurationError) as direct:
        CalibratedExplainer(learner, x, y, mode="classification")

    assert wrapper.calibrated is False
    _assert_token_error(wrapped.value, env_var, token)
    _assert_token_error(direct.value, env_var, token)


def test_explicit_perf_cache_should_keep_precedence_over_ce_cache(monkeypatch, learner_and_data):
    """An explicit primitive bypasses CE_CACHE, so the variable is not parsed."""
    learner, x, y = learner_and_data
    monkeypatch.setenv("CE_CACHE", "on,max_item=5")
    explicit = CalibratorCache(CacheConfig(enabled=True))

    explainer = CalibratedExplainer(learner, x, y, mode="classification", perf_cache=explicit)

    assert explainer.perf_cache is explicit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("force_serial=1", True), ("force_serial=yes", True), ("force_serial=off", False)],
)
def test_force_serial_should_accept_shared_labels(monkeypatch, raw, expected):
    base = ParallelConfig(force_serial_on_failure=not expected)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = _parse(monkeypatch, "CE_PARALLEL", ParallelConfig, raw, base)

    assert cfg.force_serial_on_failure is expected
