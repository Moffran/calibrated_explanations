"""CE_CACHE / CE_PARALLEL token contract (#219).

Both parsers share the enable/disable labels, honour them in any position, and
surface unrecognised tokens with a UserWarning plus an INFO log.
"""

from __future__ import annotations

import logging
import warnings

import pytest

from calibrated_explanations.cache import CacheConfig
from calibrated_explanations.core.config_manager import ConfigManager
from calibrated_explanations.parallel import ParallelConfig

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


@pytest.mark.parametrize(
    ("env_var", "config_cls", "raw", "token"),
    [
        pytest.param("CE_PARALLEL", ParallelConfig, "enable,thread", "thread", id="parallel-typo"),
        pytest.param(
            "CE_PARALLEL",
            ParallelConfig,
            "enable,threads,granularity=row",
            "granularity=row",
            id="parallel-granularity",
        ),
        pytest.param(
            "CE_PARALLEL",
            ParallelConfig,
            "enable,threads,force_serial=maybe",
            "force_serial=maybe",
            id="parallel-force-serial",
        ),
        pytest.param("CE_CACHE", CacheConfig, "enable,max_item=5", "max_item=5", id="cache-typo"),
    ],
)
def test_unrecognised_token_should_warn_and_log(
    monkeypatch, caplog, env_var, config_cls, raw, token
):
    caplog.set_level(logging.INFO, logger="calibrated_explanations.core.config_manager")

    with pytest.warns(UserWarning, match="unrecognised token") as record:
        cfg = _parse(monkeypatch, env_var, config_cls, raw)

    messages = [str(w.message) for w in record if "unrecognised token" in str(w.message)]
    assert len(messages) == 1
    assert env_var in messages[0]
    assert repr(token) in messages[0]
    assert "v1.1.0" in messages[0]
    assert "ConfigurationError" in messages[0]
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any(env_var in m and repr(token) in m for m in info)
    assert cfg.enabled is True


def test_unrecognised_token_should_leave_recognised_settings_applied(monkeypatch):
    with pytest.warns(UserWarning, match="unrecognised token"):
        cfg = _parse(monkeypatch, "CE_PARALLEL", ParallelConfig, "enable,threads,workers=3,bogus")

    assert cfg.strategy == "threads"
    assert cfg.max_workers == 3


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
