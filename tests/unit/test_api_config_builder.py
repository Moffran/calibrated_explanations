from __future__ import annotations

import pytest

from calibrated_explanations.api import config


def test_build_config_rethrows_non_exception_errors(monkeypatch):
    builder = config.ExplainerBuilder(model=object())

    def raise_keyboard_interrupt(_cfg):
        raise KeyboardInterrupt("boom")

    monkeypatch.setattr(config, "_perf_from_config", raise_keyboard_interrupt)
    with pytest.raises(KeyboardInterrupt):
        builder.build_config()
