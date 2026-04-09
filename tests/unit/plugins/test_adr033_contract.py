from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import patch

import pytest

from calibrated_explanations.plugins.cli import cmd_list
from calibrated_explanations.plugins.base import validate_plugin_meta
from calibrated_explanations.plugins.manager import PluginManager
from calibrated_explanations.plugins.registry import (
    DefaultPluginTrustPolicy,
    find_explanation_plugin_for,
    load_entrypoint_plugins,
    get_trust_policy,
    register_explanation_plugin,
    set_trust_policy,
)
from calibrated_explanations.utils.exceptions import MissingExtensionError, ValidationError
from tests.support.registry_helpers import clear_explanation_plugins


class DummyExplanationPlugin:
    def __init__(self, name: str, *, modalities: tuple[str, ...], priority: int = 0) -> None:
        self.plugin_meta = {
            "schema_version": 1,
            "name": name,
            "version": "0.1",
            "provider": "tests",
            "capabilities": ("explain",),
            "modes": ("factual",),
            "tasks": ("classification",),
            "dependencies": (),
            "data_modalities": modalities,
            "priority": priority,
            "trusted": False,
        }

    def supports(self, model: Any) -> bool:
        return True

    def supports_mode(self, mode: str, *, task: str) -> bool:
        return mode == "factual" and task == "classification"

    def initialize(self, context: Any) -> None:
        return None

    def explain_batch(self, x: Any, request: Any) -> Any:
        raise NotImplementedError


class AllowAllPolicy:
    def is_denied(self, identifier: str, *, denylist: set[str]) -> bool:
        return False

    def is_trusted(
        self,
        *,
        meta: Mapping[str, Any],
        identifier: str,
        source: str,
        trusted_identifiers: set[str],
    ) -> bool:
        return True


@pytest.fixture(autouse=True)
def reset_registry_and_policy():
    clear_explanation_plugins()
    set_trust_policy(DefaultPluginTrustPolicy())
    yield
    clear_explanation_plugins()
    set_trust_policy(DefaultPluginTrustPolicy())


def test_validate_plugin_meta_defaults_modality_and_api_version():
    meta = {
        "schema_version": 1,
        "name": "tests.base.meta",
        "version": "0.1",
        "provider": "tests",
        "capabilities": ("explain",),
    }
    validate_plugin_meta(meta)
    assert meta["plugin_api_version"] == "1.0"
    assert meta["data_modalities"] == ("tabular",)


def test_validate_plugin_meta_rejects_invalid_api_version():
    meta = {
        "schema_version": 1,
        "name": "tests.bad.version",
        "version": "0.1",
        "provider": "tests",
        "capabilities": ("explain",),
        "plugin_api_version": "v1",
    }
    with pytest.raises(ValidationError, match="MAJOR.MINOR"):
        validate_plugin_meta(meta)


def test_validate_plugin_meta_normalizes_modality_aliases():
    meta = {
        "schema_version": 1,
        "name": "tests.modality.alias",
        "version": "0.1",
        "provider": "tests",
        "capabilities": ("explain",),
        "data_modalities": ("image", "time-series"),
    }
    validate_plugin_meta(meta)
    assert meta["data_modalities"] == ("vision", "timeseries")


def test_find_explanation_plugin_for_raises_on_ambiguous_top_priority():
    p1 = DummyExplanationPlugin("tests.modality.one", modalities=("vision",), priority=10)
    p2 = DummyExplanationPlugin("tests.modality.two", modalities=("vision",), priority=10)
    register_explanation_plugin("tests.modality.one", p1, source="builtin")
    register_explanation_plugin("tests.modality.two", p2, source="builtin")

    with pytest.raises(ValidationError, match="Ambiguous explanation plugin resolution"):
        find_explanation_plugin_for(
            "vision",
            mode="factual",
            task="classification",
            model=object(),
            trusted_only=True,
        )


def test_find_explanation_plugin_for_accepts_explicit_identifier_override():
    plugin = DummyExplanationPlugin("tests.modality.explicit", modalities=("vision",), priority=1)
    register_explanation_plugin("tests.modality.explicit", plugin, source="builtin")
    identifier, resolved = find_explanation_plugin_for(
        "vision",
        mode="factual",
        task="classification",
        model=object(),
        trusted_only=True,
        identifier="tests.modality.explicit",
    )
    assert identifier == "tests.modality.explicit"
    assert resolved is plugin


def test_plugin_manager_can_consume_policy_and_registry_uses_it():
    policy = AllowAllPolicy()
    PluginManager(object(), policy=policy)
    assert get_trust_policy() is policy

    plugin = DummyExplanationPlugin("tests.policy.allowed", modalities=("tabular",))
    descriptor = register_explanation_plugin("tests.policy.allowed", plugin, source="manual")
    assert descriptor.trusted is True


def test_missing_extension_error_is_importerror():
    sys.modules.pop("calibrated_explanations.vision", None)
    with pytest.raises(MissingExtensionError) as exc_info:
        importlib.import_module("calibrated_explanations.vision")
    assert isinstance(exc_info.value, ImportError)


def test_missing_extension_error_audio():
    sys.modules.pop("calibrated_explanations.audio", None)
    with pytest.raises(MissingExtensionError) as exc_info:
        importlib.import_module("calibrated_explanations.audio")
    assert isinstance(exc_info.value, ImportError)


def test_cli_list_modality_filter(capsys):
    vision_plugin = DummyExplanationPlugin("tests.modality.vision", modalities=("vision",))
    tabular_plugin = DummyExplanationPlugin("tests.modality.tabular", modalities=("tabular",))
    register_explanation_plugin("tests.modality.vision", vision_plugin, source="builtin")
    register_explanation_plugin("tests.modality.tabular", tabular_plugin, source="builtin")

    args = SimpleNamespace(
        kind="explanations",
        trusted_only=False,
        verbose=False,
        plots=False,
        include_skipped=False,
        modality="vision",
    )
    with patch("calibrated_explanations.plugins.cli.is_identifier_denied", return_value=False):
        exit_code = cmd_list(args)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "tests.modality.vision" in output
    assert "tests.modality.tabular" not in output


def test_cli_list_modality_filter_invalid_token():
    args = SimpleNamespace(
        kind="explanations",
        trusted_only=False,
        verbose=False,
        plots=False,
        include_skipped=False,
        modality="not-a-modality",
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_list(args)
    assert exc_info.value.code == 1


def test_deprecation_warning_on_plugin_without_modality(monkeypatch):
    class EntryPointPlugin:
        plugin_meta = {
            "schema_version": 1,
            "name": "tests.modality.entrypoint.no_modality",
            "version": "0.1",
            "provider": "tests",
            "capabilities": ("explain",),
            "modes": ("factual",),
            "tasks": ("classification",),
            "dependencies": (),
            "trusted": False,
        }

        def supports(self, model: Any) -> bool:
            return True

        def supports_mode(self, mode: str, *, task: str) -> bool:
            return mode == "factual" and task == "classification"

        def initialize(self, context: Any) -> None:
            return None

        def explain_batch(self, x: Any, request: Any) -> Any:
            raise NotImplementedError

    class EntryPoint:
        name = "tests.modality.entrypoint.no_modality"
        module = "tests.modality"
        attr = "EntryPointPlugin"

        def load(self):
            return EntryPointPlugin()

    class EntryPoints:
        def select(self, *, group):
            return [EntryPoint()] if group == "calibrated_explanations.plugins" else []

    monkeypatch.setattr(
        "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
        lambda: EntryPoints(),
    )

    with pytest.warns(DeprecationWarning, match="does not declare 'data_modalities'"):
        load_entrypoint_plugins(include_untrusted=True)


def test_find_explanation_plugin_for_accepts_alias():
    plugin = DummyExplanationPlugin(
        "tests.modality.alias_match", modalities=("vision",), priority=1
    )
    register_explanation_plugin("tests.modality.alias_match", plugin, source="builtin")
    identifier, resolved = find_explanation_plugin_for(
        "image",
        mode="factual",
        task="classification",
        model=object(),
        trusted_only=True,
    )
    assert identifier == "tests.modality.alias_match"
    assert resolved is plugin
