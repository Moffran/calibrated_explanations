from __future__ import annotations

from typing import Any, Mapping

import pytest

from calibrated_explanations.plugins.base import validate_plugin_meta
from calibrated_explanations.plugins.manager import PluginManager
from calibrated_explanations.plugins.registry import (
    DefaultPluginTrustPolicy,
    find_explanation_plugin_for,
    get_trust_policy,
    register_explanation_plugin,
    set_trust_policy,
)
from calibrated_explanations.utils.exceptions import ValidationError
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
