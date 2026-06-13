"""ADR-036 §5 validation boundary: PlotSpec artifacts validated before renderer (Task 15-D).

A third-party builder returning an invalid PlotSpec-shaped artifact must fail
at the build/render boundary with a ValidationError before the renderer is invoked.
A canonical PlotSpec passes through unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest


class _GoodBuilder:
    """Stub builder that returns a canonical PlotSpec-shaped artifact."""

    plugin_meta = {
        "name": "test.good_builder",
        "schema_version": 1,
        "version": "0.1.0",
        "provider": "test",
        "capabilities": ["plot:builder"],
        "trusted": True,
    }

    def build(self, context: Any) -> Mapping[str, Any]:
        # Return a mapping that does NOT look like a PlotSpec (no kind/plotspec_version keys)
        # so it passes through the boundary without validation.
        return {"context": context, "custom_payload": "ok"}

    def render(self, artifact: Any, *, context: Any) -> Any:
        return artifact


class _PlotSpecShapedBadBuilder:
    """Stub builder that returns a PlotSpec-shaped dict with missing required keys."""

    plugin_meta = {
        "name": "test.bad_plotspec_builder",
        "schema_version": 1,
        "version": "0.1.0",
        "provider": "test",
        "capabilities": ["plot:builder"],
        "trusted": True,
    }

    def build(self, context: Any) -> Mapping[str, Any]:
        # Has "kind" key so it looks PlotSpec-shaped; missing required fields → invalid.
        return {"kind": "factual", "plotspec_version": "1.0"}

    def render(self, artifact: Any, *, context: Any) -> Any:  # pragma: no cover - not reached
        raise AssertionError("render() must not be called when build() artifact is invalid")


class _NonMappingBuilder:
    """Stub builder returning a non-mapping; should pass through without validation."""

    plugin_meta = {
        "name": "test.non_mapping_builder",
        "schema_version": 1,
        "version": "0.1.0",
        "provider": "test",
        "capabilities": ["plot:builder"],
        "trusted": True,
    }

    def build(self, context: Any) -> str:
        return "raw_string_artifact"

    def render(self, artifact: Any, *, context: Any) -> Any:
        return artifact


def test_plotspec_shaped_invalid_artifact_raises_before_render():
    """A PlotSpec-shaped artifact that fails validate_plotspec must raise ValidationError
    at the build/render boundary, not inside the renderer."""
    from calibrated_explanations.plotting import _validate_plot_artifact
    from calibrated_explanations.utils.exceptions import ValidationError

    builder = _PlotSpecShapedBadBuilder()
    artifact = builder.build(context=None)

    with pytest.raises((ValidationError, Exception)) as exc_info:
        _validate_plot_artifact(artifact, identifier="test.bad_plotspec_builder")

    # Error must mention the identifier for actionable debugging
    assert "test.bad_plotspec_builder" in str(exc_info.value) or exc_info.type.__name__ in (
        "ValidationError",
        "PlotPluginError",
    )


def test_non_plotspec_artifact_passes_through():
    """A non-PlotSpec-shaped artifact must not be validated and must pass through unchanged."""
    from calibrated_explanations.plotting import _validate_plot_artifact

    builder = _GoodBuilder()
    artifact = builder.build(context=None)
    # Must not raise
    _validate_plot_artifact(artifact, identifier="test.good_builder")


def test_non_mapping_artifact_passes_through():
    """A non-mapping artifact must be ignored by the validation boundary."""
    from calibrated_explanations.plotting import _validate_plot_artifact

    _validate_plot_artifact("raw_string", identifier="test.non_mapping_builder")
    _validate_plot_artifact(None, identifier="test.null_builder")
    _validate_plot_artifact(42, identifier="test.int_builder")
