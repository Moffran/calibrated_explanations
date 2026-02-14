"""Tests for PlotSpec serializers: to_dict and from_dict.

Focuses on semantic assertions (domain invariants) and roundtrip
verification. See .github/tests-guidance.md for patterns.
"""

import pytest

from calibrated_explanations.viz import (
    plotspec_from_dict,
    validate_plotspec,
    global_plotspec_to_dict,
    PlotKindRegistry,
)
from calibrated_explanations.viz import (
    PlotSpec,
    IntervalHeaderSpec,
    BarHPanelSpec,
    BarItem,
    GlobalPlotSpec,
    GlobalSpec,
    TriangularPlotSpec,
    TriangularSpec,
)
from calibrated_explanations.viz.plotspec import SaveBehavior
from calibrated_explanations.viz.serializers import (
    PLOTSPEC_VERSION,
    global_plotspec_from_dict,
    plotspec_to_dict,
    triangular_plotspec_to_dict,
    triangular_plotspec_from_dict,
)


def test_validate_rejects_bad_payload():
    """Verify that validation rejects malformed payloads."""
    from calibrated_explanations.utils.exceptions import ValidationError

    with pytest.raises(ValidationError):
        validate_plotspec(["not", "a", "dict"])
    with pytest.raises(ValidationError):
        validate_plotspec({})
    with pytest.raises(ValidationError):
        validate_plotspec({"plotspec_version": "1.0.0", "body": {"bars": "notalist"}})
    with pytest.raises(ValidationError):
        validate_plotspec({"plotspec_version": "1.0.0", "title": "missing body"})
    with pytest.raises(ValidationError):
        validate_plotspec(
            {
                "plotspec_version": "1.0.0",
                "body": {"bars": [{"label": "a"}]},
            }
        )


def test_interval_header_spec_optional_fields():
    """Verify that optional fields are preserved in IntervalHeaderSpec."""
    header = IntervalHeaderSpec(
        pred=0.4,
        low=0.2,
        high=0.9,
        xlim=(0.0, 1.0),
        xlabel="prediction",
        ylabel="probability",
        dual=False,
        neg_label="negative",
        pos_label="positive",
        uncertainty_color="#ccc",
        uncertainty_alpha=0.75,
    )

    # Semantic assertions: all fields are preserved
    assert header.dual is False
    assert header.xlabel == "prediction"
    assert header.neg_label == "negative"
    assert header.uncertainty_alpha == 0.75

    # Domain invariant: bounds ordering
    assert header.low <= header.pred <= header.high


def test_bar_item_and_panel_configuration():
    """Verify BarItem and BarHPanelSpec configuration."""
    items = [
        BarItem(
            label="feature",
            value=0.35,
            interval_low=-0.05,
            interval_high=0.4,
            color_role="positive",
            instance_value=3.14,
            solid_on_interval_crosses_zero=False,
        ),
        BarItem(label="baseline", value=-0.1),
    ]
    panel = BarHPanelSpec(
        bars=items,
        xlabel="Contribution",
        ylabel="Feature",
        solid_on_interval_crosses_zero=False,
    )

    # Semantic assertions: configuration preserved
    assert panel.bars[0].color_role == "positive"
    assert panel.solid_on_interval_crosses_zero is False

    # Semantic assertion: bar invariants
    bar0 = panel.bars[0]
    if bar0.interval_low is not None and bar0.interval_high is not None:
        assert bar0.interval_low <= bar0.value <= bar0.interval_high, (
            f"Bar interval violated: {bar0.interval_low} ≤ " f"{bar0.value} ≤ {bar0.interval_high}"
        )


def test_plotspec_all_fields():
    """Verify PlotSpec with all fields set."""
    header = IntervalHeaderSpec(pred=0.5, low=0.25, high=0.75)
    panel = BarHPanelSpec(bars=[BarItem(label="feat", value=0.2)])
    spec = PlotSpec(
        title="Example",
        figure_size=(6.0, 4.0),
        header=header,
        body=panel,
    )

    # Semantic assertions: all fields present and valid
    assert spec.figure_size == (6.0, 4.0)
    assert spec.body.bars[0].label == "feat"

    # Domain invariant: interval bounds
    assert spec.header.low <= spec.header.pred <= spec.header.high


def test_plotspec_from_dict_casts_values():
    """Verify that from_dict properly casts string values to correct types.

    This tests deserialization robustness when values come as strings
    or tuples instead of their expected types.
    """
    raw = {
        "plotspec_version": "1.0.0",
        "title": "dict",
        "figure_size": [5, 2],
        "header": {
            "pred": "0.5",
            "low": 0,
            "high": 1,
            "xlim": ["0", "2"],
            "xlabel": "pred",
        },
        "body": {
            "xlabel": "Contribution",
            "ylabel": "Feature",
            "bars": [
                {
                    "label": "a",
                    "value": "0.25",
                    "interval_low": 0,
                    "interval_high": "0.5",
                    "color_role": "positive",
                },
                {
                    "label": "b",
                    "value": 0.1,
                    "interval_low": None,
                    "interval_high": None,
                },
            ],
        },
    }

    spec = plotspec_from_dict(raw)

    # Semantic assertions: type coercion successful
    assert spec.figure_size == (5, 2)
    assert spec.header is not None and spec.header.pred == 0.5
    assert spec.body is not None and spec.body.bars[0].interval_high == 0.5
    assert spec.body.bars[1].interval_low is None

    # Semantic assertion: interval invariants hold after coercion
    assert spec.header.low <= spec.header.pred <= spec.header.high, (
        f"Bounds violated after coercion: {spec.header.low} ≤ "
        f"{spec.header.pred} ≤ {spec.header.high}"
    )

    # Semantic assertion: bar intervals valid
    bar0 = spec.body.bars[0]
    if bar0.interval_low is not None and bar0.interval_high is not None:
        assert bar0.interval_low <= bar0.value <= bar0.interval_high, (
            f"Bar interval violated: {bar0.interval_low} ≤ " f"{bar0.value} ≤ {bar0.interval_high}"
        )


def test_plot_kind_registry__should_reject_unknown_kind_when_validating_mode():
    """Verify unsupported PlotSpec kinds are rejected with guidance."""
    from calibrated_explanations.utils.exceptions import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        PlotKindRegistry.validate_kind_and_mode("unknown_kind", "classification")

    err = excinfo.value
    assert "Unsupported PlotSpec kind" in str(err)
    assert err.details is not None
    assert "supported_kinds" in err.details


def test_validate_plotspec__should_require_header_for_factual_probabilistic():
    """Verify factual probabilistic plots require headers for interval semantics."""
    from calibrated_explanations.utils.exceptions import ValidationError

    payload = {
        "kind": "factual_probabilistic",
        "mode": "classification",
        "feature_entries": [{"name": "f0", "weight": 0.1}],
        "body": {"xlabel": "Contribution", "ylabel": "Feature"},
    }

    with pytest.raises(ValidationError) as excinfo:
        validate_plotspec(payload)

    assert "requires 'header'" in str(excinfo.value)


def test_triangular_plotspec_roundtrip__should_preserve_metadata_and_defaults():
    """Verify triangular plotspec keeps metadata and default fields intact."""
    spec = TriangularPlotSpec(
        title="triangle",
        figure_size=(7.0, 4.0),
        triangular=TriangularSpec(
            proba=[0.2, 0.7],
            uncertainty=[0.1, 0.3],
            rule_proba=[0.6],
            rule_uncertainty=[0.4],
            num_to_show=10,
            is_probabilistic=False,
        ),
        kind="triangular",
        mode="classification",
        save_behavior=SaveBehavior(path="out", title="tri", default_exts=("png",)),
        data_slice_id="slice-1",
        rendering_seed=42,
    )

    payload = triangular_plotspec_to_dict(spec)
    restored = triangular_plotspec_from_dict(payload)

    assert restored.kind == "triangular"
    assert restored.mode == "classification"
    assert restored.figure_size == (7.0, 4.0)
    assert restored.triangular is not None
    assert restored.triangular.num_to_show == 10
    assert restored.triangular.is_probabilistic is False
    assert restored.save_behavior is not None
    assert restored.save_behavior.default_exts == ("png",)
    assert restored.data_slice_id == "slice-1"
    assert restored.rendering_seed == 42


def test_global_plotspec_to_dict__should_fall_back_to_defaults_when_values_invalid():
    """Verify axis hints default when invalid numeric sequences are supplied."""
    spec = GlobalPlotSpec(
        title="global",
        global_entries=GlobalSpec(
            proba=[object()],
            predict=None,
            uncertainty=[0.05, 0.15],
        ),
        kind="global_probabilistic",
        mode="classification",
    )

    with pytest.warns(UserWarning, match="Failed to cast sequence to floats"):
        payload = global_plotspec_to_dict(spec)

    inner = payload["plot_spec"]
    axis_hints = inner.get("axis_hints") or {}
    assert axis_hints.get("xlim") == [0.0, 1.0]
    assert axis_hints.get("ylim") == [0.05, 0.15]

    restored = global_plotspec_from_dict(payload)
    assert restored.kind == "global_probabilistic"
    assert restored.mode == "classification"


def test_plot_kind_registry_mode_mismatch_and_unknown_requirements():
    """Verify kind-mode incompatibility and unknown kind requirements are rejected."""
    from calibrated_explanations.utils.exceptions import ValidationError

    with pytest.raises(ValidationError, match="Mode 'regression' not supported"):
        PlotKindRegistry.validate_kind_and_mode("factual_probabilistic", "regression")

    with pytest.raises(ValidationError, match="Unsupported kind"):
        PlotKindRegistry.get_kind_requirements("not_a_kind")


def test_from_dict_version_guards_for_all_plotspec_types():
    """Verify envelope version mismatches are rejected for all serializers."""
    from calibrated_explanations.utils.exceptions import ValidationError

    bad_version = "9.9.9"

    with pytest.raises(ValidationError, match="unsupported or missing plotspec_version"):
        plotspec_from_dict({"plotspec_version": bad_version, "plot_spec": {}})

    with pytest.raises(ValidationError, match="unsupported or missing plotspec_version"):
        triangular_plotspec_from_dict({"plotspec_version": bad_version, "plot_spec": {}})

    with pytest.raises(ValidationError, match="unsupported or missing plotspec_version"):
        global_plotspec_from_dict({"plotspec_version": bad_version, "plot_spec": {}})


def test_plotspec_to_dict_with_none_body_sets_defaults_and_provenance():
    """Verify to_dict behavior for body-less specs and roundtrip metadata."""
    spec = PlotSpec(
        title="meta-only",
        figure_size=(3.0, 2.0),
        header=IntervalHeaderSpec(pred=0.2, low=0.1, high=0.3),
        body=None,
        kind="factual_probabilistic",
        mode="classification",
        save_behavior=SaveBehavior(path="out", title="meta", default_exts=("svg",)),
        data_slice_id="slice-meta",
        rendering_seed=7,
    )

    payload = plotspec_to_dict(spec)
    inner = payload["plot_spec"]

    assert payload["plotspec_version"] == PLOTSPEC_VERSION
    assert inner["body"] is None
    assert inner["feature_entries"] is None
    assert inner["feature_order"] == []
    assert inner["uncertainty"] is False
    assert inner["save_behavior"]["default_exts"] == ["svg"]
    assert inner["provenance"] == {"data_slice_id": "slice-meta", "rendering_seed": 7}

    restored = plotspec_from_dict(payload)
    assert restored.save_behavior is not None
    assert restored.save_behavior.default_exts == ("svg",)
    assert restored.data_slice_id == "slice-meta"
    assert restored.rendering_seed == 7


def test_plotspec_from_dict_feature_entries_support_segments_and_lines():
    """Verify feature entry optional fields deserialize through to BarItem."""
    payload = {
        "plotspec_version": PLOTSPEC_VERSION,
        "plot_spec": {
            "kind": "alternative_probabilistic",
            "mode": "classification",
            "body": {"xlabel": "Contribution", "ylabel": "Feature"},
            "feature_entries": [
                {
                    "name": "f0",
                    "weight": "0.4",
                    "low": "0.1",
                    "high": "0.8",
                    "instance_value": "A",
                    "segments": [{"x0": 0.1, "x1": 0.2, "color": "#111"}],
                    "line": "0.5",
                    "line_color": "#222",
                    "line_alpha": "0.6",
                }
            ],
            "save_behavior": {"path": "o", "title": "t", "default_exts": ["png", "svg"]},
            "provenance": {"data_slice_id": "slice-lines", "rendering_seed": 11},
        },
    }

    spec = plotspec_from_dict(payload)
    assert spec.body is not None
    assert len(spec.body.bars) == 1
    bar = spec.body.bars[0]
    assert bar.label == "f0"
    assert bar.value == 0.4
    assert bar.interval_low == 0.1
    assert bar.interval_high == 0.8
    assert bar.segments == [{"x0": 0.1, "x1": 0.2, "color": "#111"}]
    assert bar.line == 0.5
    assert bar.line_color == "#222"
    assert bar.line_alpha == 0.6
    assert spec.save_behavior is not None
    assert spec.save_behavior.default_exts == ("png", "svg")
    assert spec.data_slice_id == "slice-lines"
    assert spec.rendering_seed == 11


def test_validate_kind_specific_required_sections_and_feature_entry_fields():
    """Verify required sections and per-entry required fields by kind."""
    from calibrated_explanations.utils.exceptions import ValidationError

    with pytest.raises(ValidationError, match="requires 'feature_entries' list"):
        validate_plotspec({"kind": "factual_regression", "mode": "regression", "header": {"pred": 1}})

    with pytest.raises(ValidationError, match="missing required fields"):
        validate_plotspec(
            {
                "kind": "factual_regression",
                "mode": "regression",
                "header": {"pred": 1},
                "feature_entries": [{"name": "f0"}],
            }
        )

    with pytest.raises(ValidationError, match="requires 'triangular' section"):
        validate_plotspec({"kind": "triangular", "mode": "classification"})

    with pytest.raises(ValidationError, match="requires 'global_entries' section"):
        validate_plotspec({"kind": "global_probabilistic", "mode": "classification"})
