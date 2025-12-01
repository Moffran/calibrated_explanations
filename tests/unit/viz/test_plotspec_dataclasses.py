import pytest

from calibrated_explanations.viz import (
    BarHPanelSpec,
    BarItem,
    IntervalHeaderSpec,
    PlotSpec,
    __all__ as plotspec_all,
)


def test_interval_header_spec__should_preserve_bounds_and_labels():
    """Verify that IntervalHeaderSpec preserves all configured bounds and labels.

    Domain Invariants:
    - Bounds (pred, low, high) must satisfy: low ≤ pred ≤ high
    - Color/alpha values must be valid (hex string, 0 ≤ alpha ≤ 1)
    - Labels (xlabel, ylabel, neg_label, pos_label) must be strings
    Ref: ADR-005 Explanation Envelope, ADR-007 PlotSpec Abstraction
    """
    header = IntervalHeaderSpec(
        pred=0.75,
        low=0.25,
        high=0.9,
        xlim=(-1.0, 1.0),
        xlabel="Predicted",
        ylabel="Density",
        dual=False,
        neg_label="class 0",
        pos_label="class 1",
        uncertainty_color="#cccccc",
        uncertainty_alpha=0.42,
    )

    # Domain invariant: bounds ordering
    assert (
        header.low <= header.pred <= header.high
    ), f"Bounds must satisfy low≤pred≤high: {header.low}≤{header.pred}≤{header.high}"

    # Domain invariant: specific bound values
    assert header.pred == 0.75
    assert header.low == 0.25
    assert header.high == 0.9

    # Domain invariant: xlim range is sensible (tuple of two floats)
    assert isinstance(header.xlim, tuple) and len(header.xlim) == 2
    assert header.xlim == (-1.0, 1.0)

    # Domain invariants: labels are non-empty strings
    assert isinstance(header.xlabel, str) and header.xlabel == "Predicted"
    assert isinstance(header.ylabel, str) and header.ylabel == "Density"
    assert isinstance(header.neg_label, str) and header.neg_label == "class 0"
    assert isinstance(header.pos_label, str) and header.pos_label == "class 1"

    # Domain invariant: visual attributes are valid types
    assert header.dual is False
    assert isinstance(header.uncertainty_color, str) and header.uncertainty_color == "#cccccc"
    assert (
        0.0 <= header.uncertainty_alpha <= 1.0
    ), f"Alpha must be in [0, 1], got {header.uncertainty_alpha}"


def test_bar_item_and_panel__should_preserve_structure_and_values():
    """Verify that BarItem and BarHPanelSpec preserve field values and domain constraints.

    Domain Invariants:
    - Bar label is non-empty string
    - Bar value and interval bounds must be numeric
    - Interval bounds must satisfy: interval_low ≤ value ≤ interval_high
    - Panel contains at least one bar
    - Panel labels are strings
    Ref: ADR-005 Explanation Envelope, ADR-007 PlotSpec Abstraction
    """
    bar = BarItem(
        label="Feature A",
        value=-0.6,
        interval_low=-0.8,
        interval_high=-0.2,
        color_role="negative",
        instance_value=3.14,
        solid_on_interval_crosses_zero=False,
    )

    # Domain invariants: bar field values
    assert bar.label == "Feature A"
    assert isinstance(bar.label, str) and bar.label
    assert isinstance(bar.value, (int, float))
    assert bar.value == -0.6

    # Domain invariant: interval bounds ordering
    assert isinstance(bar.interval_low, (int, float))
    assert isinstance(bar.interval_high, (int, float))
    assert (
        bar.interval_low <= bar.value <= bar.interval_high
    ), f"Bar value ({bar.value}) must lie in [{bar.interval_low}, {bar.interval_high}]"
    assert bar.interval_low == -0.8
    assert bar.interval_high == -0.2

    # Domain invariants: optional fields
    assert bar.color_role == "negative"
    assert isinstance(bar.instance_value, (int, float))
    assert bar.instance_value == 3.14
    assert bar.solid_on_interval_crosses_zero is False

    panel = BarHPanelSpec(bars=[bar], xlabel="Contribution", ylabel="Feature")

    # Domain invariants: panel structure
    assert len(panel.bars) >= 1, "Panel must contain at least one bar"
    assert bar in panel.bars, "Panel must contain the original bar"
    assert isinstance(panel.xlabel, str) and panel.xlabel == "Contribution"
    assert isinstance(panel.ylabel, str) and panel.ylabel == "Feature"

    # Domain invariant: default solid_on_interval_crosses_zero
    assert (
        panel.solid_on_interval_crosses_zero is True
    ), "Panel default for solid_on_interval_crosses_zero should be True"


def test_plotspec_collects_header_and_body__should_preserve_references():
    """Verify that PlotSpec correctly assembles and preserves header and body references.

    Domain Invariants:
    - Title must be preserved exactly
    - Figure size must be a tuple of two positive floats
    - Header and body must be preserved as-is (reference equality)
    - Equality comparison must work for dataclass instances
    Ref: ADR-005 Explanation Envelope, ADR-007 PlotSpec Abstraction
    """
    header = IntervalHeaderSpec(pred=0.1, low=-0.2, high=0.4)
    bars = [BarItem(label="f0", value=0.3)]
    body = BarHPanelSpec(bars=bars)

    spec = PlotSpec(title="Example", figure_size=(4.0, 3.0), header=header, body=body)

    # Domain invariants: structural components
    assert isinstance(spec.title, str) and spec.title == "Example"
    assert isinstance(spec.figure_size, tuple) and len(spec.figure_size) == 2
    assert all(isinstance(s, (int, float)) and s > 0 for s in spec.figure_size)
    assert spec.figure_size == (4.0, 3.0)

    # Domain invariants: component references preserved
    assert spec.header is header, "Header reference must be preserved"
    assert spec.body is body, "Body reference must be preserved"

    # Domain invariant: dataclass equality (semantic, not identity)
    reconstructed = PlotSpec(title="Example", figure_size=(4.0, 3.0), header=header, body=body)
    assert spec == reconstructed, "PlotSpec should support equality by value"


def test_bar_structures_require_mandatory_fields__should_raise_type_error():
    """Verify that bar and panel dataclasses enforce mandatory field requirements.

    Domain Invariant: A valid BarItem requires both label and value (not optional).
    A valid BarHPanelSpec requires bars (not optional).
    """
    with pytest.raises(TypeError):
        BarItem()  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        BarItem(label="missing-value")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        BarHPanelSpec()  # type: ignore[call-arg]


def test_plotspec_module_exports_are_consistent__should_export_required_types():
    """Verify that plotspec module exports all required public types.

    Domain Invariant: The module's __all__ must include exactly the four main classes,
    and each must be importable and have the correct __name__.
    """
    expected = {"PlotSpec", "IntervalHeaderSpec", "BarHPanelSpec", "BarItem"}
    assert set(plotspec_all) == expected

    module_attrs = {name: globals()[name] for name in plotspec_all}
    for name, value in module_attrs.items():
        assert value.__name__ == name


def test_bar_item_requires_label_and_value__should_raise_on_missing_fields():
    """Verify that BarItem constructor enforces mandatory label and value fields.

    Domain Invariant: label and value are not optional (no defaults provided).
    """
    with pytest.raises(TypeError):
        BarItem()  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        BarItem(label="x")  # type: ignore[call-arg]


def test_dataclasses_reject_missing_required_fields__should_raise_on_construction():
    """Verify that all plotspec dataclasses enforce their mandatory fields at construction.

    Domain Invariant: Required fields (pred, low, high for IntervalHeaderSpec; label, value
    for BarItem; bars for BarHPanelSpec) cannot be omitted or partially supplied.
    """
    with pytest.raises(TypeError):
        IntervalHeaderSpec(pred=0.5, low=0.1)  # missing high

    with pytest.raises(TypeError):
        BarItem(value=0.2)  # missing label

    with pytest.raises(TypeError):
        BarHPanelSpec()
