"""Capability contract tests for the plugin system.

Requirements verified:
  CE-REQ-PLUGIN-DOC-001 — Plugin protocol importability contract (CE-CAP-PLUGIN-001)

These tests verify that the plugin protocols are importable from the public
calibrated_explanations.plugins submodules.
See development/capabilities/requirements/CE-REQ-PLUGIN-DOC-001.md for the full
assumption boundary.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# CE-REQ-PLUGIN-DOC-001 — Plugin protocol importability contract
# ---------------------------------------------------------------------------


def test_should_import_explainer_plugin_protocol_when_plugins_module_available():
    """Verify CE-REQ-PLUGIN-DOC-001: ExplainerPlugin is importable from plugins.base.

    Acceptance criteria (from CE-REQ-PLUGIN-DOC-001):
    - from calibrated_explanations.plugins.base import ExplainerPlugin succeeds.
    - ExplainerPlugin is not None.
    """
    from calibrated_explanations.plugins.base import ExplainerPlugin  # noqa: PLC0415

    assert (
        ExplainerPlugin is not None
    ), "CE-REQ-PLUGIN-DOC-001: ExplainerPlugin must be importable and non-None"


def test_should_import_interval_calibrator_plugin_protocol_when_plugins_module_available():
    """Verify CE-REQ-PLUGIN-DOC-001: IntervalCalibratorPlugin is importable from plugins.intervals.

    Acceptance criteria (from CE-REQ-PLUGIN-DOC-001):
    - from calibrated_explanations.plugins.intervals import IntervalCalibratorPlugin succeeds.
    - IntervalCalibratorPlugin is not None.
    """
    from calibrated_explanations.plugins.intervals import IntervalCalibratorPlugin  # noqa: PLC0415

    assert (
        IntervalCalibratorPlugin is not None
    ), "CE-REQ-PLUGIN-DOC-001: IntervalCalibratorPlugin must be importable and non-None"
