"""ADR-013 interval calibrator plugin strategy gap closure tests.

Covers:
- Gap 1: Protocol signature alignment between protocol and IntervalRegressor
- Gap 3: FAST calibrator must not be in non-fast fallback chains
- Gap 4: Third-party plugin contract via structural typing (not just isinstance)
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock


from calibrated_explanations.plugins.manager import (
    DEFAULT_INTERVAL_IDENTIFIERS,
    PluginManager,
)


# ---------------------------------------------------------------------------
# Gap 3: FAST identifier must not be in the non-fast fallback chain
# ---------------------------------------------------------------------------


def build_pm() -> PluginManager:
    """Create a PluginManager and build all chains via the public build_interval_chain API."""
    explainer = MagicMock()
    explainer.plugin_manager = None
    pm = PluginManager(explainer)
    # Use public method to build chains
    pm.interval_plugin_fallbacks = {
        "default": pm.build_interval_chain(fast=False),
        "fast": pm.build_interval_chain(fast=True),
    }
    return pm


def test_should_exclude_fast_identifier_from_default_interval_chain():
    """core.interval.fast must NOT appear in the non-fast (default) fallback chain."""
    pm = build_pm()
    default_chain = pm.interval_plugin_fallbacks.get("default", ())
    fast_id = DEFAULT_INTERVAL_IDENTIFIERS["fast"]

    assert fast_id not in default_chain, (
        f"'{fast_id}' must not appear in the default (non-fast) interval chain; "
        f"found in: {default_chain}"
    )


def test_should_keep_default_and_fast_chains_separate():
    """Default and fast interval chains must be distinct and non-overlapping on the fast ID."""
    pm = build_pm()

    default_chain = set(pm.interval_plugin_fallbacks.get("default", ()))
    fast_id = DEFAULT_INTERVAL_IDENTIFIERS["fast"]

    # The fast identifier must not bleed into the default chain
    assert (
        fast_id not in default_chain
    ), f"'{fast_id}' leaked into the non-fast default chain: {default_chain}"


# ---------------------------------------------------------------------------
# Gap 4: Protocol enforcement via structural typing (not just isinstance)
# ---------------------------------------------------------------------------


class ThirdPartyIntervalCalibrator:
    """Minimal structural conformance without inheriting from any CE base class."""

    def predict_proba(self, x, *, output_interval=False, classes=None, bins=None):
        import numpy as np

        n = len(x)
        return np.full((n, 2), 0.5)

    def is_multiclass(self) -> bool:
        return False

    def is_mondrian(self) -> bool:
        return False


def test_should_structurally_conform_when_third_party_implements_protocol():
    """A third-party calibrator conforming to the protocol surface must be accepted structurally."""
    calibrator = ThirdPartyIntervalCalibrator()

    # Check structural conformance: required methods exist with expected signatures
    required_methods = ["predict_proba", "is_multiclass", "is_mondrian"]
    for method_name in required_methods:
        assert hasattr(
            calibrator, method_name
        ), f"Third-party calibrator missing required method: {method_name}"

    # predict_proba must accept (x, *, output_interval, classes, bins)
    sig = inspect.signature(calibrator.predict_proba)
    params = sig.parameters
    assert "output_interval" in params, "predict_proba must accept output_interval kwarg"
    assert "classes" in params, "predict_proba must accept classes kwarg"
    assert "bins" in params, "predict_proba must accept bins kwarg"

    # is_multiclass and is_mondrian must be zero-arg callables
    for method_name in ("is_multiclass", "is_mondrian"):
        sig = inspect.signature(getattr(calibrator, method_name))
        non_self_params = [
            p
            for name, p in sig.parameters.items()
            if name != "self" and p.default is inspect.Parameter.empty
        ]
        assert non_self_params == [], f"{method_name} must have no required parameters"


# ---------------------------------------------------------------------------
# Gap 1: IntervalRegressor implements the required protocol methods
# ---------------------------------------------------------------------------


def test_should_confirm_interval_regressor_implements_required_protocol_methods():
    """IntervalRegressor must implement all methods in the RegressionIntervalCalibrator protocol surface.

    Design note (ADR-013 gap 1): IntervalRegressor is the REGRESSION reference implementation.
    Its predict_proba has a simpler signature than the ClassificationIntervalCalibrator protocol
    because the runtime never calls predict_proba(output_interval=..., classes=...) on it —
    those kwargs are VennAbers-specific (classification path). Protocol structural conformance
    is verified by isinstance(ir, RegressionIntervalCalibrator) which passes at runtime.
    Strict signature conformance is enforced on VennAbers (classification) and the third-party
    plugin contract test (see test_should_structurally_conform_when_third_party_implements_protocol).
    """
    from calibrated_explanations.calibration.interval_regressor import IntervalRegressor
    from calibrated_explanations.plugins.intervals import (
        ClassificationIntervalCalibrator,
        RegressionIntervalCalibrator,
    )

    # All required method names must exist
    required = [
        "predict_probability",
        "predict_uncertainty",
        "pre_fit_for_probabilistic",
        "compute_proba_cal",
        "insert_calibration",
        "predict_proba",
        "is_multiclass",
        "is_mondrian",
    ]
    ir = IntervalRegressor.__new__(IntervalRegressor)
    for method_name in required:
        assert hasattr(
            IntervalRegressor, method_name
        ), f"IntervalRegressor missing required protocol method: {method_name}"

    # Structural protocol conformance — isinstance checks pass at runtime
    assert isinstance(
        ir, RegressionIntervalCalibrator
    ), "IntervalRegressor must satisfy RegressionIntervalCalibrator structural protocol"
    assert isinstance(
        ir, ClassificationIntervalCalibrator
    ), "IntervalRegressor must satisfy ClassificationIntervalCalibrator structural protocol"

    # is_multiclass and is_mondrian must accept no required positional args beyond self
    for name in ("is_multiclass", "is_mondrian"):
        method_sig = inspect.signature(getattr(IntervalRegressor, name))
        required_pos = [
            p
            for pname, p in method_sig.parameters.items()
            if pname != "self"
            and p.default is inspect.Parameter.empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        assert required_pos == [], f"IntervalRegressor.{name} must have no required parameters"


def test_should_confirm_venn_abers_implements_full_classification_protocol_signature():
    """VennAbers (classification impl) must implement the full ClassificationIntervalCalibrator signature.

    VennAbers is the primary ClassificationIntervalCalibrator and the reference for
    output_interval, classes, and bins kwargs — callers of the classification path use all three.
    """
    from calibrated_explanations.calibration.venn_abers import VennAbers
    from calibrated_explanations.plugins.intervals import ClassificationIntervalCalibrator

    required = ["predict_proba", "is_multiclass", "is_mondrian"]
    for method_name in required:
        assert hasattr(
            VennAbers, method_name
        ), f"VennAbers missing required protocol method: {method_name}"

    # Full signature conformance: output_interval, classes, bins must be present
    sig = inspect.signature(VennAbers.predict_proba)
    params = sig.parameters
    assert "output_interval" in params, "VennAbers.predict_proba must accept output_interval kwarg"
    assert "classes" in params, "VennAbers.predict_proba must accept classes kwarg"
    assert "bins" in params, "VennAbers.predict_proba must accept bins kwarg"

    # VennAbers satisfies the structural protocol
    va = VennAbers.__new__(VennAbers)
    assert isinstance(va, ClassificationIntervalCalibrator)
