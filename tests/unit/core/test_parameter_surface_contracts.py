"""Parameter-surface contract tests (ADR-038, v0.11.6 Task 5D).

Guards the single-source-of-truth relationship between the
``CalibratedExplainer`` kwarg allow-lists and the ``WrapCalibratedExplainer``
per-method gates:

1. Structural contracts: every name accepted on a ``CalibratedExplainer``
   surface must also be accepted by the wrapper's corresponding method (minus
   the internal ``_ce_skip_reject`` escape hatch), and the explicit-formal
   reference sets must stay in sync with the actual signatures.
2. Acceptance matrices: every allow-listed public parameter is exercised
   end-to-end with a benign value, so a name that stops being consumed (or
   starts being rejected) fails a test instead of drifting silently.
3. Regressions for the Task 5D bug round: ``reject_confidence`` on the wrapper
   explain methods, ``interval_summary`` on ``wrap.predict``, the
   ``CalibratedExplainer.__init__`` kwargs blocked at ``calibrate()``, the
   legacy global plot path, calibration-state preservation on rejected
   ``calibrate()`` calls, and per-method error surfaces.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations.calibration.normalization_strategy import NormalizationStrategy
from calibrated_explanations.core.calibrated_explainer import (
    CalibratedExplainer,
    _EXPLAIN_KWARGS as CE_EXPLAIN_KWARGS,
    _INIT_EXPLICIT_PARAMS as CE_INIT_EXPLICIT_PARAMS,
    _INIT_KWARGS as CE_INIT_KWARGS,
    _PREDICT_KWARGS as CE_PREDICT_KWARGS,
    _PREDICT_PROBA_KWARGS as CE_PREDICT_PROBA_KWARGS,
)
from calibrated_explanations.core.prediction.interval_summary import IntervalSummary
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.core.wrap_explainer import (
    WrapCalibratedExplainer,
    _CALIBRATE_KWARGS as WRAP_CALIBRATE_KWARGS,
    _EXPLAIN_FAST_KWARGS as WRAP_EXPLAIN_FAST_KWARGS,
    _EXPLAIN_KWARGS as WRAP_EXPLAIN_KWARGS,
    _PREDICT_KWARGS as WRAP_PREDICT_KWARGS,
    _PREDICT_PROBA_KWARGS as WRAP_PREDICT_PROBA_KWARGS,
)
from calibrated_explanations.explanations.explanations import MultiClassCalibratedExplanations
from calibrated_explanations.explanations.guarded_options import GuardedOptions
from calibrated_explanations.utils.exceptions import ConfigurationError, ValidationError


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 4))
    y_cls = (x[:, 0] + x[:, 1] > 0).astype(int)
    y_reg = x[:, 0] * 2 + rng.normal(scale=0.1, size=120)
    return x, y_cls, y_reg


def _fitted_classification_wrapper(data):
    x, y_cls, _ = data
    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=0))
    wrapper.fit(x[:60], y_cls[:60])
    return wrapper


@pytest.fixture(scope="module")
def cls_wrapper(data):
    x, y_cls, _ = data
    wrapper = _fitted_classification_wrapper(data)
    wrapper.calibrate(x[60:100], y_cls[60:100])
    return wrapper, x[100:]


@pytest.fixture(scope="module")
def reg_wrapper(data):
    x, _, y_reg = data
    wrapper = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=10, random_state=0))
    wrapper.fit(x[:60], y_reg[:60])
    wrapper.calibrate(x[60:100], y_reg[60:100])
    return wrapper, x[100:]


@pytest.fixture(scope="module")
def multiclass_wrapper():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(150, 5))
    y = np.digitize(x[:, 0] + 0.5 * x[:, 1], bins=[-0.4, 0.6]).astype(int)
    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=20, random_state=7))
    wrapper.fit(x[:75], y[:75])
    wrapper.calibrate(x[75:120], y[75:120])
    return wrapper, x[120:]


@pytest.fixture(scope="module")
def fast_cls_wrapper(data):
    x, y_cls, _ = data
    wrapper = _fitted_classification_wrapper(data)
    wrapper.calibrate(x[60:100], y_cls[60:100], fast=True)
    return wrapper, x[100:]


# ---------------------------------------------------------------------------
# 1. Structural contracts: wrapper gates derive from CalibratedExplainer's
# ---------------------------------------------------------------------------


def test_wrap_calibrate_accepts_everything_explainer_init_accepts():
    """Anything CalibratedExplainer.__init__ accepts (via **kwargs or explicit
    formals) must pass the wrapper's calibrate() gate."""
    assert CE_INIT_KWARGS <= WRAP_CALIBRATE_KWARGS
    assert CE_INIT_EXPLICIT_PARAMS <= WRAP_CALIBRATE_KWARGS


@pytest.mark.parametrize(
    ("wrap_set", "explainer_set"),
    [
        (WRAP_PREDICT_KWARGS, CE_PREDICT_KWARGS),
        (WRAP_PREDICT_PROBA_KWARGS, CE_PREDICT_PROBA_KWARGS),
        (WRAP_EXPLAIN_KWARGS, CE_EXPLAIN_KWARGS),
    ],
    ids=["predict", "predict_proba", "explain"],
)
def test_wrap_gate_accepts_everything_explainer_accepts(wrap_set, explainer_set):
    """The wrapper may only subtract the internal _ce_skip_reject escape hatch
    from the CalibratedExplainer surface it forwards to."""
    assert explainer_set - {"_ce_skip_reject"} <= wrap_set


def test_init_explicit_params_match_signature():
    """_INIT_EXPLICIT_PARAMS is a reference set for signature formals; it must
    track CalibratedExplainer.__init__'s actual explicit parameters."""
    sig = inspect.signature(CalibratedExplainer.__init__)
    formals = {
        name
        for name, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    } - {"self", "learner", "x_cal", "y_cal"}
    assert formals == CE_INIT_EXPLICIT_PARAMS


def test_wrap_explain_fast_kwargs_match_explainer_signature():
    """CalibratedExplainer.explain_fast has no **kwargs; the wrapper's gate
    must mirror its explicit signature exactly."""
    sig = inspect.signature(CalibratedExplainer.explain_fast)
    formals = {
        name
        for name, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    } - {"self", "x", "_use_plugin"}
    assert formals == WRAP_EXPLAIN_FAST_KWARGS


@pytest.mark.parametrize(
    "method",
    [CalibratedExplainer.explain_factual, CalibratedExplainer.explore_alternatives],
    ids=["explain_factual", "explore_alternatives"],
)
def test_explain_kwargs_cover_explain_signatures(method):
    """Every explicit formal of the explain methods must be allow-listed, so a
    signature promotion cannot silently open a gap in the gates."""
    sig = inspect.signature(method)
    formals = {
        name
        for name, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    } - {"self", "x", "_use_plugin"}
    assert formals <= CE_EXPLAIN_KWARGS


# ---------------------------------------------------------------------------
# 2. Acceptance matrices: every allow-listed name works end-to-end
# ---------------------------------------------------------------------------

# Benign values per calibrate() parameter. The completeness assertions below
# force this map to be extended whenever a new name is allow-listed.
_CALIBRATE_VALUES = {
    "alternative_plugin": None,
    "bins": None,
    "categorical_features": [0],
    "categorical_labels": None,
    "class_labels": None,
    "condition_source": "prediction",
    "default_reject_policy": RejectPolicy.NONE,
    "difficulty_estimator": None,
    "factual_plugin": None,
    "fast": False,
    "fast_interval_plugin": None,
    "fast_plugin": None,
    "feature_names": None,
    "features_to_ignore": [],
    "interval_plugin": None,
    "interval_summary": "mean",
    "mode": "classification",
    "noise_type": "uniform",
    "oob": False,
    "perf_cache": None,
    "perf_parallel": None,
    "plot_style": None,
    "predict_function": None,
    "preprocessor_metadata": None,
    "reject": False,
    "reuse_conditional": False,
    "sample_percentiles": [25, 50, 75],
    "scale_factor": 5,
    "seed": 42,
    "severity": 1,
    "suppress_crepes_errors": False,
    "verbose": False,
}

_EXPLAIN_VALUES = {
    "bins": None,
    "features_to_ignore": [],
    "guarded_options": None,
    "interval_summary": None,
    "low_high_percentiles": (5, 95),
    "multi_labels_enabled": False,
    "reject_confidence": 0.9,
    "reject_policy": None,
    "threshold": None,
    "verbose": False,
}

_EXPLAIN_FAST_VALUES = {
    "bins": None,
    "low_high_percentiles": (5, 95),
    "reject_policy": None,
    "threshold": None,
}

_PREDICT_VALUES = {
    "bins": None,
    "classes": None,
    "feature": None,
    "interval_summary": "mean",
    "low_high_percentiles": (5, 95),
    "reject_confidence": 0.9,
    "reject_policy": None,
    "threshold": None,
}

_PREDICT_PROBA_VALUES = {
    "bins": None,
    "interval_summary": "mean",
    "normalization": NormalizationStrategy.SCALE,
    "reject_confidence": 0.9,
    "reject_policy": None,
}


def test_acceptance_matrices_cover_every_allow_listed_name():
    """Adding a name to any wrapper gate requires adding a benign value here,
    so new parameters cannot land without an end-to-end acceptance test."""
    assert set(_CALIBRATE_VALUES) == set(WRAP_CALIBRATE_KWARGS)
    assert set(_EXPLAIN_VALUES) == set(WRAP_EXPLAIN_KWARGS)
    assert set(_EXPLAIN_FAST_VALUES) == set(WRAP_EXPLAIN_FAST_KWARGS)
    assert set(_PREDICT_VALUES) == set(WRAP_PREDICT_KWARGS)
    assert set(_PREDICT_PROBA_VALUES) == set(WRAP_PREDICT_PROBA_KWARGS)


@pytest.mark.parametrize(("name", "value"), sorted(_CALIBRATE_VALUES.items()))
def test_calibrate_accepts_every_allow_listed_kwarg(data, name, value):
    x, y_cls, _ = data
    wrapper = _fitted_classification_wrapper(data)
    wrapper.calibrate(x[60:100], y_cls[60:100], **{name: value})
    assert wrapper.calibrated


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
@pytest.mark.parametrize(("name", "value"), sorted(_EXPLAIN_VALUES.items()))
def test_explain_methods_accept_every_allow_listed_kwarg(cls_wrapper, method_name, name, value):
    wrapper, x_test = cls_wrapper
    result = getattr(wrapper, method_name)(x_test[:2], **{name: value})
    assert result is not None


@pytest.mark.parametrize(("name", "value"), sorted(_EXPLAIN_FAST_VALUES.items()))
def test_explain_fast_accepts_every_allow_listed_kwarg(fast_cls_wrapper, name, value):
    wrapper, x_test = fast_cls_wrapper
    result = wrapper.explain_fast(x_test[:2], **{name: value})
    assert result is not None


@pytest.mark.parametrize(("name", "value"), sorted(_PREDICT_VALUES.items()))
def test_predict_accepts_every_allow_listed_kwarg(cls_wrapper, name, value):
    wrapper, x_test = cls_wrapper
    result = wrapper.predict(x_test[:2], **{name: value})
    assert result is not None


@pytest.mark.parametrize(("name", "value"), sorted(_PREDICT_PROBA_VALUES.items()))
def test_predict_proba_accepts_every_allow_listed_kwarg(cls_wrapper, name, value):
    wrapper, x_test = cls_wrapper
    result = wrapper.predict_proba(x_test[:2], **{name: value})
    assert result is not None


# ---------------------------------------------------------------------------
# 3. Task 5D bug regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
def test_reject_confidence_accepted_with_reject_policy_on_wrap_explain(cls_wrapper, method_name):
    """Bug 5D-2: reject_confidence is valid on the explain methods whenever
    reject_policy is set (proven at the CalibratedExplainer level in 5C); the
    wrapper gate must not reject it."""
    wrapper, x_test = cls_wrapper
    result = getattr(wrapper, method_name)(x_test[:2], reject_policy="flag", reject_confidence=0.8)
    assert result is not None


def test_interval_summary_accepted_on_wrap_predict(cls_wrapper):
    """Bug 5D-3: CalibratedExplainer.predict consumes interval_summary, so the
    wrapper's predict() gate must accept it (predict_proba already did)."""
    wrapper, x_test = cls_wrapper
    result = wrapper.predict(x_test[:2], interval_summary="mean")
    assert len(result) == 2


def test_calibrate_accepts_fast_mode_tuning_kwargs(data):
    """Bug 5D-4: fast=True is allowed at calibrate(), so its tuning knobs
    (noise_type/scale_factor/severity) must be too."""
    x, y_cls, _ = data
    wrapper = _fitted_classification_wrapper(data)
    wrapper.calibrate(
        x[60:100],
        y_cls[60:100],
        fast=True,
        noise_type="uniform",
        scale_factor=3,
        severity=1,
    )
    assert wrapper.calibrated


def test_rejected_calibrate_preserves_previous_calibration(data):
    """Bug 5D-5: a calibrate() call rejected by the kwargs gates must not
    discard a previously successful calibration."""
    x, y_cls, _ = data
    wrapper = _fitted_classification_wrapper(data)
    wrapper.calibrate(x[60:100], y_cls[60:100])
    assert wrapper.calibrated

    with pytest.raises(ConfigurationError, match="unknown keyword arguments"):
        wrapper.calibrate(x[60:100], y_cls[60:100], not_a_parameter=1)
    assert wrapper.calibrated

    with pytest.raises(ConfigurationError, match="not valid here"):
        wrapper.calibrate(x[60:100], y_cls[60:100], threshold=0.5)
    assert wrapper.calibrated

    # And the surviving explainer still works end-to-end.
    assert wrapper.predict(x[100:102]) is not None


def test_unknown_kwarg_error_names_the_rejecting_method(data):
    """Bug 5D-6: genuinely unknown names must report the per-method surface,
    not just the class name."""
    x, y_cls, _ = data
    wrapper = _fitted_classification_wrapper(data)
    with pytest.raises(
        ConfigurationError, match=r"WrapCalibratedExplainer\.calibrate received unknown"
    ):
        wrapper.calibrate(x[60:100], y_cls[60:100], not_a_parameter=1)


@pytest.mark.viz
class TestLegacyGlobalPlotKwargForwarding:
    """Bug 5D-1: the legacy plot_global forwarded its entire kwargs dict into
    predict/predict_proba, which the fail-fast gates now reject. Exercise the
    legacy path for real (show=True under Agg) -- with show=False and
    matplotlib unloaded the legacy path silently no-ops, which is exactly how
    the break slipped past the 5C verification."""

    @pytest.fixture(autouse=True)
    def _agg_backend(self, monkeypatch):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        from calibrated_explanations.viz import _matplotlib_compat as compat
        from calibrated_explanations.viz._matplotlib_compat import (
            _require_matplotlib as require_matplotlib,
        )

        require_matplotlib()
        monkeypatch.setattr(compat.plt, "show", lambda *args, **kwargs: None)
        yield
        compat.plt.close("all")

    def test_classification_use_legacy(self, cls_wrapper):
        from calibrated_explanations.viz import _matplotlib_compat as compat

        wrapper, x_test = cls_wrapper
        wrapper.plot(x_test[:2], show=True, use_legacy=True)
        assert compat.plt.get_fignums(), "legacy plot must actually render a figure"

    def test_regression_use_legacy(self, reg_wrapper):
        from calibrated_explanations.viz import _matplotlib_compat as compat

        wrapper, x_test = reg_wrapper
        wrapper.plot(x_test[:2], show=True, use_legacy=True)
        assert compat.plt.get_fignums(), "legacy plot must actually render a figure"

    def test_regression_thresholded_use_legacy(self, reg_wrapper):
        from calibrated_explanations.viz import _matplotlib_compat as compat

        wrapper, x_test = reg_wrapper
        wrapper.plot(x_test[:2], threshold=0.0, show=True, use_legacy=True)
        assert compat.plt.get_fignums(), "legacy plot must actually render a figure"

    def test_regression_low_high_percentiles_use_legacy(self, reg_wrapper):
        from calibrated_explanations.viz import _matplotlib_compat as compat

        wrapper, x_test = reg_wrapper
        wrapper.plot(x_test[:2], show=True, use_legacy=True, low_high_percentiles=(10, 90))
        assert compat.plt.get_fignums(), "legacy plot must actually render a figure"

    def test_plot_only_kwargs_do_not_reach_prediction_gates(self, cls_wrapper, tmp_path):
        from calibrated_explanations.viz import _matplotlib_compat as compat

        wrapper, x_test = cls_wrapper
        wrapper.plot(
            x_test[:2],
            show=True,
            use_legacy=True,
            path=str(tmp_path),
            save_ext=[],
        )
        # Proves path/save_ext (plot-only kwargs) did not reach predict()/
        # predict_proba() and cause a ConfigurationError -- the plot still
        # completed and rendered a figure.
        assert compat.plt.get_fignums(), "legacy plot must actually render a figure"


@pytest.mark.viz
class TestUnifiedPlotPredictionValidationContract:
    """Unified validation contract (pre-v4 S4-H6, v0.11.6 task 54).

    predict/predict_proba/plot(use_legacy=False)/plot(use_legacy=True) must
    reject an invalid ``threshold`` identically, and a misspelled plot-only
    kwarg must produce a governed signal (INFO + UserWarning) on every plot
    path instead of silently forwarding."""

    @pytest.fixture(autouse=True)
    def _agg_backend(self, monkeypatch):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        from calibrated_explanations.viz import _matplotlib_compat as compat
        from calibrated_explanations.viz._matplotlib_compat import (
            _require_matplotlib as require_matplotlib,
        )

        require_matplotlib()
        monkeypatch.setattr(compat.plt, "show", lambda *args, **kwargs: None)
        yield
        compat.plt.close("all")

    @pytest.mark.parametrize("show", [True, False])
    @pytest.mark.parametrize("use_legacy", [False, True])
    def test_classification_threshold_rejected_on_all_four_surfaces(
        self, cls_wrapper, use_legacy, show
    ):
        wrapper, x_test = cls_wrapper
        with pytest.raises(ValidationError, match="only supported for mode='regression'"):
            wrapper.predict(x_test[:2], threshold=0.5)
        with pytest.raises(ValidationError, match="only supported for mode='regression'"):
            wrapper.predict_proba(x_test[:2], threshold=0.5)
        # pre-v4 S4-H6: this used to silently no-op for use_legacy=True,
        # show=False -- the exact combination that let the bug slip past
        # earlier verification.
        with pytest.raises(ValidationError, match="only supported for mode='regression'"):
            wrapper.plot(x_test[:2], threshold=0.5, use_legacy=use_legacy, show=show)

    @pytest.mark.parametrize("use_legacy", [False, True])
    def test_misspelled_plot_kwarg_emits_governed_fallback_signal_on_wrapper_plot(
        self, cls_wrapper, use_legacy
    ):
        wrapper, x_test = cls_wrapper
        with pytest.warns(UserWarning, match="filter_topp"):
            wrapper.plot(x_test[:2], filter_topp=1, use_legacy=use_legacy, show=False)

    def test_legacy_and_plotspec_plot_paths_consume_the_same_validated_payload(
        self, monkeypatch, reg_wrapper
    ):
        """The legacy renderer must receive the exact prediction payload the
        wrapper/core validation produced, instead of independently
        re-deriving (and potentially bypassing validation for) its own."""
        from calibrated_explanations.viz import _matplotlib_compat as compat

        wrapper, x_test = reg_wrapper
        captured = {}
        original = compat.plot_global

        def spy(*args, **kwargs):
            captured["payload"] = kwargs.get("_validated_payload")
            return original(*args, **kwargs)

        monkeypatch.setattr(compat, "plot_global", spy)

        wrapper.plot(x_test[:2], threshold=0.0, use_legacy=True, show=True)
        legacy_payload = captured["payload"]
        assert legacy_payload is not None

        direct_proba, (direct_low, direct_high) = wrapper.explainer.predict_proba(
            x_test[:2], uq_interval=True, threshold=0.0, bins=None
        )
        np.testing.assert_allclose(legacy_payload["proba"], direct_proba)
        np.testing.assert_allclose(legacy_payload["low"], direct_low)
        np.testing.assert_allclose(legacy_payload["high"], direct_high)

    def test_misspelled_plot_kwarg_emits_governed_fallback_signal_on_item_plot(self, cls_wrapper):
        wrapper, x_test = cls_wrapper
        explanations = wrapper.explain_factual(x_test[:2])
        with pytest.warns(UserWarning, match="filter_topp"):
            explanations[0].plot(filter_topp=1, show=False)


def _selected_probability_from_predict_proba(proba_result, *, class_index):
    proba_payload = getattr(proba_result, "prediction", proba_result)
    if isinstance(proba_payload, tuple):
        proba_payload = proba_payload[0]
    proba = np.asarray(proba_payload, dtype=float)
    if proba.ndim != 2:
        raise AssertionError(f"Expected a 2D probability matrix, got shape {proba.shape}")
    return float(proba[0, int(class_index)])


def _extract_narrative_probability(text):
    match = re.search(r"Calibrated [Pp]robability(?:[^:\n]*)?:\s*([0-9.]+)", text)
    assert match is not None, text
    return float(match.group(1))


def assert_probability_surface_alignment(explanation, expected_probability):
    prediction = explanation.prediction
    json_prediction = explanation.calibrated_explanations.to_json()["explanations"][0]["prediction"]
    narrative = explanation.to_narrative(output_format="text", expertise_level="beginner")

    assert prediction["predict"] == pytest.approx(expected_probability)
    assert json_prediction["predict"] == pytest.approx(expected_probability)
    assert _extract_narrative_probability(narrative) == pytest.approx(
        expected_probability, abs=1e-3
    )


@pytest.mark.parametrize("summary", ["regularized_mean", "mean", "lower", "upper"])
def test_interval_summary_aligns_factual_prediction_payloads_with_predict_proba(
    cls_wrapper, summary
):
    wrapper, x_test = cls_wrapper
    expected_probability = _selected_probability_from_predict_proba(
        wrapper.predict_proba(x_test[:1], interval_summary=summary),
        class_index=1,
    )

    explanation = wrapper.explain_factual(x_test[:1], interval_summary=summary)[0]

    assert_probability_surface_alignment(explanation, expected_probability)


@pytest.mark.parametrize("summary", ["regularized_mean", "mean", "lower", "upper"])
def test_interval_summary_aligns_alternative_prediction_payloads_with_predict_proba(
    cls_wrapper, summary
):
    wrapper, x_test = cls_wrapper
    expected_probability = _selected_probability_from_predict_proba(
        wrapper.predict_proba(x_test[:1], interval_summary=summary),
        class_index=1,
    )

    explanation = wrapper.explore_alternatives(x_test[:1], interval_summary=summary)[0]

    assert_probability_surface_alignment(explanation, expected_probability)


def test_interval_summary_propagates_through_direct_core_plugin_and_legacy_paths(cls_wrapper):
    wrapper, x_test = cls_wrapper
    core = wrapper.explainer
    assert core is not None
    expected_probability = _selected_probability_from_predict_proba(
        core.predict_proba(x_test[:1], interval_summary="lower"),
        class_index=1,
    )

    plugin_explanation = core.explain_factual(x_test[:1], interval_summary="lower")[0]
    legacy_explanation = core.explain_factual(
        x_test[:1], interval_summary="lower", _use_plugin=False
    )[0]

    assert_probability_surface_alignment(plugin_explanation, expected_probability)
    assert_probability_surface_alignment(legacy_explanation, expected_probability)


def test_interval_summary_aligns_reject_filtered_explanations_with_predict_proba(cls_wrapper):
    wrapper, x_test = cls_wrapper
    expected_probability = _selected_probability_from_predict_proba(
        wrapper.predict_proba(
            x_test[:1],
            interval_summary="lower",
            reject_policy="flag",
            reject_confidence=0.8,
        ),
        class_index=1,
    )

    explanation = wrapper.explain_factual(
        x_test[:1],
        interval_summary="lower",
        reject_policy="flag",
        reject_confidence=0.8,
    )[0]

    assert_probability_surface_alignment(explanation, expected_probability)


def test_interval_summary_changes_feature_effect_predictions(cls_wrapper):
    wrapper, x_test = cls_wrapper
    lower = wrapper.explain_factual(x_test[:1], interval_summary="lower")[0]
    upper = wrapper.explain_factual(x_test[:1], interval_summary="upper")[0]

    assert not np.allclose(lower.feature_predict["predict"], upper.feature_predict["predict"])


@pytest.mark.parametrize("summary", ["mean", "lower", "upper"])
def test_interval_summary_aligns_regression_threshold_explanations_with_predict_proba(
    reg_wrapper, summary
):
    wrapper, x_test = reg_wrapper
    threshold = 0.0
    expected_probability = _selected_probability_from_predict_proba(
        wrapper.predict_proba(x_test[:1], threshold=threshold, interval_summary=summary),
        class_index=1,
    )

    explanation = wrapper.explain_factual(
        x_test[:1], threshold=threshold, interval_summary=summary
    )[0]

    assert_probability_surface_alignment(explanation, expected_probability)


@pytest.mark.parametrize("summary", ["mean", "lower", "upper"])
def test_interval_summary_aligns_multiclass_prediction_payloads_with_predict_proba(
    multiclass_wrapper, summary
):
    wrapper, x_test = multiclass_wrapper
    predicted_class = int(wrapper.predict(x_test[:1], interval_summary=summary)[0])
    expected_probability = _selected_probability_from_predict_proba(
        wrapper.predict_proba(x_test[:1], interval_summary=summary),
        class_index=predicted_class,
    )

    explanation = wrapper.explain_factual(x_test[:1], interval_summary=summary)[0]

    assert_probability_surface_alignment(explanation, expected_probability)


def _snapshot_public_runtime_state(explainer: CalibratedExplainer) -> dict[str, object]:
    return {
        "mode": explainer.mode,
        "suppress_crepes_errors": explainer.suppress_crepes_errors,
        "seed": explainer.seed,
        "sample_percentiles": list(explainer.sample_percentiles),
        "features_to_ignore": list(explainer.features_to_ignore),
        "fast": explainer.fast,
        "noise_type": explainer.noise_type,
        "scale_factor": explainer.scale_factor,
        "severity": explainer.severity,
    }


def test_task50_mode_normalization_is_consistent_for_wrapper_and_direct_core(data):
    x, y_cls, _ = data
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x[:60], y_cls[:60])
    wrapper.calibrate(x[60:100], y_cls[60:100], mode="Classification")

    core = CalibratedExplainer(model, x[60:100], y_cls[60:100], mode="Classification")

    assert wrapper.explainer is not None
    assert wrapper.explainer.mode == "classification"
    assert core.mode == "classification"


@pytest.mark.parametrize(
    ("invalid_kwargs", "param"),
    [
        ({"suppress_crepes_errors": "no"}, "suppress_crepes_errors"),
        ({"fast": "yes"}, "fast"),
        ({"sample_percentiles": []}, "sample_percentiles"),
        ({"sample_percentiles": [80, 20]}, "sample_percentiles"),
        ({"sample_percentiles": [101]}, "sample_percentiles"),
        ({"features_to_ignore": [99]}, "features_to_ignore"),
        ({"seed": "42"}, "seed"),
        ({"fast": True, "noise_type": "laplace"}, "noise_type"),
        ({"fast": True, "scale_factor": 0}, "scale_factor"),
        ({"fast": True, "severity": -1}, "severity"),
    ],
)
def test_task50_constructor_boundary_matrix_matches_between_wrapper_and_direct_core(
    data, invalid_kwargs, param
):
    x, y_cls, _ = data
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x[:60], y_cls[:60])
    wrapper.calibrate(x[60:100], y_cls[60:100], mode="classification")

    assert wrapper.explainer is not None
    before = _snapshot_public_runtime_state(wrapper.explainer)
    explainer_id = id(wrapper.explainer)

    with pytest.raises(ValidationError) as wrap_exc:
        wrapper.calibrate(x[60:100], y_cls[60:100], mode="classification", **invalid_kwargs)

    with pytest.raises(ValidationError) as core_exc:
        CalibratedExplainer(
            model, x[60:100], y_cls[60:100], mode="classification", **invalid_kwargs
        )

    assert wrap_exc.value.details == core_exc.value.details
    assert wrap_exc.value.details is not None
    assert wrap_exc.value.details.get("param") == param
    assert wrapper.calibrated
    assert wrapper.explainer is not None
    assert id(wrapper.explainer) == explainer_id
    assert _snapshot_public_runtime_state(wrapper.explainer) == before


@pytest.mark.parametrize(
    ("call_kwargs", "param"),
    [
        ({"multi_labels_enabled": "yes"}, "multi_labels_enabled"),
        ({"features_to_ignore": [99]}, "features_to_ignore"),
    ],
)
def test_task50_classification_explain_invalid_values_match_between_wrapper_and_direct_core(
    cls_wrapper, call_kwargs, param
):
    wrapper, x_test = cls_wrapper
    core = wrapper.explainer
    assert core is not None

    before = _snapshot_public_runtime_state(core)

    with pytest.raises(ValidationError) as wrap_exc:
        wrapper.explain_factual(x_test[:2], **call_kwargs)
    with pytest.raises(ValidationError) as core_exc:
        core.explain_factual(x_test[:2], **call_kwargs)

    assert wrap_exc.value.details == core_exc.value.details
    assert wrap_exc.value.details is not None
    assert wrap_exc.value.details.get("param") == param
    assert _snapshot_public_runtime_state(core) == before


def test_task50_regression_low_high_percentiles_short_tuple_matches_between_wrapper_and_direct_core(
    reg_wrapper,
):
    wrapper, x_test = reg_wrapper
    core = wrapper.explainer
    assert core is not None

    before = _snapshot_public_runtime_state(core)

    with pytest.raises(ValidationError) as wrap_exc:
        wrapper.explain_factual(x_test[:2], low_high_percentiles=(95,))
    with pytest.raises(ValidationError) as core_exc:
        core.explain_factual(x_test[:2], low_high_percentiles=(95,))

    assert wrap_exc.value.details == core_exc.value.details
    assert wrap_exc.value.details == {
        "param": "low_high_percentiles",
        "value": [95],
        "actual_length": 1,
    }
    assert _snapshot_public_runtime_state(core) == before


@pytest.mark.parametrize("method_name", ["predict", "predict_proba", "explain_factual"])
def test_task50_threshold_length_mismatch_raises_validation_error_across_public_surfaces(
    reg_wrapper, method_name
):
    wrapper, x_test = reg_wrapper
    core = wrapper.explainer
    assert core is not None
    threshold = []
    before = _snapshot_public_runtime_state(core)

    def call(target):
        method = getattr(target, method_name)
        if method_name == "predict_proba":
            return method(x_test[:2], threshold=threshold)
        return method(x_test[:2], threshold=threshold)

    with pytest.raises(ValidationError) as wrap_exc:
        call(wrapper)
    with pytest.raises(ValidationError) as core_exc:
        call(core)

    assert wrap_exc.value.details == core_exc.value.details
    assert wrap_exc.value.details == {
        "param": "threshold",
        "expected_length": 2,
        "actual_length": 0,
    }
    assert _snapshot_public_runtime_state(core) == before


def test_task50_cleared_controls_remain_rejected(data):
    x, _, y_reg = data
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x[:60], y_reg[:60])

    with pytest.raises(ValidationError, match="sample_percentiles must be sorted"):
        wrapper.calibrate(x[60:100], y_reg[60:100], mode="regression", sample_percentiles=[75, 25])

    with pytest.raises(ValidationError, match="condition_source must be either"):
        wrapper.calibrate(
            x[60:100],
            y_reg[60:100],
            mode="regression",
            condition_source=" prediction ",
        )


# ---------------------------------------------------------------------------
# 5. T12 (#206): multi_labels_enabled / interval_summary graduated from the
#    ADR-038 §3 experimental **kwargs route to explicit keyword-only formals.
# ---------------------------------------------------------------------------

_GRADUATED_EXPLAIN_PARAMS = ("multi_labels_enabled", "interval_summary")


@pytest.mark.parametrize(
    "method",
    [
        CalibratedExplainer.explain_factual,
        CalibratedExplainer.explore_alternatives,
        WrapCalibratedExplainer.explain_factual,
        WrapCalibratedExplainer.explore_alternatives,
    ],
    ids=[
        "core.explain_factual",
        "core.explore_alternatives",
        "wrap.explain_factual",
        "wrap.explore_alternatives",
    ],
)
def test_should_expose_graduated_explain_params_as_typed_keyword_only_formals(method):
    params = inspect.signature(method).parameters

    multi = params["multi_labels_enabled"]
    summary = params["interval_summary"]

    assert multi.kind is inspect.Parameter.KEYWORD_ONLY
    assert summary.kind is inspect.Parameter.KEYWORD_ONLY
    assert multi.default is False
    assert summary.default is None
    assert multi.annotation == "bool"
    assert summary.annotation == "IntervalSummary | str | None"
    # The graduated names stay on the explain allow-list (single source of truth
    # for the wrapper gate and the cross-surface check).
    assert set(_GRADUATED_EXPLAIN_PARAMS) <= CE_EXPLAIN_KWARGS
    assert WRAP_EXPLAIN_KWARGS == CE_EXPLAIN_KWARGS


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
@pytest.mark.parametrize("bad_value", ["yes", 1, None], ids=["str", "int", "none"])
def test_should_reject_non_bool_multi_labels_enabled_identically_on_wrapper_and_core(
    cls_wrapper, method_name, bad_value
):
    wrapper, x_test = cls_wrapper
    core = wrapper.explainer
    assert core is not None

    with pytest.raises(ValidationError) as wrap_exc:
        getattr(wrapper, method_name)(x_test[:1], multi_labels_enabled=bad_value)
    with pytest.raises(ValidationError) as core_exc:
        getattr(core, method_name)(x_test[:1], multi_labels_enabled=bad_value)

    assert wrap_exc.value.details == core_exc.value.details
    assert core_exc.value.details["param"] == "multi_labels_enabled"
    assert core_exc.value.details["actual_type"] == type(bad_value).__name__


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
@pytest.mark.parametrize("bad_value", ["median", "LOWER", 3], ids=["unknown", "wrong-case", "int"])
def test_should_reject_invalid_interval_summary_identically_on_wrapper_and_core(
    cls_wrapper, method_name, bad_value
):
    wrapper, x_test = cls_wrapper
    core = wrapper.explainer
    assert core is not None

    with pytest.raises(ValidationError) as wrap_exc:
        getattr(wrapper, method_name)(x_test[:1], interval_summary=bad_value)
    with pytest.raises(ValidationError) as core_exc:
        getattr(core, method_name)(x_test[:1], interval_summary=bad_value)

    assert wrap_exc.value.details == core_exc.value.details
    assert core_exc.value.details["param"] == "interval_summary"
    assert core_exc.value.details["valid_values"] == [
        "regularized_mean",
        "mean",
        "lower",
        "upper",
    ]


@pytest.fixture(scope="module")
def lower_summary_cls_wrapper(data):
    """Classification wrapper whose explainer-level strategy is LOWER."""
    x, y_cls, _ = data
    wrapper = _fitted_classification_wrapper(data)
    wrapper.calibrate(x[60:100], y_cls[60:100], interval_summary="lower")
    return wrapper, x[100:]


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
def test_should_keep_explainer_level_interval_summary_when_call_omits_it(
    lower_summary_cls_wrapper, method_name
):
    wrapper, x_test = lower_summary_cls_wrapper
    lower_probability = _selected_probability_from_predict_proba(
        wrapper.predict_proba(x_test[:1], interval_summary="lower"), class_index=1
    )
    regularized_probability = _selected_probability_from_predict_proba(
        wrapper.predict_proba(x_test[:1], interval_summary="regularized_mean"), class_index=1
    )
    assert lower_probability != pytest.approx(regularized_probability)
    explain = getattr(wrapper, method_name)

    omitted = explain(x_test[:1])[0]
    explicit_none = explain(x_test[:1], interval_summary=None)[0]
    overridden = explain(x_test[:1], interval_summary=IntervalSummary.REGULARIZED_MEAN)[0]

    assert_probability_surface_alignment(omitted, lower_probability)
    assert_probability_surface_alignment(explicit_none, lower_probability)
    assert_probability_surface_alignment(overridden, regularized_probability)


@pytest.mark.parametrize("summary", list(IntervalSummary), ids=lambda s: s.value)
def test_should_treat_interval_summary_enum_and_string_identically(cls_wrapper, summary):
    wrapper, x_test = cls_wrapper

    from_enum = wrapper.explain_factual(x_test[:1], interval_summary=summary)[0]
    from_string = wrapper.explain_factual(x_test[:1], interval_summary=summary.value)[0]

    assert from_enum.prediction["predict"] == pytest.approx(from_string.prediction["predict"])
    assert np.allclose(from_enum.feature_predict["predict"], from_string.feature_predict["predict"])


@pytest.mark.parametrize(
    ("call_summary", "expected"),
    [("upper", IntervalSummary.UPPER), (None, IntervalSummary.LOWER)],
    ids=["call-override", "explainer-default"],
)
def test_should_deliver_canonical_interval_summary_to_plugin_request_field_only(
    lower_summary_cls_wrapper, monkeypatch, call_summary, expected
):
    wrapper, x_test = lower_summary_cls_wrapper
    core = wrapper.explainer
    assert core is not None
    plugin, _identifier = core.explanation_orchestrator.ensure_plugin("factual")
    original_explain_batch = plugin.explain_batch
    captured_requests = []

    def _capturing_explain_batch(x, request):
        captured_requests.append(request)
        return original_explain_batch(x, request)

    monkeypatch.setattr(plugin, "explain_batch", _capturing_explain_batch)

    core.explain_factual(x_test[:1], interval_summary=call_summary)

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.interval_summary is expected
    assert "interval_summary" not in request.extras
    assert "multi_labels_enabled" not in request.extras


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
def test_should_return_one_explanation_per_class_when_multi_labels_enabled(
    multiclass_wrapper, method_name
):
    wrapper, x_test = multiclass_wrapper
    core = wrapper.explainer
    assert core is not None
    n_classes = len(np.unique(core.y_cal))
    assert n_classes == 3

    result = getattr(wrapper, method_name)(x_test[:2], multi_labels_enabled=True)

    assert isinstance(result, MultiClassCalibratedExplanations)
    assert len(result.explanations) == 2
    for per_instance in result.explanations:
        assert sorted(per_instance) == list(range(n_classes))


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
def test_should_apply_interval_summary_to_each_class_when_multi_labels_enabled(
    multiclass_wrapper, method_name
):
    wrapper, x_test = multiclass_wrapper
    proba = wrapper.predict_proba(x_test[:1], interval_summary="upper")

    result = getattr(wrapper, method_name)(
        x_test[:1], multi_labels_enabled=True, interval_summary="upper"
    )

    for class_index, explanation in result.explanations[0].items():
        expected = _selected_probability_from_predict_proba(proba, class_index=class_index)
        assert explanation.prediction["predict"] == pytest.approx(expected)


def test_should_warn_when_multi_labels_enabled_on_binary_problem(cls_wrapper):
    wrapper, x_test = cls_wrapper

    with pytest.warns(UserWarning, match="appears to be binary"):
        result = wrapper.explain_factual(x_test[:1], multi_labels_enabled=True)

    assert isinstance(result, MultiClassCalibratedExplanations)
    assert sorted(result.explanations[0]) == [0, 1]


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
def test_should_reject_multi_labels_enabled_for_regression(reg_wrapper, method_name):
    wrapper, x_test = reg_wrapper
    core = wrapper.explainer
    assert core is not None

    with pytest.raises(ValidationError) as wrap_exc:
        getattr(wrapper, method_name)(x_test[:1], multi_labels_enabled=True)
    with pytest.raises(ValidationError) as core_exc:
        getattr(core, method_name)(x_test[:1], multi_labels_enabled=True)

    assert wrap_exc.value.details == core_exc.value.details
    assert core_exc.value.details["param"] == "multi_labels_enabled"
    assert core_exc.value.details["mode"] == "regression"


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
@pytest.mark.parametrize(
    ("call_kwargs", "ignored"),
    [
        ({"multi_labels_enabled": True}, "multi_labels_enabled"),
        ({"interval_summary": "upper"}, "interval_summary"),
    ],
    ids=["multi_labels", "interval_summary"],
)
def test_should_warn_that_guarded_path_ignores_graduated_controls(
    multiclass_wrapper, method_name, call_kwargs, ignored
):
    wrapper, x_test = multiclass_wrapper

    with pytest.warns(
        UserWarning, match=rf"{ignored} is not supported together with guarded_options"
    ):
        result = getattr(wrapper, method_name)(
            x_test[:1], guarded_options=GuardedOptions(), **call_kwargs
        )

    assert not isinstance(result, MultiClassCalibratedExplanations)
    assert len(result.explanations) == 1


def test_should_keep_call_time_interval_summary_when_fast_feature_filter_is_enabled(
    data, monkeypatch
):
    """Regression: the FAST feature-filter request rebuild dropped interval_summary,
    so explanations silently reverted to the default strategy."""
    monkeypatch.setenv("CE_FEATURE_FILTER", "on,top_k=2")
    x, y_cls, _ = data
    wrapper = _fitted_classification_wrapper(data)
    wrapper.calibrate(x[60:100], y_cls[60:100])
    x_test = x[100:]
    expected_probability = _selected_probability_from_predict_proba(
        wrapper.predict_proba(x_test[:1], interval_summary="lower"), class_index=1
    )

    explanation = wrapper.explain_factual(x_test[:1], interval_summary="lower")[0]

    assert explanation.prediction["predict"] == pytest.approx(expected_probability)
