import os
from typing import Any

import numpy as np
import pandas as pd
import pytest
from unittest.mock import create_autospec

from calibrated_explanations.core.calibrated_explainer import (
    CalibratedExplainer,
    _PredictBridgeMonitor,
    _coerce_string_tuple,
    _read_pyproject_section,
)
from calibrated_explanations.core.exceptions import DataShapeError
from calibrated_explanations.plugins.predict import PredictBridge
from calibrated_explanations.plugins.registry import EXPLANATION_PROTOCOL_VERSION


class DummyLearner:
    """Minimal learner implementation for calibration-focused tests."""

    def __init__(
        self,
        *,
        mode: str = "classification",
        oob_decision_function: np.ndarray | None = None,
        oob_prediction: np.ndarray | None = None,
    ) -> None:
        self.mode = mode
        self.fitted_ = True  # ensures check_is_fitted succeeds
        self.oob_decision_function_ = oob_decision_function
        self.oob_prediction_ = oob_prediction

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DummyLearner":  # pragma: no cover - unused
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        probs = np.zeros((len(x), 2))
        probs[:, 0] = 0.4
        probs[:, 1] = 0.6
        return probs


class DummyIntervalLearner:
    """Interval learner returning deterministic zero arrays."""

    def predict_uncertainty(self, x: np.ndarray, *_args: Any, **_kwargs: Any) -> tuple[np.ndarray, ...]:
        n = x.shape[0]
        zeros = np.zeros(n)
        return zeros, zeros, zeros, None

    def predict_probability(self, x: np.ndarray, *_args: Any, **_kwargs: Any) -> tuple[np.ndarray, ...]:
        n = x.shape[0]
        zeros = np.zeros(n)
        return zeros, zeros, zeros, None


def _patch_interval_initializers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure interval initialization is lightweight for focused unit tests."""

    def _initialize(explainer: CalibratedExplainer, *_args: Any, **_kwargs: Any) -> None:
        explainer.interval_learner = DummyIntervalLearner()
        explainer._CalibratedExplainer__initialized = True  # noqa: SLF001

    monkeypatch.setattr(
        "calibrated_explanations.core.calibration_helpers.initialize_interval_learner",
        _initialize,
    )
    monkeypatch.setattr(
        "calibrated_explanations.core.calibration_helpers.initialize_interval_learner_for_fast_explainer",
        _initialize,
    )


def _make_explainer(
    monkeypatch: pytest.MonkeyPatch,
    learner: DummyLearner,
    x_cal: np.ndarray,
    y_cal: Any,
    **kwargs: Any,
) -> CalibratedExplainer:
    _patch_interval_initializers(monkeypatch)
    return CalibratedExplainer(learner, x_cal, y_cal, **kwargs)


def test_read_pyproject_section_handles_multiple_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: "os.PathLike[str]") -> None:
    module = __import__("calibrated_explanations.core.calibrated_explainer", fromlist=["_tomllib"])
    monkeypatch.chdir(tmp_path)

    # No TOML reader available -> early fallback
    monkeypatch.setattr(module, "_tomllib", None)
    assert _read_pyproject_section(("tool",)) == {}

    class DummyToml:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def load(self, _fh: Any) -> dict[str, Any]:
            return self._payload

    # File missing -> still fallback
    monkeypatch.setattr(module, "_tomllib", DummyToml({}))
    assert _read_pyproject_section(("tool", "missing")) == {}

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool]\nname='demo'\n", encoding="utf-8")

    # Value present but not a mapping -> coerced to empty result
    monkeypatch.setattr(
        module,
        "_tomllib",
        DummyToml({"tool": {"calibrated_explanations": {"explanations": ["value"]}}}),
    )
    assert (
        _read_pyproject_section(("tool", "calibrated_explanations", "explanations"))
        == {}
    )

    # Proper mapping -> returned as dictionary copy
    monkeypatch.setattr(
        module,
        "_tomllib",
        DummyToml({"tool": {"calibrated_explanations": {"explanations": {"key": "value"}}}}),
    )
    assert _read_pyproject_section(("tool", "calibrated_explanations", "explanations")) == {
        "key": "value"
    }


def test_coerce_string_tuple_variants() -> None:
    assert _coerce_string_tuple("alpha") == ("alpha",)
    assert _coerce_string_tuple(["alpha", "", "beta", 1]) == ("alpha", "beta")
    assert _coerce_string_tuple(None) == ()
    assert _coerce_string_tuple(123) == ()


def test_predict_bridge_monitor_tracks_usage() -> None:
    bridge = create_autospec(PredictBridge, instance=True)
    monitor = _PredictBridgeMonitor(bridge)

    payload = {"x": np.ones((2, 2))}
    monitor.predict(payload, mode="factual", task="classification")
    monitor.predict_interval(payload, task="classification")
    monitor.predict_proba(payload)

    assert monitor.calls == ("predict", "predict_interval", "predict_proba")
    assert monitor.used

    monitor.reset_usage()
    assert monitor.calls == ()
    assert not monitor.used


def test_oob_predictions_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = DummyLearner(oob_decision_function=np.array([0.2, 0.7, 0.8]))
    x_cal = np.arange(3).reshape(-1, 1)
    y_cal = np.array([0, 1, 0])
    explainer = _make_explainer(monkeypatch, learner, x_cal, y_cal, oob=True)

    assert np.array_equal(explainer.y_cal, np.array([0, 1, 1]))


def test_oob_predictions_multiclass_categorical(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas.core.arrays.categorical  # noqa: F401  ensure module is available
    calls: list[str] = []

    def fake_safe_isinstance(_obj: Any, name: str) -> bool:
        calls.append(name)
        return name == "pandas.core.arrays.categorical.Categorical"

    monkeypatch.setattr(
        "calibrated_explanations.core.calibrated_explainer.safe_isinstance",
        fake_safe_isinstance,
    )

    learner = DummyLearner(
        oob_decision_function=np.array(
            [[0.1, 0.2, 0.7], [0.6, 0.2, 0.2], [0.2, 0.5, 0.3]]
        )
    )
    x_cal = np.arange(3).reshape(-1, 1)
    y_cal = pd.Categorical(["cat", "dog", "bird"])
    explainer = _make_explainer(monkeypatch, learner, x_cal, y_cal, oob=True)

    assert "pandas.core.arrays.categorical.Categorical" in calls
    assert explainer.label_map == {"bird": 0, "cat": 1, "dog": 2}
    assert explainer.class_labels == {0: "bird", 1: "cat", 2: "dog"}
    assert np.array_equal(explainer.y_cal, np.array([2, 0, 1]))


def test_oob_predictions_regression_length_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = DummyLearner(mode="regression", oob_prediction=np.array([1.0, 2.0]))
    x_cal = np.arange(6).reshape(-1, 2)
    y_cal = np.linspace(0.0, 1.0, 3)

    with pytest.raises(DataShapeError):
        _make_explainer(monkeypatch, learner, x_cal, y_cal, mode="regression", oob=True)


def test_build_explanation_chain_includes_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = DummyLearner()
    x_cal = np.ones((2, 2))
    y_cal = np.array([0, 1])
    explainer = _make_explainer(monkeypatch, learner, x_cal, y_cal)

    explainer._explanation_plugin_overrides["factual"] = "override.plugin"
    explainer._pyproject_explanations = {
        "factual": "py.plugin",
        "factual_fallbacks": ("py.fb",),
    }

    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FACTUAL", "env.plugin")
    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS", "env.fb1, env.fb2")

    class Descriptor:
        def __init__(self, metadata: dict[str, Any]) -> None:
            self.metadata = metadata

    descriptor_map = {
        "override.plugin": Descriptor({"fallbacks": ("meta.fb",)}),
        "env.plugin": Descriptor({}),
        "env.fb1": Descriptor({}),
        "env.fb2": Descriptor({}),
        "py.plugin": Descriptor({}),
        "py.fb": Descriptor({}),
        "meta.fb": Descriptor({}),
        "core.explanation.factual": Descriptor({}),
    }
    monkeypatch.setattr(
        "calibrated_explanations.core.calibrated_explainer.find_explanation_descriptor",
        lambda identifier: descriptor_map.get(identifier),
    )

    chain = explainer._build_explanation_chain("factual")
    assert chain[0] == "override.plugin"
    assert "meta.fb" in chain
    assert chain[-1] == "core.explanation.factual"


def test_explanation_metadata_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = DummyLearner()
    x_cal = np.ones((2, 2))
    y_cal = np.array([0, 1])
    explainer = _make_explainer(monkeypatch, learner, x_cal, y_cal)

    base = {"schema_version": EXPLANATION_PROTOCOL_VERSION}
    assert "missing tasks" in explainer._check_explanation_runtime_metadata(
        base, identifier="demo", mode="factual"
    )

    bad_tasks = base | {"tasks": ("regression",)}
    assert "does not support task" in explainer._check_explanation_runtime_metadata(
        bad_tasks, identifier="demo", mode="factual"
    )

    missing_modes = base | {"tasks": ("classification",)}
    assert "missing modes" in explainer._check_explanation_runtime_metadata(
        missing_modes, identifier="demo", mode="factual"
    )

    wrong_mode = missing_modes | {"modes": ("fast",)}
    assert "does not declare mode" in explainer._check_explanation_runtime_metadata(
        wrong_mode, identifier="demo", mode="factual"
    )

    missing_caps = missing_modes | {"modes": ("factual",)}
    assert "missing required capabilities" in explainer._check_explanation_runtime_metadata(
        missing_caps, identifier="demo", mode="factual"
    )

    ok = missing_caps | {
        "capabilities": ("explain", "explanation:factual", "task:both"),
    }
    assert (
        explainer._check_explanation_runtime_metadata(ok, identifier="demo", mode="factual")
        is None
    )


def test_instantiate_plugin_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = DummyLearner()
    x_cal = np.ones((2, 2))
    y_cal = np.array([0, 1])
    explainer = _make_explainer(monkeypatch, learner, x_cal, y_cal)

    assert explainer._instantiate_plugin(None) is None

    def plugin_factory() -> str:
        return "plugin"

    plugin_factory.plugin_meta = {"name": "factory"}  # type: ignore[attr-defined]
    assert explainer._instantiate_plugin(plugin_factory) is plugin_factory

    class Prototype:
        def __init__(self) -> None:
            self.value = "fresh"

    proto = Prototype()
    inst = explainer._instantiate_plugin(proto)
    assert isinstance(inst, Prototype)
    assert inst is not proto


def test_runtime_and_preprocessor_metadata_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = DummyLearner()
    x_cal = np.ones((2, 2))
    y_cal = np.array([0, 1])
    explainer = _make_explainer(monkeypatch, learner, x_cal, y_cal)

    assert explainer.runtime_telemetry == {}

    explainer.set_preprocessor_metadata({"scale": 2})
    assert explainer.preprocessor_metadata == {"scale": 2}

    explainer.set_preprocessor_metadata(None)
    assert explainer.preprocessor_metadata is None


def test_x_y_cal_setters_and_append(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = DummyLearner()
    x_cal = np.ones((2, 2))
    y_cal = np.array([0, 1])
    explainer = _make_explainer(monkeypatch, learner, x_cal, y_cal)

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    explainer.x_cal = df
    assert isinstance(explainer.x_cal, np.ndarray)

    explainer.y_cal = np.array([[1], [0]])
    assert explainer.y_cal.shape == (2,)

    with pytest.raises(DataShapeError):
        explainer.append_cal(np.ones((1, explainer.num_features + 1)), np.array([1]))

    explainer.append_cal(np.ones((1, explainer.num_features)), np.array([1]))
    assert explainer.y_cal.shape[0] == 3


def test_ensure_interval_state_and_coerce_override(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = DummyLearner()
    x_cal = np.ones((2, 2))
    y_cal = np.array([0, 1])
    explainer = _make_explainer(monkeypatch, learner, x_cal, y_cal)

    # Remove keys to emulate legacy state
    for key in [
        "_interval_plugin_hints",
        "_interval_plugin_fallbacks",
        "_interval_plugin_identifiers",
        "_telemetry_interval_sources",
        "_interval_preferred_identifier",
        "_interval_context_metadata",
    ]:
        delattr(explainer, key)

    explainer._ensure_interval_runtime_state()
    assert "default" in explainer._interval_plugin_identifiers

    assert explainer._coerce_plugin_override(None) is None
    assert explainer._coerce_plugin_override("plugin") == "plugin"

    def factory() -> str:
        return "plugin"

    override = explainer._coerce_plugin_override(factory)
    assert override == "plugin"


def test_interval_metadata_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = DummyLearner(mode="regression")
    x_cal = np.ones((2, 2))
    y_cal = np.array([0.1, 0.2])
    explainer = _make_explainer(monkeypatch, learner, x_cal, y_cal, mode="regression")

    base = {"schema_version": 1}
    assert "metadata unavailable" in explainer._check_interval_runtime_metadata(
        None, identifier="demo", fast=False
    )

    bad_version = base | {"schema_version": 2}
    assert "unsupported interval" in explainer._check_interval_runtime_metadata(
        bad_version, identifier="demo", fast=False
    )

    missing_modes = base | {"capabilities": ("interval:regression",)}
    assert "missing modes" in explainer._check_interval_runtime_metadata(
        missing_modes, identifier="demo", fast=False
    )

    wrong_mode = missing_modes | {"modes": ("classification",)}
    assert "does not support mode" in explainer._check_interval_runtime_metadata(
        wrong_mode, identifier="demo", fast=False
    )

    missing_caps = missing_modes | {"modes": ("regression",), "capabilities": ()}
    assert "missing capability" in explainer._check_interval_runtime_metadata(
        missing_caps, identifier="demo", fast=False
    )

    ok = missing_caps | {"capabilities": ("interval:regression",), "fast_compatible": True}
    assert (
        explainer._check_interval_runtime_metadata(ok, identifier="demo", fast=False)
        is None
    )


def test_gather_interval_hints(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = DummyLearner()
    x_cal = np.ones((2, 2))
    y_cal = np.array([0, 1])
    explainer = _make_explainer(monkeypatch, learner, x_cal, y_cal)

    explainer._interval_plugin_hints = {
        "factual": ("one", "two"),
        "alternative": ("two", "three"),
        "fast": ("fast-one",),
    }
    assert explainer._gather_interval_hints(fast=True) == ("fast-one",)
    assert explainer._gather_interval_hints(fast=False) == ("one", "two", "three")
