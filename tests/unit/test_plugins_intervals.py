from __future__ import annotations

from types import MappingProxyType

from calibrated_explanations.plugins.intervals import IntervalCalibratorContext


class BrokenMetadata:
    def __iter__(self):
        raise RuntimeError("no iteration")

    def items(self):
        return [("stored", "value")]


def test_interval_context_copies_against_iteration_errors():
    context = IntervalCalibratorContext(
        learner=object(),
        calibration_splits=[],
        bins={},
        residuals={},
        difficulty={},
        metadata=BrokenMetadata(),
        fast_flags={},
    )

    assert context.metadata == {"stored": "value"}


def test_interval_context_setstate_normalizes_metadata_and_plugin_state():
    context = IntervalCalibratorContext(
        learner=object(),
        calibration_splits=[],
        bins={},
        residuals={},
        difficulty={},
        metadata={"a": 1},
        fast_flags={},
        plugin_state={"ok": True},
    )

    state = context.__getstate__()
    state["metadata"] = {"restored": 1}
    state["plugin_state"] = (("k", "v"),)

    restored = object.__new__(IntervalCalibratorContext)
    restored.__setstate__(state)

    assert isinstance(restored.metadata, MappingProxyType)
    assert dict(restored.metadata) == {"restored": 1}
    assert isinstance(restored.plugin_state, dict)
    assert restored.plugin_state == {"k": "v"}


def test_interval_context_setstate_accepts_missing_metadata_branch():
    restored = object.__new__(IntervalCalibratorContext)
    state = {
        "learner": object(),
        "calibration_splits": [],
        "bins": {},
        "residuals": {},
        "difficulty": {},
        "metadata": None,
        "fast_flags": {},
        "plugin_state": {"ok": True},
    }

    restored.__setstate__(state)
    assert restored.metadata is None
    assert restored.plugin_state == {"ok": True}
