from __future__ import annotations

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
