import numpy as np
import pytest


def test_validate_explanation_batch_errors():
    from calibrated_explanations.plugins.explanations import (
        ExplanationBatch,
        validate_explanation_batch,
    )
    from calibrated_explanations.utils.exceptions import ValidationError

    # Non-batch object raises
    with pytest.raises(ValidationError):
        validate_explanation_batch(object())

    # Batch with non-type container_cls
    bad = ExplanationBatch(
        container_cls=123, explanation_cls=int, instances=(), collection_metadata={}
    )
    with pytest.raises(ValidationError):
        validate_explanation_batch(bad)

    # Batch with container_cls that is a type but does not inherit CalibratedExplanations
    class NotCal:
        pass

    bad2 = ExplanationBatch(
        container_cls=NotCal, explanation_cls=int, instances=(), collection_metadata={}
    )
    with pytest.raises(ValidationError):
        validate_explanation_batch(bad2)

    # Batch with explanation_cls not a class
    from calibrated_explanations.explanations.explanations import CalibratedExplanations

    bad3 = ExplanationBatch(
        container_cls=CalibratedExplanations,
        explanation_cls=123,
        instances=(),
        collection_metadata={},
    )
    with pytest.raises(ValidationError):
        validate_explanation_batch(bad3)


def test_legacy_predict_bridge_interval_invariant():
    from calibrated_explanations.plugins.builtins import LegacyPredictBridge
    from calibrated_explanations.utils.exceptions import ValidationError

    class DummyBad:
        def predict(self, x, uq_interval=False, bins=None, calibrated=False):
            preds = np.asarray([1.0 for _ in x])
            # low > high to trigger invariant
            low = np.asarray([0.9 for _ in x])
            high = np.asarray([0.1 for _ in x])
            if uq_interval:
                return preds, (low, high)
            return preds

    bridge = LegacyPredictBridge(DummyBad())
    x = [[1.0]]
    with pytest.raises(ValidationError):
        bridge.predict(x, mode="regression", task="regression")
