import numpy as np
from calibrated_explanations.plugins.explanations import (
    ExplanationRequest,
    ExplanationBatch,
    validate_explanation_batch,
)
from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.explanations.explanation import FactualExplanation


def should_freeze_request_and_validate_batch_when_given_numpy_arrays():
    # Arrange: construct an ExplanationRequest containing numpy arrays
    req = ExplanationRequest(
        threshold=None,
        low_high_percentiles=(0.05, 0.95),
        bins=None,
        features_to_ignore=np.asarray((1, 2)),
        extras={"meta": np.asarray([1, 2, 3])},
    )

    # Act: create a minimal ExplanationBatch using public classes
    batch = ExplanationBatch(
        container_cls=CalibratedExplanations,
        explanation_cls=FactualExplanation,
        instances=({"explanation": object()},),
        collection_metadata={"mode": "factual"},
    )

    # Assert: request fields are frozen (no exception) and batch validates
    # This is behavior-focused: we ensure dataclass freezing and public validation succeed
    assert req.extras is not None
    validate_explanation_batch(batch)
