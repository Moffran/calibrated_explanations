import numpy as np


def test_collection_to_batch_and_validate():
    from calibrated_explanations.explanations.explanations import CalibratedExplanations
    from calibrated_explanations.plugins.builtins import collection_to_batch
    from calibrated_explanations.plugins.explanations import validate_explanation_batch

    class DummyExplainer:
        pass

    x = np.asarray([[1.0, 2.0]])
    coll = CalibratedExplanations(DummyExplainer(), x, y_threshold=0.5, bins=None)
    # ensure empty explanations list
    coll.explanations = []
    batch = collection_to_batch(coll)
    assert batch is not None
    assert hasattr(batch, "collection_metadata")

    # validation should accept the batch
    validated = validate_explanation_batch(batch)
    assert validated is batch
