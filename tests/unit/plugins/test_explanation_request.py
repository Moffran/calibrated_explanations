import numpy as np
from types import MappingProxyType

from calibrated_explanations.plugins.explanations import ExplanationRequest


def test_explanation_request_freezes_bins_and_extras():
    bins = np.array([1, 2, 3])
    extras = {"foo": [1, 2, 3], "bar": {"x": 1}}
    req = ExplanationRequest(
        threshold=None,
        low_high_percentiles=None,
        bins=bins,
        features_to_ignore=(),
        extras=extras,
        feature_filter_per_instance_ignore=None,
    )

    # bins should be converted to an immutable tuple structure
    assert isinstance(req.bins, tuple)
    assert req.bins == (1, 2, 3)

    # extras should be a MappingProxyType (immutable mapping)
    assert isinstance(req.extras, MappingProxyType)
    assert req.extras["foo"] == (1, 2, 3)
