from __future__ import annotations

import pytest

from calibrated_explanations import cache, parallel


def test_cache_module_rejects_unexpected_attributes():
    with pytest.raises(AttributeError):
        _ = cache.__getattr__("missing")


def test_parallel_module_rejects_unexpected_attributes():
    with pytest.raises(AttributeError):
        _ = parallel.__getattr__("missing")
