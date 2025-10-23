from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.utils import helper as helper_module


def test_safe_first_element_handles_various_shapes():
    assert helper_module.safe_first_element(0.5) == 0.5
    assert helper_module.safe_first_element([], default=1.23) == 1.23

    matrix = np.array([[0.1, 0.2, 0.3]])
    assert helper_module.safe_first_element(matrix) == pytest.approx(0.1)
    assert helper_module.safe_first_element(matrix, col=1) == pytest.approx(0.2)

    vector = np.array([9.0, 8.0])
    assert helper_module.safe_first_element(vector, col=1) == pytest.approx(8.0)
    assert helper_module.safe_first_element(vector, col=5) == pytest.approx(0.0)


def test_safe_mean_returns_default_on_failures():
    assert helper_module.safe_mean([]) == 0.0
    assert helper_module.safe_mean([1, 2, 3]) == pytest.approx(2.0)

    class BadSeq:
        def __iter__(self):
            raise RuntimeError("boom")

    assert helper_module.safe_mean(BadSeq(), default=4.2) == 4.2
