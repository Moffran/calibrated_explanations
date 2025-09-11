import numpy as np
import pytest

from calibrated_explanations.utils import helper


def test_assert_threshold_scalars_and_lists():
    assert helper.assert_threshold(0.5, [1, 2]) == 0.5
    assert helper.assert_threshold((0.2, 0.8), [1, 2]) == (0.2, 0.8)
    with pytest.raises(AssertionError):
        helper.assert_threshold([0.1, 0.2, 0.3], [1, 2])


def test_safe_mean_and_first_element():
    assert helper.safe_mean([], default=3.14) == 3.14
    assert helper.safe_first_element([], default=2.5) == 2.5
    assert helper.safe_first_element(np.array([10])) == 10.0
