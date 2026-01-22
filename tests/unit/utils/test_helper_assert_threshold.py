import pytest


def test_assert_threshold_various():
    from calibrated_explanations.utils.helper import assert_threshold

    # None returns None
    assert assert_threshold(None, [1, 2, 3]) is None

    # Scalar
    assert assert_threshold(0.5, [1, 2]) == 0.5

    # Tuple of two
    assert assert_threshold((0.2, 0.8), [1, 2]) == (0.2, 0.8)

    # List of same length
    assert assert_threshold([0.1, 0.2], [1, 2]) == [0.1, 0.2]

    # Invalid tuple length raises
    with pytest.raises(AssertionError):
        assert_threshold([(0.1, 0.9)], [1, 2])


def test_find_interval_descriptor_none():
    from calibrated_explanations.plugins.registry import find_interval_descriptor

    # No plugins registered under this random id
    assert find_interval_descriptor("nonexistent.plugin.id") is None
