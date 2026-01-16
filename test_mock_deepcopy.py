from copy import deepcopy
from unittest.mock import MagicMock


def test_deepcopy_mock():
    m = MagicMock()
    m.num_features = 3
    m.list_prop = [1, 2, 3]

    m_copy = deepcopy(m)

    print(f"Original num_features: {m.num_features}")
    print(f"Copy num_features: {m_copy.num_features}")

    print(f"Original list_prop: {m.list_prop}")
    print(f"Copy list_prop: {m_copy.list_prop}")


if __name__ == "__main__":
    test_deepcopy_mock()
