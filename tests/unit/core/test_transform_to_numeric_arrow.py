import pytest
import pandas as pd
from pandas.api.types import is_integer_dtype

from calibrated_explanations.utils.helper import transform_to_numeric


def test_transform_to_numeric_with_pyarrow_string_dtype():
    # Require pyarrow for Arrow-backed string dtype; skip otherwise
    pytest.importorskip("pyarrow")

    s_feat = pd.Series(["red", "blue", "red"], dtype="string[pyarrow]")
    s_target = pd.Series(["yes", "no", "yes"], dtype="string[pyarrow]")
    df = pd.DataFrame({"feat": s_feat, "target": s_target})

    ndf, categorical_features, categorical_labels, target_labels, mappings = transform_to_numeric(
        df.copy(), "target"
    )

    assert isinstance(mappings, dict)
    assert isinstance(target_labels, dict)
    assert categorical_features is not None
    assert categorical_labels is not None
    # The feature column should be mapped to integer labels (either Int64 or categorical with integer categories)
    from pandas import CategoricalDtype

    if isinstance(ndf["feat"].dtype, CategoricalDtype):
        # categories should be integer dtype
        assert is_integer_dtype(ndf["feat"].cat.categories.dtype)
    else:
        assert is_integer_dtype(ndf["feat"].dtype)
