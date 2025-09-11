import pandas as pd

from calibrated_explanations.utils.helper import transform_to_numeric


def test_transform_to_numeric_basic():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "target": ["t1", "t2"]})
    ndf, cat_feats, cat_labels, target_labels, mappings = transform_to_numeric(df.copy(), "target")
    assert "b" in df.columns
    assert isinstance(mappings, dict)
    # target labels mapping created
    assert isinstance(target_labels, dict)


def test_transform_to_numeric_with_existing_mappings():
    df = pd.DataFrame({"a": ["c", "d"], "target": ["u", "v"]})
    mappings = {"a": {"c": 0, "d": 1}, "target": {"u": 0, "v": 1}}
    ndf, cat_feats, cat_labels, target_labels, new_mappings = transform_to_numeric(
        df.copy(), "target", mappings
    )
    assert new_mappings == mappings
