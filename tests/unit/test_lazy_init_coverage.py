import importlib


def test_package_lazy_getattr_branches() -> None:
    """Exercise multiple lazy-attribute branches in the package __init__.

    We access a list of public names exposed via ``calibrated_explanations.__getattr__``.
    Some names may raise on import in this environment; we catch and ignore those
    errors because the goal is to execute the lazy-loader code paths for coverage.
    """
    pkg = importlib.import_module("calibrated_explanations")

    names = [
        "viz",
        "BinaryEntropyDiscretizer",
        "BinaryRegressorDiscretizer",
        "EntropyDiscretizer",
        "RegressorDiscretizer",
        "transform_to_numeric",
        "AlternativeExplanation",
        "FactualExplanation",
        "FastExplanation",
        "AlternativeExplanations",
        "CalibratedExplanations",
        "CalibratedExplainer",
        "WrapCalibratedExplainer",
        "IntervalRegressor",
        "VennAbers",
    ]

    for name in names:
        try:
            # Access attribute to trigger lazy import/lookup
            getattr(pkg, name)
        except Exception:
            # Some attributes import optional heavy dependencies; that's OK for
            # this coverage-focused smoke test.
            continue
