"""
Calibrated Explanations (calibrated_explanations).

is a Python package for explaining black-box models.

It is based on the paper "Calibrated Explanations: with Uncertainty Information and Counterfactuals"
by Helena Löfström et al.
"""

import importlib
import logging as _logging
from contextlib import suppress

# Expose viz namespace lazily via __getattr__ (avoid importing heavy backends eagerly)
# Note: avoid eager imports of explanation, viz and discretizer modules here.
# Those modules import heavy dependencies (numpy, pandas, plotting backends)
# and should be loaded lazily via __getattr__ below. Importing them at
# package import time increases startup cost significantly.

# Provide a default no-op handler to avoid "No handler" warnings for library users.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__version__ = "v0.10.0-dev"

# Note: core submodules are intentionally not imported here to avoid importing
# large backends and to make deprecation transitions explicit. We still expose
# the public symbols lazily so `from calibrated_explanations import CalibratedExplainer`
# works without triggering an eager import of `calibrated_explanations.core`.
__all__ = [
    "CalibratedExplainer",
    "WrapCalibratedExplainer",
    "transform_to_numeric",
]

# Emit structured deprecation warnings for the documented, unsanctioned
# top-level symbols. Tests in the suite expect these deprecations to be
# available when symbols are resolved. Importing the small helper here
# keeps messages consistent across the codebase.
with suppress(Exception):
    from .utils import deprecate_public_api_symbol  # type: ignore

    # Emit deprecations for documented unsanctioned exports. These are
    # intentionally informational and follow ADR-011 guidance.
    with suppress(Exception):
        deprecate_public_api_symbol(
            "viz",
            "from calibrated_explanations import viz",
            "from calibrated_explanations.viz import PlotSpec, plots, matplotlib_adapter",
            extra_context=(
                "The viz namespace is now a submodule. Import specific classes/functions "
                "from it directly."
            ),
        )
        for sym, cur, rec in [
            (
                "AlternativeExplanation",
                "from calibrated_explanations import AlternativeExplanation",
                "from calibrated_explanations.explanations.explanation import AlternativeExplanation",
            ),
            (
                "FactualExplanation",
                "from calibrated_explanations import FactualExplanation",
                "from calibrated_explanations.explanations.explanation import FactualExplanation",
            ),
            (
                "FastExplanation",
                "from calibrated_explanations import FastExplanation",
                "from calibrated_explanations.explanations.explanation import FastExplanation",
            ),
            (
                "AlternativeExplanations",
                "from calibrated_explanations import AlternativeExplanations",
                "from calibrated_explanations.explanations import AlternativeExplanations",
            ),
            (
                "CalibratedExplanations",
                "from calibrated_explanations import CalibratedExplanations",
                "from calibrated_explanations.explanations import CalibratedExplanations",
            ),
            (
                "BinaryEntropyDiscretizer",
                "from calibrated_explanations import BinaryEntropyDiscretizer",
                "from calibrated_explanations.utils import BinaryEntropyDiscretizer",
            ),
            (
                "BinaryRegressorDiscretizer",
                "from calibrated_explanations import BinaryRegressorDiscretizer",
                "from calibrated_explanations.utils import BinaryRegressorDiscretizer",
            ),
            (
                "EntropyDiscretizer",
                "from calibrated_explanations import EntropyDiscretizer",
                "from calibrated_explanations.utils import EntropyDiscretizer",
            ),
            (
                "RegressorDiscretizer",
                "from calibrated_explanations import RegressorDiscretizer",
                "from calibrated_explanations.utils import RegressorDiscretizer",
            ),
            (
                "IntervalRegressor",
                "from calibrated_explanations import IntervalRegressor",
                "from calibrated_explanations.calibration import IntervalRegressor",
            ),
            (
                "VennAbers",
                "from calibrated_explanations import VennAbers",
                "from calibrated_explanations.calibration import VennAbers",
            ),
            (
                "plotting",
                "from calibrated_explanations import plotting",
                "from calibrated_explanations.viz import PlotSpec, plots, matplotlib_adapter",
            ),
        ]:
            with suppress(Exception):
                deprecate_public_api_symbol(sym, cur, rec)


def __getattr__(name: str):
    """Lazy import for public symbols (sanctioned and deprecated per ADR-001 Stage 3).

    This avoids importing `calibrated_explanations.core` at package import time
    while preserving the public API surface for users and tests.

    Sanctioned symbols (no deprecation warning):
    - CalibratedExplainer, WrapCalibratedExplainer, transform_to_numeric

    Deprecated symbols (emit DeprecationWarning, to be removed in v0.11.0):
    - Explanation classes: AlternativeExplanation, FactualExplanation, etc.
    - Discretizers: BinaryEntropyDiscretizer, EntropyDiscretizer, etc.
    - Calibrators: IntervalRegressor, VennAbers
    - Visualization: viz namespace
    """
    # ===================================================================
    # SANCTIONED SYMBOLS (no deprecation warning)
    # ===================================================================

    if name == "CalibratedExplainer":
        from .core.calibrated_explainer import CalibratedExplainer

        globals()[name] = CalibratedExplainer
        return CalibratedExplainer

    if name == "WrapCalibratedExplainer":
        from .core.wrap_explainer import WrapCalibratedExplainer

        globals()[name] = WrapCalibratedExplainer
        return WrapCalibratedExplainer

    if name == "transform_to_numeric":
        module = importlib.import_module(f"{__name__}.utils")
        value = getattr(module, name)
        globals()[name] = value
        return value

    # ===================================================================
    # DEPRECATED SYMBOLS (emit DeprecationWarning per ADR-001 Stage 3)
    # ===================================================================

    from .utils import deprecate_public_api_symbol

    if name == "viz":
        deprecate_public_api_symbol(
            "viz",
            "from calibrated_explanations import viz",
            "from calibrated_explanations.viz import PlotSpec, plots, matplotlib_adapter",
            extra_context="The viz namespace is now a submodule. Import specific classes/functions from it directly.",
        )
        module = importlib.import_module(f"{__name__}.viz")
        globals()[name] = module
        return module

    if name in {
        "AlternativeExplanation",
        "FactualExplanation",
        "FastExplanation",
    }:
        deprecate_public_api_symbol(
            name,
            f"from calibrated_explanations import {name}",
            f"from calibrated_explanations.explanations.explanation import {name}",
            extra_context="Explanation domain classes should be imported from the explanations submodule.",
        )
        module = importlib.import_module(f"{__name__}.explanations.explanation")
        value = getattr(module, name)
        globals()[name] = value
        return value

    if name in {
        "AlternativeExplanations",
        "CalibratedExplanations",
    }:
        deprecate_public_api_symbol(
            name,
            f"from calibrated_explanations import {name}",
            f"from calibrated_explanations.explanations import {name}",
            extra_context="Explanation collections should be imported from the explanations submodule.",
        )
        module = importlib.import_module(f"{__name__}.explanations.explanations")
        value = getattr(module, name)
        globals()[name] = value
        return value

    if name in {
        "BinaryEntropyDiscretizer",
        "BinaryRegressorDiscretizer",
        "EntropyDiscretizer",
        "RegressorDiscretizer",
    }:
        deprecate_public_api_symbol(
            name,
            f"from calibrated_explanations import {name}",
            f"from calibrated_explanations.utils import {name}",
            extra_context="Discretizers are internal utilities; import from the discretizers submodule.",
        )
        module = importlib.import_module(f"{__name__}.utils")
        value = getattr(module, name)
        globals()[name] = value
        return value

    if name == "IntervalRegressor":
        deprecate_public_api_symbol(
            "IntervalRegressor",
            "from calibrated_explanations import IntervalRegressor",
            "from calibrated_explanations.calibration import IntervalRegressor",
            extra_context="Calibrators are domain components; import from the calibration submodule.",
        )
        from .calibration.interval_regressor import IntervalRegressor

        globals()[name] = IntervalRegressor
        return IntervalRegressor

    if name == "VennAbers":
        deprecate_public_api_symbol(
            "VennAbers",
            "from calibrated_explanations import VennAbers",
            "from calibrated_explanations.calibration import VennAbers",
            extra_context="Calibrators are domain components; import from the calibration submodule.",
        )
        from .calibration.venn_abers import VennAbers

        globals()[name] = VennAbers
        return VennAbers

    if name == "plotting":
        deprecate_public_api_symbol(
            "plotting",
            "from calibrated_explanations import plotting",
            "from calibrated_explanations.viz import PlotSpec, plots, matplotlib_adapter, ...",
            extra_context="The plotting module is deprecated. Use calibrated_explanations.viz submodule directly. Will be removed in v0.11.0.",
        )
        module = importlib.import_module(f"{__name__}.plotting")
        globals()[name] = module
        return module

    raise AttributeError(name)
