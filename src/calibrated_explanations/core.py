"""Deprecated shim (Phase 1A). Use `from calibrated_explanations.core import ...`."""
from warnings import warn
from .core import *  # type: ignore # noqa: F401,F403 # pylint: disable=wildcard-import, import-self, unused-wildcard-import

warn(
    "Importing from calibrated_explanations.core (module) is deprecated; "
    "use the core package: `from calibrated_explanations.core import CalibratedExplainer`.",
    DeprecationWarning,
    stacklevel=2,
)
