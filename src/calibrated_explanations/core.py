"""Deprecated shim (Phase 1A). Use `from calibrated_explanations.core import ...`."""

from contextlib import suppress
from warnings import warn

with suppress(Exception):  # pragma: no cover - defensive
    from calibrated_explanations.core import *  # type: ignore # noqa: F401,F403 # pylint: disable=wildcard-import, unused-wildcard-import

warn(
    "The legacy module 'calibrated_explanations.core' is deprecated; import from the package form instead: "
    "`from calibrated_explanations.core import CalibratedExplainer`.",
    DeprecationWarning,
    stacklevel=2,
)
