"""Shim for the calibrated_explanations.vision modality namespace (ADR-033)."""

from __future__ import annotations

from .utils.exceptions import MissingExtensionError

try:
    from ce_vision import *  # noqa: F401,F403
    from ce_vision import __all__  # noqa: F401
except ImportError:
    raise MissingExtensionError(
        "calibrated_explanations.vision requires the 'ce-vision' package. "
        "Install with: pip install calibrated_explanations[vision]"
    ) from None
