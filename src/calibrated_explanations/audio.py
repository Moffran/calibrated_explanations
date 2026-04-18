"""Shim for the calibrated_explanations.audio modality namespace (ADR-033)."""

from __future__ import annotations

from .utils.exceptions import MissingExtensionError

try:
    from ce_audio import *  # noqa: F401,F403
    from ce_audio import __all__  # noqa: F401
except ImportError:
    raise MissingExtensionError(
        "calibrated_explanations.audio requires the 'ce-audio' package. "
        "Install with: pip install calibrated_explanations[audio]"
    ) from None
