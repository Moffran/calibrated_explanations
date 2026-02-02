"""Legacy compatibility shim for the reject package.

This module historically defined `RejectPolicy` inline. The implementation was
migrated to the package submodule `core.reject.policy` to support new
canonical names. To maintain backwards compatibility for imports of
``calibrated_explanations.core.reject``, re-export the public symbols from the
new module and emit a DeprecationWarning.
"""

from __future__ import annotations

import warnings

_emitted_warning = False
try:
    # Preferred: package-relative import when used as installed package
    from .reject.policy import RejectPolicy, is_policy_enabled  # type: ignore
    _emitted_warning = True
except Exception:
    # Fallback when executing the file directly (no package context), e.g.
    # tests that import the module from its file path. Try absolute import.
    try:
        from calibrated_explanations.core.reject.policy import (
            RejectPolicy,
            is_policy_enabled,
        )
    except Exception:
        # Let import errors propagate for unexpected situations
        raise

if _emitted_warning:
    warnings.warn(
        "calibrated_explanations.core.reject module is deprecated; import from "
        "calibrated_explanations.core.reject.policy instead.",
        DeprecationWarning,
        stacklevel=2,
    )

__all__ = ["RejectPolicy", "is_policy_enabled"]
