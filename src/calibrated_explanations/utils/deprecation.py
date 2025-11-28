"""Central deprecation helper for ADR-011 migration gates.

This module provides structured deprecation warnings that are:
- Consistent across all deprecation sites
- Testable and mockable
- Tagged with removal version and alternative imports
- Suitable for CI/CD enforcement
"""

import warnings
from typing import Optional


def deprecate_public_api_symbol(
    symbol_name: str,
    current_import: str,
    recommended_import: str,
    removal_version: str = "v0.11.0",
    extra_context: Optional[str] = None,
) -> None:
    """Emit structured deprecation warning for top-level API symbols.
    
    This function centralizes deprecation messaging for unsanctioned exports
    from calibrated_explanations.__init__, following ADR-001 Stage 3 and 
    ADR-011 deprecation policy.
    
    Args:
        symbol_name: Name of the symbol being accessed (e.g., "CalibratedExplanations")
        current_import: Current (deprecated) import path 
                        (e.g., "from calibrated_explanations import CalibratedExplanations")
        recommended_import: Recommended new import path
                           (e.g., "from calibrated_explanations.explanations import CalibratedExplanations")
        removal_version: Version in which the symbol will be removed from __init__.py (default: v0.11.0)
        extra_context: Optional additional migration guidance or explanation
        
    Examples:
        >>> deprecate_public_api_symbol(
        ...     "CalibratedExplanations",
        ...     "from calibrated_explanations import CalibratedExplanations",
        ...     "from calibrated_explanations.explanations import CalibratedExplanations",
        ...     extra_context="Explanation dataclasses are domain objects; import from the submodule.",
        ... )
    """
    message = (
        f"\n{symbol_name!r} imported from top level is deprecated and will be removed in {removal_version}.\n"
        f"  ❌ DEPRECATED: {current_import}\n"
        f"  ✓ RECOMMENDED: {recommended_import}\n"
    )
    
    if extra_context:
        message += f"\n  Details: {extra_context}\n"
    
    message += f"\nSee https://calibrated-explanations.readthedocs.io/en/latest/migration/api_surface_narrowing.html for migration guide.\n"
    
    warnings.warn(
        message.rstrip(),
        category=DeprecationWarning,
        stacklevel=3,
    )
