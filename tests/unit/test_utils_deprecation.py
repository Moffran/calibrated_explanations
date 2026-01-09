from __future__ import annotations

import pytest

from calibrated_explanations.utils import deprecation


def test_deprecate_public_api_symbol_emits_warning():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        deprecation.deprecate_public_api_symbol(
            "MySymbol",
            "from calibrated_explanations import MySymbol",
            "from calibrated_explanations.explanations import MySymbol",
        )


def test_deprecate_public_api_symbol_includes_extra_context():
    with pytest.warns(DeprecationWarning, match="Details"):
        deprecation.deprecate_public_api_symbol(
            "OtherSymbol",
            "from calibrated_explanations import OtherSymbol",
            "from calibrated_explanations.explanations import OtherSymbol",
            extra_context="See the migration guide",
        )
