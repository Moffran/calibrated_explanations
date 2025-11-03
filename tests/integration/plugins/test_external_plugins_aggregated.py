"""Integration test for the aggregated external-plugins installer.

This test imports the `external_plugins.fast_explanations` bundle and calls
its `register()` helper twice to ensure the registration path is available and
idempotent.
"""

import importlib


def test_external_plugins_bundle_registers_idempotent():
    """Import the external bundle and ensure register() is callable and
    idempotent.
    """

    fast = importlib.import_module("external_plugins.fast_explanations")
    register = getattr(fast, "register", None)
    assert register is not None and callable(register)

    # Calling multiple times should not raise.
    register()
    register()
