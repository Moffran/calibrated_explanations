import importlib
import sys

import pytest


def test_should_raise_import_error_when_perf_cache_shim_imported() -> None:
    for name in list(sys.modules):
        if name == "calibrated_explanations.perf.cache" or name.startswith(
            "calibrated_explanations.perf.cache."
        ):
            sys.modules.pop(name)

    with pytest.raises(ImportError):
        importlib.import_module("calibrated_explanations.perf.cache")
