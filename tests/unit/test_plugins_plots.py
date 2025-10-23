import importlib
import sys


def test_plots_typealias_fallback(monkeypatch):
    module_name = "calibrated_explanations.plugins.plots"
    original_module = sys.modules.get(module_name)
    if original_module is not None:
        sys.modules.pop(module_name)

    original_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "typing":
            raise ImportError("TypeAlias unavailable")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    try:
        module = importlib.import_module(module_name)
        assert module.TypeAlias is object
    finally:
        monkeypatch.setattr(importlib, "import_module", original_import)
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
            importlib.reload(original_module)
        else:
            importlib.import_module(module_name)
