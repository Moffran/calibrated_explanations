import importlib
import sys
import pytest

pytestmark = pytest.mark.integration


def test_import_smoke_modules():
    """Smoke test: import a set of public/impl modules to exercise module-level code.

    This is intentionally small and deterministic — it only imports modules and
    asserts they load. It helps increase coverage for lazy-loaded code paths
    without exercising heavy plotting backends.
    """
    modules = [
        "calibrated_explanations.core.calibrated_explainer",
        "calibrated_explanations.core.explain._computation",
        "calibrated_explanations.core.explain._feature_filter",
        "calibrated_explanations.plugins.builtins",
        "calibrated_explanations.explanations.explanation",
        "calibrated_explanations.viz.builders",
        "calibrated_explanations.plugins.registry",
    ]

    for name in modules:
        mod = importlib.import_module(name)
        assert mod is not None


def test_production_package_does_not_expose_tests_support():
    """Packaging isolation: production import graph must not reach tests.support.

    Verifies that no production module re-exports or forwards symbols from
    ``tests.support``, satisfying Task 4 hardening outcome 3 (test helper
    modules are non-runtime artifacts unreachable through production namespaces).
    """

    importlib.import_module("calibrated_explanations.plugins.registry")
    ce_modules = [name for name in sys.modules if name.startswith("calibrated_explanations")]
    for mod_name in ce_modules:
        mod = sys.modules[mod_name]
        for attr in vars(mod).values():
            mod_of_attr = getattr(attr, "__module__", None) or ""
            assert not mod_of_attr.startswith("tests.support"), (
                f"Production module {mod_name!r} exposes symbol from "
                f"tests.support via attribute with __module__={mod_of_attr!r}"
            )
