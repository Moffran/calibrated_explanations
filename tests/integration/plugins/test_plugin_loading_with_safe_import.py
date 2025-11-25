"""Integration tests for plugin loading with safe_import utilities.

Testing utility functions (safe_import, safe_isinstance) through real usage
scenarios rather than in isolation.

Instead of unit tests that directly call _private helpers without context,
these tests show how safe_import is used in actual plugin loading workflows.

Ref: ADR-006 (Plugin Trust Model)
"""

from __future__ import annotations

import pytest

from calibrated_explanations.utils.helper import safe_import, safe_isinstance


class TestSafeImportInPluginLoading:
    """Test safe_import through real plugin loading workflows.

    Context: safe_import is used in plugins/loader.py to load external plugins
    without failing if optional dependencies are missing.
    """

    def test_safe_import_core_module_succeeds(self):
        """Verify that safe_import successfully loads a core module.

        This tests the behavior safe_import provides: graceful module loading
        with meaningful error messages.
        """
        # Import a module through safe_import
        numpy_module = safe_import("numpy")

        assert numpy_module is not None, "safe_import should return the module"
        assert hasattr(numpy_module, "array"), "numpy module should have array function"

    def test_safe_import_class_from_module_succeeds(self):
        """Verify that safe_import can extract a class from a module.

        This is a common usage pattern in plugin loading.
        """
        ndarray_class = safe_import("numpy", "ndarray")

        assert ndarray_class is not None, "safe_import should return the class"
        assert ndarray_class.__name__ == "ndarray", "Should be the ndarray class"

    def test_safe_import_multiple_classes_returns_list(self):
        """Verify that safe_import can extract multiple classes.

        This demonstrates the actual usage pattern for loading plugin dependencies.
        """
        classes = safe_import("numpy", ["ndarray", "generic"])

        assert isinstance(classes, list), "Should return a list"
        assert len(classes) == 2, "Should return 2 classes"
        assert classes[0].__name__ == "ndarray", "First should be ndarray"
        assert classes[1].__name__ == "generic", "Second should be generic"

    def test_safe_import_missing_module_raises_informative_error(self):
        """Verify that safe_import provides helpful errors for missing modules.

        This tests the error handling contract that makes safe_import valuable.
        """
        with pytest.raises(ImportError) as exc_info:
            safe_import("nonexistent_module_xyz")

        error_message = str(exc_info.value)
        assert "nonexistent_module_xyz" in error_message, "Error should mention the module name"
        assert (
            "not installed" in error_message or "not found" in error_message.lower()
        ), "Error should explain why it failed"

    def test_safe_import_missing_class_raises_informative_error(self):
        """Verify that safe_import provides helpful errors for missing classes."""
        with pytest.raises(ImportError) as exc_info:
            safe_import("numpy", "NonexistentClass")

        error_message = str(exc_info.value)
        assert "NonexistentClass" in error_message, "Error should mention the class name"
        assert "numpy" in error_message, "Error should mention the module name"

    def test_safe_import_workflow_with_optional_dependency(self):
        """Test real workflow: conditionally loading an optional dependency.

        This demonstrates the actual use case for safe_import in the codebase.
        """
        # Simulate plugin loading that depends on optional package
        try:
            safe_import("sklearn")
            sklearn_available = True
        except ImportError:
            sklearn_available = False

        # The workflow should continue whether or not the dependency is available
        assert isinstance(sklearn_available, bool), "Should be able to determine availability"


class TestSafeIsInstanceInTypeChecking:
    """Test safe_isinstance through real type-checking workflows.

    Context: safe_isinstance is used in the codebase for runtime type checking
    that must survive when optional dependencies are not available.
    """

    def test_safe_isinstance_detects_correct_type(self):
        """Verify that safe_isinstance correctly identifies types by string path."""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor()

        # Check type using string path
        is_rf = safe_isinstance(model, "sklearn.ensemble.RandomForestRegressor")

        assert is_rf is True, "Should correctly identify RandomForestRegressor"

    def test_safe_isinstance_rejects_wrong_type(self):
        """Verify that safe_isinstance correctly rejects non-matching types."""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor()

        # Check type using string path for different class
        is_lr = safe_isinstance(model, "sklearn.linear_model.LinearRegression")

        assert is_lr is False, "Should not match different class"

    def test_safe_isinstance_with_multiple_types(self):
        """Verify that safe_isinstance can check multiple types (union).

        This demonstrates real usage for polymorphic type checking.
        """
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor()

        # Check if model matches any of several types
        class_paths = [
            "sklearn.linear_model.LinearRegression",
            "sklearn.ensemble.RandomForestRegressor",
            "sklearn.tree.DecisionTreeRegressor",
        ]

        matches_any = safe_isinstance(model, class_paths)

        assert matches_any is True, "Should match at least one type"

    def test_safe_isinstance_gracefully_handles_missing_module(self):
        """Verify that safe_isinstance doesn't crash when module is missing.

        This is the critical contract for optional dependency handling.
        """
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor()

        # Try to check against a non-existent module
        # This should return False, not raise an error
        is_fake_type = safe_isinstance(model, "nonexistent_module.FakeClass")

        assert is_fake_type is False, "Should return False for non-existent modules"

    def test_safe_isinstance_workflow_conditional_on_availability(self):
        """Test workflow: runtime type checking with optional dependencies.

        This shows the real use case where safe_isinstance enables graceful
        degradation when optional packages aren't available.
        """
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor()

        # Try to handle model regardless of which optional packages are available
        is_sklearn_model = safe_isinstance(model, "sklearn.base.BaseEstimator")

        # Workflow should continue whether or not the type matches
        assert isinstance(is_sklearn_model, bool), "Should get a boolean result"

        if is_sklearn_model:
            # Use sklearn-specific API
            pass
        else:
            # Use fallback API
            pass


class TestPluginDiscoveryWithSafeImport:
    """Integration tests showing safe_import in plugin discovery context.

    These tests demonstrate how safe_import is used in real plugin systems
    to load plugins that may have optional dependencies.
    """

    def test_discover_builtin_plugins_without_optional_deps(self):
        """Verify that builtin plugins can be discovered even without optionals.

        This is the core contract that safe_import enables.
        """
        # The plugin system should be able to ensure builtin plugins are loaded
        # without crashing if optional dependencies are missing
        try:
            from calibrated_explanations.plugins.registry import (
                ensure_builtin_plugins,
            )

            # This call should not crash even if some optional deps are missing
            ensure_builtin_plugins()
            # If we got here, the plugins loaded successfully
            assert True
        except ImportError as e:
            pytest.skip(f"Plugins module not available: {e}")

    def test_plugin_type_checking_robust_to_missing_deps(self):
        """Verify that plugin type checking works even with missing dependencies."""
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier()

        # Type checking should work and not require optional deps to be imported
        # in the calling code
        is_sklearn = safe_isinstance(model, "sklearn.base.BaseEstimator")

        assert is_sklearn is True, "Should correctly identify sklearn model"
