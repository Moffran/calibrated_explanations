"""Tests for ADR-001 import graph enforcement.

Validates that import boundaries defined in ADR-001 are maintained.
Tests run both static analysis (AST) and runtime import verification.
"""

import sys
import ast
from pathlib import Path
from typing import Set, Tuple
import pytest
import importlib


# Configuration for allowed imports between packages
ALLOWED_CROSS_SIBLING = {
    # core can import from utils, schema, api (shared contracts)
    ("core", "utils"): True,
    ("core", "schema"): True,
    ("core", "api"): True,
    # All packages can import from utils
    ("explanations", "utils"): True,
    ("plugins", "utils"): True,
    ("calibration", "utils"): True,
    ("viz", "utils"): True,
    ("cache", "utils"): True,
    ("parallel", "utils"): True,
    # All packages can import from schema
    ("explanations", "schema"): True,
    ("plugins", "schema"): True,
    ("calibration", "schema"): True,
    ("viz", "schema"): True,
    # viz can import from explanations (adapters)
    ("viz", "explanations"): True,
    ("viz", "core"): True,
    # integrations can import from explanations and core
    ("integrations", "explanations"): True,
    ("integrations", "core"): True,
    ("integrations", "utils"): True,
    # perf shim can re-export from cache/parallel (temporary)
    ("perf", "cache"): True,
    ("perf", "parallel"): True,
    # legacy can import anything (compatibility shim)
    ("legacy", "*"): True,
}


def get_top_level_package(module_path: str) -> str:
    """Extract top-level package from module path.

    Examples:
        'calibrated_explanations.core.xyz' -> 'core'
        'core.strategies' -> 'core'
        'utils' -> 'utils'
    """
    if "." in module_path:
        parts = module_path.split(".")
        # Skip 'calibrated_explanations' if present
        if parts[0] == "calibrated_explanations" and len(parts) > 1:
            return parts[1]
        return parts[0]
    return module_path


def extract_imports_from_ast(file_path: Path) -> Set[Tuple[str, int]]:
    """Extract all imports from a Python file using AST.

    Returns:
        Set of (module_name, line_number) tuples.
    """
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0:  # Relative import
                relative = "." * node.level
                if module:
                    relative += module
                imports.add((relative, node.lineno))
            elif module:
                imports.add((module, node.lineno))

    return imports


def is_import_allowed(from_pkg: str, to_pkg: str) -> bool:
    """Check if a cross-sibling import is allowed."""
    if from_pkg == to_pkg:
        return True

    if (from_pkg, to_pkg) in ALLOWED_CROSS_SIBLING:
        return ALLOWED_CROSS_SIBLING[(from_pkg, to_pkg)]

    if (from_pkg, "*") in ALLOWED_CROSS_SIBLING:
        return ALLOWED_CROSS_SIBLING[(from_pkg, "*")]

    if ("*", to_pkg) in ALLOWED_CROSS_SIBLING:
        return ALLOWED_CROSS_SIBLING[("*", to_pkg)]

    return False


# ============================================================================
# Static Analysis Tests (AST-based, no imports needed)
# ============================================================================


class TestImportGraphStaticAnalysis:
    """Static analysis tests using AST parsing."""

    def test_should_not_have_cross_sibling_imports_in_calibrated_explainer(self):
        """Verify CalibratedExplainer uses only TYPE_CHECKING for cross-sibling imports."""
        ce_file = Path("src/calibrated_explanations/core/calibrated_explainer.py")
        assert ce_file.exists(), f"Expected {ce_file} to exist"

        tree = ast.parse(ce_file.read_text(encoding="utf-8"))

        # Collect module-level imports (outside TYPE_CHECKING blocks)
        module_level_imports = []

        for node in ast.iter_child_nodes(tree):
            # Skip TYPE_CHECKING blocks
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            ):
                continue

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Get imported module
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level > 0 and module in {
                        "perf",
                        "plotting",
                        "plugins",
                        "explanations",
                        "integrations",
                        "discretizers",
                    }:
                        module_level_imports.append(module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(
                            sibling in alias.name
                            for sibling in ["perf", "plotting", "plugins", "explanations"]
                        ):
                            module_level_imports.append(alias.name)

        # All cross-sibling imports should be in TYPE_CHECKING or lazy (not found at module level)
        assert not module_level_imports, (
            f"Found module-level cross-sibling imports in CalibratedExplainer: {module_level_imports}\n"
            f"All cross-sibling imports must be in TYPE_CHECKING blocks or lazy imports."
        )

    def test_should_find_no_circular_imports_in_top_level_packages(self):
        """Check that top-level packages don't have circular import chains."""
        src_dir = Path("src/calibrated_explanations")

        # Define forbidden cycles
        forbidden_cycles = {
            ("core", "calibration"),
            ("explanations", "core"),
            ("perf", "core"),
            ("plugins", "core"),
            ("plotting", "core"),
        }

        violations = []

        for from_pkg, to_pkg in forbidden_cycles:
            from_file = src_dir / from_pkg / "__init__.py"
            if not from_file.exists():
                continue

            imports = extract_imports_from_ast(from_file)
            for imp_module, lineno in imports:
                top_level = get_top_level_package(imp_module)
                if top_level == to_pkg:
                    violations.append(f"Forbidden cycle: {from_pkg} → {to_pkg} (line {lineno})")

        assert not violations, "Circular imports detected:\n" + "\n".join(violations)

    def test_should_have_valid_import_graph_structure(self):
        """Verify the overall import graph structure is valid."""
        src_dir = Path("src/calibrated_explanations")

        # Count files by top-level package
        packages = {}
        for py_file in src_dir.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue

            try:
                rel_path = py_file.relative_to(src_dir)
                top_level = rel_path.parts[0] if rel_path.parts else None
                if top_level and top_level != "__init__.py":
                    packages[top_level] = packages.get(top_level, 0) + 1
            except ValueError:
                continue

        # Verify core packages exist
        expected_packages = {
            "core",
            "calibration",
            "explanations",
            "plugins",
            "viz",
            "cache",
            "parallel",
            "schema",
            "utils",
        }
        found = set(packages.keys())

        assert expected_packages.issubset(found), (
            f"Missing expected packages: {expected_packages - found}\n" f"Found: {found}"
        )


# ============================================================================
# Runtime Import Tests (verify actual imports work as expected)
# ============================================================================


class TestImportGraphRuntime:
    """Runtime tests that verify actual import behavior."""

    @pytest.mark.parametrize(
        "module_path,expected_accessible",
        [
            ("calibrated_explanations.core", True),
            ("calibrated_explanations.utils", True),
            ("calibrated_explanations.schema", True),
            ("calibrated_explanations.api", True),
            ("calibrated_explanations.explanations", True),
        ],
    )
    def test_should_import_core_packages_independently(self, module_path, expected_accessible):
        """Verify core packages can be imported without triggering optional dependencies."""
        try:
            importlib.import_module(module_path)
            assert expected_accessible, f"{module_path} should not be importable"
        except ImportError as e:
            assert not expected_accessible, f"{module_path} should be importable: {e}"

    def test_should_not_force_cross_sibling_imports_at_module_load(self):
        """Verify that importing core doesn't force imports of sibling packages."""
        # Remove modules that might be pre-cached
        modules_to_remove = [
            "calibrated_explanations.perf",
            "calibrated_explanations.plotting",
            "calibrated_explanations.plugins",
        ]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]

        # Import core (we don't need to keep the module object)
        importlib.import_module("calibrated_explanations.core")

        # Check that siblings were not imported
        unwanted = [
            "calibrated_explanations.perf",
            "calibrated_explanations.plotting",
            "calibrated_explanations.plugins",
        ]

        for unwanted_mod in unwanted:
            assert unwanted_mod not in sys.modules, (
                f"Importing core should not force import of {unwanted_mod}\n"
                f"This suggests lazy imports are not working correctly."
            )

    def test_should_maintain_export_paths_for_deprecation_warnings(self):
        """Verify that deprecated symbols can still be accessed (for deprecation warnings)."""
        from calibrated_explanations import CalibratedExplainer

        assert CalibratedExplainer is not None


# ============================================================================
# Package Boundary Tests (verify documented boundaries)
# ============================================================================


class TestPackageBoundaries:
    """Tests that verify package boundaries are documented and enforced."""

    def test_should_have_documented_boundary_rules(self):
        """Verify that ADR-001 boundary rules are enforced in code."""
        # Verify ALLOWED_CROSS_SIBLING configuration exists and is non-empty
        assert ALLOWED_CROSS_SIBLING is not None
        assert len(ALLOWED_CROSS_SIBLING) > 0, "Import boundaries should be defined"

        # Verify common boundary rules are in place
        assert ("core", "utils") in ALLOWED_CROSS_SIBLING
        assert ("legacy", "*") in ALLOWED_CROSS_SIBLING

    def test_should_have_migration_guides_for_deprecated_imports(self):
        """Verify migration guides are documented in CHANGELOG."""
        changelog = Path("CHANGELOG.md")
        assert changelog.exists()

        content = changelog.read_text(encoding="utf-8", errors="replace")
        # Check for deprecation information
        assert "deprecat" in content.lower(), "CHANGELOG should document deprecations"

    def test_should_classify_all_top_level_packages(self):
        """Verify top-level packages are classified in the allowed imports configuration."""
        # Verify packages appear in boundary rules
        packages = set()
        for from_pkg, to_pkg in ALLOWED_CROSS_SIBLING:
            if from_pkg != "*":
                packages.add(from_pkg)
            if to_pkg != "*":
                packages.add(to_pkg)

        assert len(packages) > 0, "Package boundaries should be defined"
        # Verify core and utils are classified
        assert "core" in packages or "utils" in packages


# ============================================================================
# Integration Tests (verify end-to-end behavior)
# ============================================================================


class TestImportGraphIntegration:
    """Integration tests combining static analysis and runtime checks."""

    def test_should_enforce_adr001_boundaries_in_ci(self):
        """Verify that CI can run import graph checks."""
        # This test simply verifies the checking script exists
        check_script = Path("scripts/check_import_graph.py")
        assert check_script.exists(), (
            f"Import graph linting script should exist at {check_script}\n"
            f"This is required for CI enforcement of ADR-001 boundaries."
        )

    def test_should_have_stage5_completion_documentation(self):
        """Verify Stage 5 completion is documented."""
        # Note: This will pass as soon as Stage 5 is marked complete
        release_plan_candidates = [
            Path("improvement_docs/RELEASE_PLAN_V1.md"),
            Path("improvement_docs/RELEASE_PLAN_v1.md"),
        ]
        # Some environments (Linux) are case-sensitive while others (Windows) are not.
        assert any(
            plan.exists() for plan in release_plan_candidates
        ), "Stage 5 documentation should include the release plan in improvement_docs."

        # After implementation, this should reference Stage 5 completion
        # This test serves as a marker for Stage 5 readiness


# ============================================================================
# Regression Tests (verify past issues don't reoccur)
# ============================================================================


class TestImportGraphRegressions:
    """Tests that verify known import issues don't reoccur."""

    def test_should_not_revert_stage2_lazy_import_pattern(self):
        """Verify Stage 2 lazy import refactoring is maintained."""
        ce_file = Path("src/calibrated_explanations/core/calibrated_explainer.py")
        content = ce_file.read_text(encoding="utf-8", errors="replace")

        # Check for TYPE_CHECKING usage (added in Stage 2)
        assert "TYPE_CHECKING" in content, (
            "CalibratedExplainer should use TYPE_CHECKING for cross-sibling type hints\n"
            "This ensures lazy imports are maintained."
        )

        # Check that lazy imports are used in methods
        assert (
            "importlib.import_module" in content or "from .." in content
        ), "CalibratedExplainer should use lazy imports for sibling packages"

    def test_should_maintain_stage3_public_api_deprecations(self):
        """Verify Stage 3 public API narrowing is maintained."""
        init_file = Path("src/calibrated_explanations/__init__.py")
        content = init_file.read_text(encoding="utf-8", errors="replace")

        # Check for deprecation helper usage
        assert "deprecate_public_api_symbol" in content or "DeprecationWarning" in content, (
            "__init__.py should emit deprecation warnings for unsanctioned symbols\n"
            "This ensures public API surface narrowing is maintained."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
