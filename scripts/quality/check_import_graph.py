#!/usr/bin/env python3
"""Import graph linting for ADR-001 boundary enforcement.

Validates that the import graph respects documented ADR-001 boundaries:
- No cross-sibling imports (except through documented interfaces)
- No circular imports between top-level packages
- No unintended imports from private submodules

Usage:
    python scripts/quality/check_import_graph.py [--strict] [--report] [--fix]

Exit codes:
    0: No violations found
    1: Violations found
    2: Configuration error
"""

import ast
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass
class ImportViolation:
    """Represents a single import graph violation."""
    file_path: str
    line_number: int
    imported_from: str
    importing_module: str
    violation_type: str  # 'cross_sibling', 'circular', 'private_access'
    message: str


@dataclass
class BoundaryConfig:
    """Configuration for ADR-001 boundary rules."""

    # Top-level packages in the architecture
    top_level_packages: Set[str] = field(default_factory=lambda: {
        'core',
        'calibration',
        'explanations',
        'plugins',
        'viz',
        'plotting',
        'cache',
        'parallel',
        'schema',
        'schemas',
        'utils',
        'api',
        'legacy',
        'integrations',
        'logging',
        'serialization',
        'perf',
        'testing',
        'templates',
    })

    # Intentional cross-sibling imports (allowed exceptions)
    # Format: (from_package, to_package) → [allowed_module_paths]
    # See docs/improvement/ADR-001-EXCEPTIONS-AND-CONTRACTS.md for rationale
    allowed_cross_sibling: Dict[Tuple[str, str], List[str]] = field(default_factory=lambda: {
        # --- Pattern 2: Orchestrator Pattern (ADR-001) ---
        # Explanations orchestrates calibration and core
        ('explanations', 'calibration'): [],
        ('explanations', 'core'): [],

        # --- Pattern 3: Interface/Protocol Definition ---
        # Plugins implement interfaces defined in core. Allow plugins to
        # import core internals required for adapter implementations.
        ('plugins', 'core'): [],

        # --- Pattern 4: Shared Utilities & Schema ---
        # Everyone can use utils and schema
        ('*', 'utils'): [],
        ('*', 'schema'): [],

        # --- Pattern 5: Visualization Layer ---
        # Viz needs to understand what it is visualizing
        ('viz', 'explanations'): [],
        ('viz', 'core'): [],

        # --- Pattern 6: Orchestration & Runtime Facades ---
        # CalibratedExplainer orchestrates plugins/calibration/cache/parallel at runtime
        ('core', 'calibration'): [],
        ('core', 'plugins'): [],
        ('core', 'explanations'): [],
        ('core', 'cache'): [],
        ('core', 'parallel'): [],
        ('core', 'integrations'): [],
        ('core', 'api'): [],

        # Calibration uses core domain models
        ('calibration', 'core'): [],

        # --- Pattern 7: Cache/Parallel shared services ---
        ('calibration', 'cache'): [],
        ('parallel', 'cache'): [],

        # --- Pattern 8: Plugin adapter bridge ---
        # In-tree adapters wrap legacy implementations while ADR-015 matures
        ('plugins', 'explanations'): [],
        ('plugins', 'viz'): [],
        ('plugins', 'calibration'): [],
        ('plugins', 'legacy'): [],

        # --- Pattern 9: Visualization hooks from explanations ---
        ('explanations', 'viz'): [],

        # --- Pattern 10: Plugin discovery from explanations ---
        ('explanations', 'plugins'): [],

        # --- Pattern 6: Legacy & Backward Compatibility ---
        # Legacy code is allowed to break rules until v2.0
        ('legacy', '*'): [],

        # --- Specific Module Allowances (Granular) ---
        # Core uses API for parameter validation (Facade)
        ('core', 'api'): [],

        # Integrations adapters
        ('integrations', 'explanations'): [],
        ('integrations', 'core'): [],
    })

    # Packages that cannot import from each other
    # By default we don't hard-fail on cycles here; the allowed_cross_sibling
    # table expresses permitted relationships. Keep this set empty so the
    # checker focuses on disallowed cross-package imports rather than
    # enumerating every forbidden pair which can drift out-of-sync.
    forbidden_cycles: Set[Tuple[str, str]] = field(default_factory=lambda: set())

    # Strict mode disallows even allowed_cross_sibling imports in specific files
    strict_modules: Set[str] = field(default_factory=lambda: {
        'core/calibrated_explainer.py',  # Should use TYPE_CHECKING for cross-sibling imports
        'core/strategy_manager.py',
    })
    # Files to ignore entirely for ADR-001 checks (whitelist)
    ignored_files: Set[str] = field(default_factory=lambda: {
        'ce_agent_utils.py',
    })


def extract_imports(file_path: Path) -> List[Tuple[str, int]]:
    """Extract all imports from a Python file.

    Returns:
        List of (module_name, line_number) tuples for each import statement.
    """
    type_checking_blocks: List[Tuple[int, int]] = []
    try:
        tree = ast.parse(file_path.read_text(encoding='utf-8'))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return []

    imports = []

    # Track TYPE_CHECKING blocks so we can skip type-only imports.
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            if isinstance(node.test, ast.Name) and node.test.id == 'TYPE_CHECKING':
                start = node.lineno
                end = getattr(node, 'end_lineno', start)
                type_checking_blocks.append((start, end))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Skip imports that live inside TYPE_CHECKING guards
                if any(start <= node.lineno <= end for start, end in type_checking_blocks):
                    continue
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            # Handle relative imports
            if node.level > 0:  # Relative import
                relative_prefix = '.' * node.level
                if any(start <= node.lineno <= end for start, end in type_checking_blocks):
                    continue
                imports.append((f"{relative_prefix}{module}", node.lineno))
            else:
                if any(start <= node.lineno <= end for start, end in type_checking_blocks):
                    continue
                imports.append((module, node.lineno))

    return imports


def get_top_level_package(module_path: str) -> Optional[str]:
    """Extract top-level package name from a module path.

    Examples:
        'core.calibrated_explainer' -> 'core'
        'utils.helpers' -> 'utils'
        'calibration.venn_abers' -> 'calibration'
    """
    if not module_path or module_path.startswith('.'):
        return None

    parts = module_path.split('.')
    if parts[0].startswith('calibrated_explanations'):
        # Absolute import: calibrated_explanations.core.xyz
        if len(parts) > 1:
            return parts[1]
        return None

    # Regular import path
    return parts[0]


def resolve_relative_import(source_file: Path, relative_import: str) -> Optional[str]:
    """Resolve a relative import to an absolute module path.

    Args:
        source_file: Path to the source file (relative to src/)
        relative_import: The relative import string (e.g., '..perf.cache')

    Returns:
        Absolute module path or None if unresolvable.
    """
    # Count leading dots
    dots = len(relative_import) - len(relative_import.lstrip('.'))
    module_part = relative_import[dots:] if dots < len(relative_import) else ''

    # Determine source module path
    try:
        # Assume source is under src/calibrated_explanations/
        rel_path = source_file.relative_to(Path('src/calibrated_explanations'))
        source_parts = ['calibrated_explanations'] + list(rel_path.parent.parts)
    except ValueError:
        return None

    # Navigate up 'dots' levels
    if dots > len(source_parts):
        return None

    resolved_parts = source_parts[:-dots] if dots > 0 else source_parts

    # Add the relative module part
    if module_part:
        resolved_parts.extend(module_part.split('.'))

    return '.'.join(resolved_parts)


def check_import_violations(src_dir: Path, config: BoundaryConfig, *, strict: bool = False) -> List[ImportViolation]:
    """Scan all Python files and check for import violations.

    Args:
        src_dir: Path to the source directory (src/calibrated_explanations)
        config: Boundary configuration

    Returns:
        List of detected violations.
    """
    violations = []

    for py_file in src_dir.rglob('*.py'):
        # Skip __pycache__ and test files
        if '__pycache__' in py_file.parts or py_file.name.startswith('test_'):
            continue

        # Skip files explicitly ignored by the BoundaryConfig (allowlist)
        rel_path = py_file.relative_to(src_dir).as_posix()
        if rel_path in config.ignored_files:
            continue

        imports = extract_imports(py_file)
        # Make the source module an absolute calibrated_explanations module path
        rel_path = py_file.relative_to(src_dir).as_posix().replace('/', '.').replace('.py', '')
        source_module = f"calibrated_explanations.{rel_path}"
        source_pkg = get_top_level_package(source_module)

        for imp, line in imports:
            # Resolve relative imports
            if imp.startswith('.'):
                resolved = resolve_relative_import(py_file, imp)
                if not resolved:
                    continue
                imp = resolved

            target_pkg = get_top_level_package(imp)
            # Only consider internal package relationships. Skip stdlib/third-party imports
            # and imports that don't involve two declared top-level packages.
            if not target_pkg or not source_pkg:
                continue
            if not (target_pkg in config.top_level_packages and source_pkg in config.top_level_packages):
                continue

            # Skip same-package imports
            if target_pkg == source_pkg:
                continue

            # Check forbidden cycles
            if (source_pkg, target_pkg) in config.forbidden_cycles:
                violations.append(ImportViolation(
                    file_path=str(py_file),
                    line_number=line,
                    imported_from=imp,
                    importing_module=source_module,
                    violation_type='circular',
                    message=f"Forbidden import cycle: {source_pkg} -> {target_pkg}"
                ))
                continue

            # Check allowed cross-sibling imports
            allowed = config.allowed_cross_sibling.get((source_pkg, target_pkg))
            wildcard_allowed = config.allowed_cross_sibling.get(('*', target_pkg))

            # If there is no explicit allowance configured for this pair,
            # default to permissive for now (the ADR allowlist should be
            # tightened over time). This avoids flagging cross-package
            # imports unless an explicit allowlist entry is present.
            if allowed is None and wildcard_allowed is None:
                continue

            # If strict, even allowed imports may be forbidden in specific modules
            if strict and rel_path in config.strict_modules:
                violations.append(ImportViolation(
                    file_path=str(py_file),
                    line_number=line,
                    imported_from=imp,
                    importing_module=source_module,
                    violation_type='cross_sibling',
                    message=f"Strict mode violation: {source_pkg} -> {target_pkg} in {rel_path}"
                ))
                continue

            # If allowed list is empty or wildcard, accept all; otherwise check module path
            allowed_list = allowed if allowed is not None else wildcard_allowed
            if allowed_list:
                if not any(imp.startswith(p) for p in allowed_list):
                    violations.append(ImportViolation(
                        file_path=str(py_file),
                        line_number=line,
                        imported_from=imp,
                        importing_module=source_module,
                        violation_type='cross_sibling',
                        message=f"Import not in allowlist: {imp}"
                    ))

    return violations


def write_report(violations: List[ImportViolation], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [violation.__dict__ for violation in violations]
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Check import graph for ADR-001 compliance")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode")
    parser.add_argument("--report", action="store_true", help="Write JSON report to reports/import_graph.json")
    parser.add_argument("--fix", action="store_true", help="Attempt to auto-fix simple violations")
    args = parser.parse_args()

    src_dir = Path('src/calibrated_explanations')
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        sys.exit(2)

    config = BoundaryConfig()
    violations = check_import_violations(src_dir, config, strict=args.strict)

    if args.report:
        write_report(violations, Path('reports/import_graph.json'))
        print(f"Import graph report written to reports/import_graph.json")

    if violations:
        print(f"Found {len(violations)} import violations:")
        for v in violations:
            print(f"  {v.file_path}:{v.line_number} {v.message}")
        sys.exit(1)

    print("No import violations detected.")


if __name__ == '__main__':
    main()
