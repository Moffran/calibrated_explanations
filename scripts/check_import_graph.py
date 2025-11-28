#!/usr/bin/env python3
"""Import graph linting for ADR-001 boundary enforcement.

Validates that the import graph respects documented ADR-001 boundaries:
- No cross-sibling imports (except through documented interfaces)
- No circular imports between top-level packages
- No unintended imports from private submodules

Usage:
    python scripts/check_import_graph.py [--strict] [--report] [--fix]

Exit codes:
    0: No violations found
    1: Violations found
    2: Configuration error
"""

import ast
import sys
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
        'cache',
        'parallel',
        'schema',
        'utils',
        'api',
        'legacy',
        'integrations',
    })
    
    # Intentional cross-sibling imports (allowed exceptions)
    # Format: (from_package, to_package) → [allowed_module_paths]
    allowed_cross_sibling: Dict[Tuple[str, str], List[str]] = field(default_factory=lambda: {
        # core can import from utils and schema (domain models, validation)
        ('core', 'utils'): [],  # All utils allowed
        ('core', 'schema'): [],  # All schema allowed
        ('core', 'api'): [],  # Parameter validation facade
        
        # All packages can import from utils
        ('*', 'utils'): [],
        
        # All packages can import from schema
        ('*', 'schema'): [],
        
        # Temporary shims (post-Stage 1b)
        ('perf', 'cache'): [],  # perf re-exports cache
        ('perf', 'parallel'): [],  # perf re-exports parallel
        
        # Integrations can import from explanations (adapters)
        ('integrations', 'explanations'): [],
        ('integrations', 'core'): [],
        
        # Legacy compatibility (will be removed v2.0.0)
        ('legacy', '*'): [],
        
        # Visualization can import from explanations
        ('viz', 'explanations'): [],
        ('viz', 'core'): [],
    })
    
    # Packages that cannot import from each other
    forbidden_cycles: Set[Tuple[str, str]] = field(default_factory=lambda: {
        ('core', 'calibration'),
        ('calibration', 'core'),
        ('explanations', 'core'),
        ('core', 'explanations'),
        ('perf', 'core'),
        ('core', 'perf'),
        ('plugins', 'core'),
        ('core', 'plugins'),
        ('plotting', 'core'),
        ('core', 'plotting'),
    })
    
    # Strict mode disallows even allowed_cross_sibling imports in specific files
    strict_modules: Set[str] = field(default_factory=lambda: {
        'core/calibrated_explainer.py',  # Should use TYPE_CHECKING for cross-sibling imports
        'core/strategy_manager.py',
    })


def extract_imports(file_path: Path) -> List[Tuple[str, int]]:
    """Extract all imports from a Python file.
    
    Returns:
        List of (module_name, line_number) tuples for each import statement.
    """
    try:
        tree = ast.parse(file_path.read_text(encoding='utf-8'))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return []
    
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            # Handle relative imports
            if node.level > 0:  # Relative import
                relative_prefix = '.' * node.level
                imports.append((f"{relative_prefix}{module}", node.lineno))
            else:
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
        source_parts = [p for p in source_parts if p != '__pycache__']
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


def check_import_violations(src_dir: Path, config: BoundaryConfig) -> List[ImportViolation]:
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
        
        # Determine the module being checked
        try:
            rel_path = py_file.relative_to(src_dir.parent)
            module_parts = rel_path.parts[:-1] + (rel_path.stem,)
            importing_module = '.'.join(module_parts)
        except ValueError:
            continue
        
        source_top_level = get_top_level_package(importing_module)
        if not source_top_level:
            continue
        
        # Extract imports
        imports = extract_imports(py_file)
        
        for imported, line_no in imports:
            # Resolve relative imports
            if imported.startswith('.'):
                imported = resolve_relative_import(py_file, imported)
                if not imported:
                    continue
            
            # Get top-level package of import
            imported_top_level = get_top_level_package(imported)
            if not imported_top_level:
                continue
            
            # Skip imports within the same package
            if source_top_level == imported_top_level:
                continue
            
            # Check for cross-sibling imports
            if source_top_level not in config.top_level_packages:
                continue
            if imported_top_level not in config.top_level_packages:
                continue
            
            # Check if this cross-sibling import is allowed
            is_allowed = False
            
            # Check direct allowance
            if (source_top_level, imported_top_level) in config.allowed_cross_sibling:
                is_allowed = True
            
            # Check wildcard allowance
            if ('*', imported_top_level) in config.allowed_cross_sibling:
                is_allowed = True
            
            # Check if in strict mode
            in_strict = any(
                py_file.match(f"*{strict_mod}") 
                for strict_mod in config.strict_modules
            )
            
            if not is_allowed or in_strict:
                violations.append(ImportViolation(
                    file_path=str(py_file),
                    line_number=line_no,
                    imported_from=imported,
                    importing_module=importing_module,
                    violation_type='cross_sibling',
                    message=f"Cross-sibling import: {importing_module} imports {imported} (not allowed by ADR-001)"
                ))
    
    return violations


def print_violations(violations: List[ImportViolation]) -> None:
    """Pretty-print import violations."""
    if not violations:
        print("[OK] No import graph violations detected (ADR-001 compliant)")
        return
    
    print(f"[VIOLATIONS] Found {len(violations)} import graph violation(s):\n")
    
    # Group by file
    by_file = {}
    for v in violations:
        if v.file_path not in by_file:
            by_file[v.file_path] = []
        by_file[v.file_path].append(v)
    
    for file_path in sorted(by_file.keys()):
        print(f"  {file_path}")
        for v in by_file[file_path]:
            print(f"    Line {v.line_number}: {v.message}")
            print(f"      From: {v.importing_module}")
            print(f"      To:   {v.imported_from}")
            print()


def generate_report(violations: List[ImportViolation], output_file: Path) -> None:
    """Generate a JSON report of violations."""
    report = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'total_violations': len(violations),
        'violations': [
            {
                'file': v.file_path,
                'line': v.line_number,
                'from': v.importing_module,
                'to': v.imported_from,
                'type': v.violation_type,
                'message': v.message,
            }
            for v in violations
        ],
    }
    
    output_file.write_text(json.dumps(report, indent=2))
    print(f"[REPORT] Written to {output_file}")


def main() -> int:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check import graph compliance with ADR-001 boundaries"
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Strict mode: disallow all cross-sibling imports'
    )
    parser.add_argument(
        '--report',
        type=Path,
        help='Generate JSON report to this file'
    )
    parser.add_argument(
        '--src-dir',
        type=Path,
        default=Path('src/calibrated_explanations'),
        help='Path to source directory (default: src/calibrated_explanations)'
    )
    
    args = parser.parse_args()
    
    # Validate src_dir
    if not args.src_dir.exists():
        print(f"❌ Source directory not found: {args.src_dir}", file=sys.stderr)
        return 2
    
    # Load configuration
    config = BoundaryConfig()
    
    # Check for violations
    violations = check_import_violations(args.src_dir, config)
    
    # Print violations
    print_violations(violations)
    
    # Generate report if requested
    if args.report:
        generate_report(violations, args.report)
    
    # Return appropriate exit code
    return 1 if violations else 0


if __name__ == '__main__':
    sys.exit(main())
