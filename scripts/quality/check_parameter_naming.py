"""Parameter naming CI guard (ADR-011 / v0.11.0 removed aliases).

Walks all .py files under --root, finds every public FunctionDef/AsyncFunctionDef
(name does not start with '_'), and flags any argument whose name appears in
BANNED_PUBLIC_PARAM_NAMES.

BANNED_PUBLIC_PARAM_NAMES mirrors REMOVED_ALIAS_MAP.keys() in api/params.py exactly.
Add new removed aliases to REMOVED_ALIAS_MAP first; this script reads that source
automatically so no change here is needed.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Derive the banned set from the authoritative REMOVED_ALIAS_MAP in api/params.py.
# ---------------------------------------------------------------------------
_PARAMS_MODULE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "calibrated_explanations"
    / "api"
    / "params.py"
)


def _load_banned_names() -> frozenset[str]:
    """Extract REMOVED_ALIAS_MAP.keys() by AST-parsing api/params.py."""
    tree = ast.parse(_PARAMS_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        # REMOVED_ALIAS_MAP is annotated (ast.AnnAssign): dict[str, str] = {...}
        if isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
            if (
                isinstance(target, ast.Name)
                and target.id == "REMOVED_ALIAS_MAP"
                and isinstance(value, ast.Dict)
            ):
                return frozenset(
                    k.value
                    for k in value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                )
        # Fallback: plain assignment (ast.Assign) without annotation
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "REMOVED_ALIAS_MAP"
            and isinstance(node.value, ast.Dict)
        ):
            return frozenset(
                k.value
                for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            )
    return frozenset()


BANNED_PUBLIC_PARAM_NAMES: frozenset[str] = _load_banned_names()

# Local variable names banned in the guarded/significance code paths (core/explain/).
# These represent the statistical significance threshold; the canonical names are
# `significance` (internal) and `GuardedOptions.confidence` (public API).
# `alpha`/`_alpha` must never be used as a local variable name for this concept.
# Visualisation-related `alpha` (opacity) is confined to `viz/` and is exempt.
_BANNED_LOCAL_VAR_NAMES: frozenset[str] = frozenset({"alpha", "_alpha", "alphas", "_alphas"})
_BANNED_LOCAL_VAR_PATHS: tuple[str, ...] = ("core/explain",)


def _check_file(path: Path) -> list[str]:
    """Return violation strings for one file."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    violations: list[str] = []

    # Check local variable names in guarded/significance paths.
    path_str = path.as_posix()
    if any(segment in path_str for segment in _BANNED_LOCAL_VAR_PATHS):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in _BANNED_LOCAL_VAR_NAMES:
                        violations.append(
                            f"{path}:{target.lineno}: "
                            f"banned local variable name '{target.id}' "
                            "(use 'significance' or '_significance' instead)"
                        )
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in _BANNED_LOCAL_VAR_NAMES
            ):
                violations.append(
                    f"{path}:{node.target.lineno}: "
                    f"banned local variable name '{node.target.id}' "
                    "(use 'significance' or '_significance' instead)"
                )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        all_args = (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
            + ([node.args.vararg] if node.args.vararg else [])
            + ([node.args.kwarg] if node.args.kwarg else [])
        )
        for arg in all_args:
            if arg.arg in BANNED_PUBLIC_PARAM_NAMES:
                violations.append(
                    f"{path}:{arg.lineno}: "
                    f"{node.name}() uses removed alias '{arg.arg}' as a parameter name"
                )
    return violations


def check_root(root: Path) -> list[str]:
    """Walk root recursively and return all violations."""
    violations: list[str] = []
    for py_file in sorted(root.rglob("*.py")):
        violations.extend(_check_file(py_file))
    return violations


def main(argv: list[str] | None = None) -> int:
    """Run the parameter naming check and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="src/calibrated_explanations",
        help="Root directory to scan (default: src/calibrated_explanations).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any violations are found (CI mode).",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: --root '{root}' is not a directory.", file=sys.stderr)
        return 2

    violations = check_root(root)
    for v in violations:
        print(v)

    if violations:
        count = len(violations)
        print(
            f"\nParameter naming: {count} violation(s) found. "
            "Add the removed alias to REMOVED_ALIAS_MAP in api/params.py "
            "and rename the parameter.",
            file=sys.stderr,
        )
        return 1 if args.check else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
