"""Shared helpers for test-level anti-pattern detection."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Tuple, Set


def extract_private_symbols_from_ast(txt: str, path: Path) -> Set[str] | None:
    """Return the private attributes/getattr names identified via AST parsing."""
    syms: Set[str] = set()
    try:
        tree = ast.parse(txt, filename=str(path))
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_") and not attr.startswith("__"):
                syms.add(attr)
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name != "getattr":
                continue
            if len(node.args) >= 2:
                arg = node.args[1]
                val = None
                if isinstance(arg, ast.Constant) and isinstance(
                    arg.value, str
                ):  # pragma: no branch
                    val = arg.value
                elif isinstance(arg, ast.Str):  # pragma: no branch - Python <3.8 compatibility
                    val = arg.s  # type: ignore[attr-defined]
                if val and val.startswith("_") and not val.startswith("__"):
                    syms.add(val)
    return syms


def parse_version_token(s: str) -> Tuple[int, ...] | None:
    """Return a tuple of version components or None for invalid tokens."""
    if not isinstance(s, str):
        return None
    token = s.lstrip("vV")
    token = token.split("-", 1)[0]
    parts = token.split(".")
    try:
        return tuple(int(p) for p in parts)
    except Exception:
        return None
