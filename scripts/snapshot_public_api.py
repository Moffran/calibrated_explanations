# pylint: disable=line-too-long, missing-function-docstring, too-many-locals, import-outside-toplevel, invalid-name, no-member, unused-import
"""Snapshot the current public API surface of calibrated_explanations.

Creates a timestamped text file in benchmarks/ listing exported symbols from the
package root plus selected key modules. Intended for Phase 1A mechanical refactor
regression checking.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure local src/ on sys.path for in-place execution
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PACKAGE = "calibrated_explanations"
OUT_DIR = ROOT / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_module_public(mod):
    exported = set()
    if hasattr(mod, "__all__") and mod.__all__:
        exported.update(mod.__all__)
    else:
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            # only include callables, classes, modules, simple constants
            if (
                inspect.isfunction(obj)
                or inspect.isclass(obj)
                or inspect.ismodule(obj)
                or isinstance(obj, (int, float, str, tuple))
            ):
                exported.add(name)
    return sorted(exported)


def main():
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    lines = []

    root_pkg = importlib.import_module(PACKAGE)
    lines.append(f"# Public API snapshot for {PACKAGE} @ {ts} UTC")
    lines.append("# Root package exports:")
    for name in list_module_public(root_pkg):
        lines.append(name)

    # Optionally traverse immediate submodules (non-recursive) to capture their __all__
    lines.append("\n# Selected submodule exports (__all__ if defined):")
    for m in pkgutil.iter_modules(root_pkg.__path__):
        full_name = f"{PACKAGE}.{m.name}"
        try:
            sub = importlib.import_module(full_name)
        except ImportError:
            continue
        if hasattr(sub, "__all__") and sub.__all__:
            lines.append(f"[{full_name}] -> __all__:")
            for name in sorted(sub.__all__):
                lines.append(f"  {name}")

    out_file = OUT_DIR / f"api_public_{ts}.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote API snapshot to {out_file}")


if __name__ == "__main__":  # pragma: no cover
    main()
