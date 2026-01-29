# pylint: disable=line-too-long, missing-function-docstring, too-many-locals, import-outside-toplevel, invalid-name, no-member, unused-import
"""Snapshot the current public API surface of calibrated_explanations.

Creates a timestamped text file in tests/benchmarks/ listing exported symbols from the
package root plus selected key modules.
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
OUT_DIR = ROOT / "tests/benchmarks"
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

    # Note: We intentionally do NOT traverse submodules here.
    # The public API is defined solely by the root package exports.
    # Submodules like .core, .api, .calibration are internal implementation details
    # unless explicitly re-exported by the root __init__.py.
    
    out_file = OUT_DIR / f"api_public_{ts}.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote API snapshot to {out_file}")


if __name__ == "__main__":  # pragma: no cover
    main()
