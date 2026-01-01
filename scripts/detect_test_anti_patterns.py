"""Compatibility wrapper for the anti-pattern detector script."""

from __future__ import annotations

import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path


def _load_detector_module():
    script_path = Path(__file__).parent / "anti-pattern-analysis" / "detect_test_anti_patterns.py"
    spec = spec_from_loader("detect_test_anti_patterns", SourceFileLoader("detect_test_anti_patterns", str(script_path)))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load anti-pattern detector from {script_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    detector = _load_detector_module()
    return detector.main()


if __name__ == "__main__":
    raise SystemExit(main())
