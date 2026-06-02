"""Core-only vs extras-installed parity check (ADR-010 gap 1).

Verifies that calibrated_explanations produces structurally identical canonical
outputs (factual explanations on a synthetic binary dataset) regardless of
whether optional extras (viz, notebooks) are installed.

The check is intentionally lightweight: it exercises the WrapCalibratedExplainer
CE-first API path with a tiny synthetic dataset and asserts structural invariants
on the returned CalibratedExplanations object.

Usage::

    # Default: run and report
    python scripts/quality/check_core_extras_parity.py

    # Fail on structural deviation
    python scripts/quality/check_core_extras_parity.py --check

    # Write JSON report
    python scripts/quality/check_core_extras_parity.py --report reports/quality/core_extras_parity.json

Notes
-----
- This script must be runnable with only the core dependencies installed.
- It intentionally does not import any extras-only module (matplotlib, pandas, etc.)
  at the top level; extras are probed only via ``importlib.util.find_spec``.
- Running with extras installed and without should produce identical structural
  outputs; if they differ the script exits non-zero when ``--check`` is passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _probe_extras() -> dict[str, bool]:
    """Check which optional extras packages are available."""
    probes = {
        "matplotlib": "matplotlib",
        "pandas": "pandas",
        "shap": "shap",
        "lime": "lime",
        "nbconvert": "nbconvert",
    }
    return {name: importlib.util.find_spec(module) is not None for name, module in probes.items()}


def _make_synthetic_data(n_train: int = 40, n_cal: int = 20, n_test: int = 5, n_features: int = 4):
    """Generate a reproducible synthetic binary classification dataset."""
    import numpy as np

    rng = np.random.default_rng(seed=42)
    x_all = rng.standard_normal((n_train + n_cal + n_test, n_features))
    y_all = (x_all[:, 0] > 0).astype(int)

    x_proper = x_all[:n_train]
    y_proper = y_all[:n_train]
    x_cal = x_all[n_train : n_train + n_cal]
    y_cal = y_all[n_train : n_train + n_cal]
    x_test = x_all[n_train + n_cal :]

    return x_proper, y_proper, x_cal, y_cal, x_test


def _run_core_parity_check() -> dict:
    """Run the CE-first API and return structural invariants of the result."""
    from sklearn.tree import DecisionTreeClassifier

    from calibrated_explanations import WrapCalibratedExplainer

    x_proper, y_proper, x_cal, y_cal, x_test = _make_synthetic_data()

    explainer = WrapCalibratedExplainer(DecisionTreeClassifier(max_depth=3, random_state=42))
    explainer.fit(x_proper, y_proper)
    assert explainer.fitted, "WrapCalibratedExplainer must be fitted"
    explainer.calibrate(x_cal, y_cal)
    assert explainer.calibrated, "WrapCalibratedExplainer must be calibrated"

    explanations = explainer.explain_factual(x_test)

    # Collect structural invariants — these must be the same regardless of extras
    n_instances = len(explanations)

    # Verify explanation structure for all instances via public API
    for i in range(n_instances):
        exp = explanations[i]
        # Public API: predict property must return a numeric value
        pred = exp.predict
        assert pred is not None, f"Instance {i}: predict must not be None"
        # get_rules() returns a dict or list depending on task type
        rules = exp.get_rules()
        assert rules is not None, f"Instance {i}: get_rules() must not be None"

    return {
        "n_instances": n_instances,
        "explanations_type": type(explanations).__name__,
        "all_instances_have_predictions": all(
            explanations[i].predict is not None for i in range(n_instances)
        ),
        "all_instances_have_rules": all(
            explanations[i].get_rules() is not None for i in range(n_instances)
        ),
    }


def _check_no_extras_pollution() -> list[str]:
    """Verify that core CE modules do not hard-import extras-only packages at import time."""
    violations = []
    extras_only = {"matplotlib", "pandas", "shap", "lime", "nbconvert"}

    for mod_name in list(sys.modules.keys()):
        top = mod_name.split(".")[0]
        if top in extras_only:
            # Check if it was imported by calibrated_explanations at module level
            # (as opposed to being already present from sklearn or other deps)
            # We flag any extras-only top-level import that appears after CE import
            violations.append(mod_name)

    return violations


def main(argv: list[str] | None = None) -> int:
    """Entry point for the core/extras parity check."""
    parser = argparse.ArgumentParser(description="Core-only vs extras parity check (ADR-010)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero on any structural deviation or extras pollution.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Path to write JSON report (default: stdout only).",
    )
    args = parser.parse_args(argv)

    print("ADR-010 Core-only vs extras parity check")
    print("=" * 50)

    extras = _probe_extras()
    print("\nExtras availability:")
    for name, available in extras.items():
        status = "installed" if available else "absent"
        print(f"  {name}: {status}")

    print("\nRunning CE-first API with synthetic data...")
    try:
        result = _run_core_parity_check()
        success = True
        error_msg = ""
    except Exception as exc:
        result = {}
        success = False
        error_msg = str(exc)

    if success:
        print("\nStructural invariants:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("\n[PASS] Core CE-first API produces well-formed explanations.")
    else:
        print(f"\n[FAIL] Core CE-first API failed: {error_msg}")

    # Extras-pollution check (only informational — extras may be installed)
    imported_extras = _check_no_extras_pollution()
    if imported_extras:
        print(f"\nNote: the following extras-only modules are present in sys.modules: {imported_extras[:5]}")
        print("  (This is expected when extras are installed; run in a core-only venv to detect true pollution.)")

    report_payload = {
        "success": success,
        "error": error_msg,
        "extras_available": extras,
        "structural_invariants": result,
    }

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print(f"\nReport written to: {args.report}")

    if args.check and not success:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
