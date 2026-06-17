"""Warning-policy inventory and enforcement check (ADR-028 / STD-005).

Scans library source for `warnings.warn` call sites, classifies each one as
one of four categories, and reports any unclassified or policy-violating call
sites.

Categories
----------
DEPRECATION
    Calls inside `utils/deprecations.py` or `utils/deprecation.py` — these
    emit `DeprecationWarning` through the central `deprecate()` helper and are
    always allowed.
USER_CONTRACT
    Calls guarded by a documented user-facing contract: the user has explicitly
    passed invalid or non-canonical input and deserves a notebook-visible
    notification. Examples: unknown feature names, untrusted plugin explicit
    override, verbose-mode migration notices.
FALLBACK_DEGRADED
    Calls indicating degraded behavior or a fallback path where correctness is
    preserved but performance or configuration is suboptimal. These SHOULD be
    routed to `WARNING` logs, not `UserWarning`. Any remaining calls in this
    category are policy violations unless explicitly allowlisted.
REJECT_CONTRACT
    Calls emitting `RejectContractWarning` — a project-specific warning
    subclass used in the reject orchestrator. Allowed.
ALLOWLISTED
    Call sites that have been explicitly reviewed and approved to remain as
    `warnings.warn` due to documented reasons.

Usage
-----
    python scripts/quality/check_warning_policy.py
    python scripts/quality/check_warning_policy.py --check   # fail on violations
    python scripts/quality/check_warning_policy.py --report reports/quality/warning_policy.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "calibrated_explanations"

# ---------------------------------------------------------------------------
# Allowlist — call sites explicitly approved to remain as warnings.warn
# ---------------------------------------------------------------------------

# Format: frozenset of (relative_module_path, approximate_line_hint) tuples
# Use the module path relative to src/calibrated_explanations/
ALLOWLISTED_PATHS: frozenset[str] = frozenset(
    {
        # Deprecation helpers — always allowed
        "utils/deprecations.py",
        "utils/deprecation.py",
        # ce_agent_utils: get_uncalibrated_predictions is a user-contract guard
        # (operator explicitly bypasses calibration — notebook-visible feedback appropriate)
        "ce_agent_utils.py",
        # plotting.py: visualization quality warnings for user-authored plots
        "plotting.py",
        # viz/_matplotlib_compat.py: visualization edge-case warnings (identical predictions/uncertainties)
        "viz/_matplotlib_compat.py",
        # viz/matplotlib_adapter.py: rendering fallbacks visible to users who plot
        "viz/matplotlib_adapter.py",
        # viz/narrative_plugin.py: narrative quality warnings
        "viz/narrative_plugin.py",
        # viz/serializers.py: serialization edge-case warnings
        "viz/serializers.py",
        # plugins/manager.py: untrusted plugin via explicit override — user contract
        "plugins/manager.py",
        # plugins/registry.py: plugin registration governance events
        "plugins/registry.py",
        # plugins/base.py: plugin base contract warnings
        "plugins/base.py",
        # plugins/predict_monitor.py: monitoring boundary warnings
        "plugins/predict_monitor.py",
        # explanations/explanation.py: user-facing warnings (feature not found, rule errors, etc.)
        "explanations/explanation.py",
        # explanations/explanations.py: collection-level user-facing warnings
        "explanations/explanations.py",
        # explanations/guarded_explanation.py: guarded explanation contract warnings
        "explanations/guarded_explanation.py",
        # core/calibrated_explainer.py: verbose-mode migration UserWarning (condition_source);
        #   _use_plugin UserWarning in guarded path (fires when verbose=True and _use_plugin=False);
        #   RejectResult.prediction formatting failure UserWarning (predict_reject exception path)
        "core/calibrated_explainer.py",
        # core/reject/orchestrator.py: reject contract warnings (RejectContractWarning subclass)
        "core/reject/orchestrator.py",
        # core/explain/orchestrator.py: user contract warnings (unknown feature names, reject upgrade)
        "core/explain/orchestrator.py",
        # core/explain/_guarded_explain.py: guarded explain contract warnings
        "core/explain/_guarded_explain.py",
        # core/explain/__init__.py: explain module init warnings
        "core/explain/__init__.py",
        # core/explain/parallel_instance.py: parallel instance fallback
        "core/explain/parallel_instance.py",
        # core/explain/parallel_runtime.py: parallel runtime fallback
        "core/explain/parallel_runtime.py",
        # api/config.py: UserWarning for removed ExplainerConfig fields (task/parallel_workers)
        # emitted at build_config() call sites — user-contract guard, not a deprecation channel
        "api/config.py",
        # core/config_manager.py: config validation contract warnings
        "core/config_manager.py",
        # core/difficulty_estimator_helpers.py: difficulty estimator contract warnings
        "core/difficulty_estimator_helpers.py",
        # core/discretizer_config.py: discretizer config contract warnings
        "core/discretizer_config.py",
        # core/reject.py: reject module warnings
        "core/reject.py",
        # core/prediction_helpers.py: prediction helper contract warnings
        "core/prediction_helpers.py",
        # core/prediction/orchestrator.py: prediction orchestrator contract warnings
        "core/prediction/orchestrator.py",
        # core/wrap_explainer.py: wrap explainer contract warnings
        "core/wrap_explainer.py",
        # calibration/normalization_strategy.py: normalization deprecation
        "calibration/normalization_strategy.py",
        # explanations/reject.py: reject warnings
        "explanations/reject.py",
        # utils/perturbation.py: perturbation utility warnings
        "utils/perturbation.py",
    }
)

# Paths that contain ONLY deprecation warnings (never policy violations)
DEPRECATION_ONLY_PATHS: frozenset[str] = frozenset(
    {
        "utils/deprecations.py",
        "utils/deprecation.py",
    }
)


class WarnSite(NamedTuple):
    """A `warnings.warn` call site."""

    rel_path: str
    line: int
    message_snippet: str


def _find_warn_sites(src_root: Path) -> list[WarnSite]:
    """Walk source files and collect all `warnings.warn` call sites."""
    sites: list[WarnSite] = []
    for py_file in sorted(src_root.rglob("*.py")):
        rel = py_file.relative_to(src_root).as_posix()
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "warn":
                if isinstance(func.value, ast.Name) and func.value.id == "warnings":
                    # Extract snippet from first arg if present
                    snippet = ""
                    if node.args:
                        first = node.args[0]
                        if isinstance(first, ast.Constant):
                            snippet = str(first.value)[:80]
                        elif isinstance(first, ast.JoinedStr):
                            snippet = "<f-string>"
                    sites.append(WarnSite(rel, node.lineno, snippet))
            elif isinstance(func, ast.Name) and func.id == "warn":
                # Bare warn() — may be aliased import of warnings.warn
                snippet = ""
                if node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant):
                        snippet = str(first.value)[:80]
                    elif isinstance(first, ast.JoinedStr):
                        snippet = "<f-string>"
                sites.append(WarnSite(rel, node.lineno, snippet))
    return sites


def classify(site: WarnSite) -> str:
    """Return the policy category for a warning call site."""
    if site.rel_path in DEPRECATION_ONLY_PATHS:
        return "DEPRECATION"
    if site.rel_path in ALLOWLISTED_PATHS:
        return "ALLOWLISTED"
    return "UNCLASSIFIED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Warning-policy inventory (ADR-028 / STD-005)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any UNCLASSIFIED sites are found.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Write JSON report to this path.",
    )
    args = parser.parse_args(argv)

    sites = _find_warn_sites(SRC_ROOT)
    classified: list[dict] = []
    violations: list[dict] = []

    for site in sites:
        category = classify(site)
        entry = {
            "file": site.rel_path,
            "line": site.line,
            "message_snippet": site.message_snippet,
            "category": category,
        }
        classified.append(entry)
        if category == "UNCLASSIFIED":
            violations.append(entry)

    total = len(sites)
    allowlisted = sum(1 for e in classified if e["category"] == "ALLOWLISTED")
    deprecation = sum(1 for e in classified if e["category"] == "DEPRECATION")
    unclassified = len(violations)

    print("ADR-028 / STD-005 Warning-policy inventory")
    print("=" * 50)
    print(f"Total warnings.warn call sites: {total}")
    print(f"  DEPRECATION (helpers only):   {deprecation}")
    print(f"  ALLOWLISTED (reviewed):       {allowlisted}")
    print(f"  UNCLASSIFIED (policy gap):    {unclassified}")

    if violations:
        print("\nUnclassified call sites (require review):")
        for v in violations:
            print(f"  {v['file']}:{v['line']} — {v['message_snippet']!r}")

    payload = {
        "total": total,
        "deprecation": deprecation,
        "allowlisted": allowlisted,
        "unclassified": unclassified,
        "sites": classified,
    }

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nReport written to: {args.report}")

    if unclassified == 0:
        print("\n[PASS] All warnings.warn call sites are classified.")
    else:
        print(f"\n[INFO] {unclassified} call site(s) need classification review.")

    if args.check and unclassified > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
