"""Warning-policy inventory and fallback-site enforcement (ADR-028 / STD-005).

Scans library source for ``warnings.warn`` call sites and governed fallback
sites. The warning inventory is a drift check for existing visible warnings.
The fallback registry is the release gate: every known fallback site must
either emit the governed ``UserWarning`` + ``INFO`` pairing or carry an
explicit, reviewable exemption.

Usage
-----
    python scripts/quality/check_warning_policy.py
    python scripts/quality/check_warning_policy.py --check
    python scripts/quality/check_warning_policy.py --report reports/quality/warning_policy.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "calibrated_explanations"

VISIBLE_FALLBACK_POLICY: Final[str] = (
    "Retained user-visible fallbacks must emit UserWarning plus an INFO log. "
    "Internal best-effort fallbacks may remain exempt only when this registry "
    "records a reason and the available logger signal."
)

FALLBACK_MESSAGE_PATTERNS: Final[tuple[str, ...]] = (
    r"fall(?:ing)? back",
    r"fallback",
    r"feature filter enforcement skipped",
    r"using fallback feature_filter_config",
    r"drops the configured mondrian categorizer",
    r"failed to save state",
)


@dataclass(frozen=True)
class FallbackSiteSpec:
    """Registry entry describing one governed fallback site."""

    site_id: str
    rel_path: str
    context: str
    message_pattern: str
    disposition: str
    reason: str
    required_warning: bool = False
    required_log_level: str = "INFO"
    notes: str = ""


FALLBACK_SITE_REGISTRY: Final[tuple[FallbackSiteSpec, ...]] = (
    FallbackSiteSpec(
        site_id="parallel_pool_init_sequential_fallback",
        rel_path="parallel/parallel.py",
        context="ParallelExecutor.__enter__",
        message_pattern=r"Failed to initialize parallel pool.*falling back to sequential",
        disposition="user_visible",
        required_warning=True,
        reason=(
            "ADR-004 graceful degradation: an explicitly requested pool that cannot "
            "start runs sequentially, so the downgrade must be user-visible."
        ),
    ),
    FallbackSiteSpec(
        site_id="parallel_map_force_serial_fallback",
        rel_path="parallel/parallel.py",
        context="ParallelExecutor.map",
        message_pattern=r"Parallel execution failed.*falling back to sequential",
        disposition="user_visible",
        required_warning=True,
        reason=(
            "force_serial_on_failure opts into serial recovery after a parallel "
            "failure; the retained fallback must still be user-visible."
        ),
    ),
    FallbackSiteSpec(
        site_id="parallel_joblib_unavailable_thread_fallback",
        rel_path="parallel/parallel.py",
        context="ParallelExecutor.joblib_strategy",
        message_pattern=r"Joblib is not available; falling back to thread-based",
        disposition="user_visible",
        required_warning=True,
        reason=(
            "An explicitly requested joblib strategy runs on threads when joblib is "
            "not installed, so the substitution must be user-visible."
        ),
    ),
    FallbackSiteSpec(
        site_id="runtime_env_unrecognised_token_ignored",
        rel_path="core/config_manager.py",
        context="warn_unrecognised_env_token",
        message_pattern=r"contains unrecognised token .*it is ignored",
        disposition="user_visible",
        required_warning=True,
        reason=(
            "ADR-034 / #219: an unrecognised CE_CACHE or CE_PARALLEL token is ignored, "
            "so the requested behaviour is not applied and the drop must be "
            "user-visible. Escalates to ConfigurationError in v1.1.0 (#225)."
        ),
    ),
    FallbackSiteSpec(
        site_id="cache_backend_minimal_lru_fallback",
        rel_path="cache/cache.py",
        context="_notify_minimal_cache_backend",
        message_pattern=r"Cache backend fallback: cachetools not available",
        disposition="exempt",
        required_log_level="WARNING",
        reason=(
            "The requested cache is still honoured: the in-package backend implements "
            "the same LRU/TTL semantics as cachetools (an optional 'perf' extra). The "
            "notice is logged once per process, only when an enabled cache is built."
        ),
    ),
    FallbackSiteSpec(
        site_id="execution_plugin_unsupported_legacy_fallback",
        rel_path="plugins/builtins.py",
        context="explain_batch",
        message_pattern=r"Execution plugin unsupported; falling back to legacy sequential execution",
        disposition="exempt",
        required_log_level="WARNING",
        reason=(
            "Plugin capability mismatches are governed as internal execution-routing "
            "fallbacks. They must remain statically registered even when the runtime "
            "signal is logger-only."
        ),
    ),
    FallbackSiteSpec(
        site_id="execution_plugin_supports_failure_legacy_fallback",
        rel_path="plugins/builtins.py",
        context="explain_batch",
        message_pattern=r"Execution plugin supports\(\) check failed.*falling back to legacy",
        disposition="exempt",
        required_log_level="WARNING",
        reason=(
            "Capability-check failures currently degrade through the legacy execution "
            "path. The site stays allowlisted only through this explicit registry entry."
        ),
    ),
    FallbackSiteSpec(
        site_id="execution_plugin_runtime_failure_legacy_fallback",
        rel_path="plugins/builtins.py",
        context="explain_batch",
        message_pattern=r"Execution plugin failed for mode .*falling back to legacy",
        disposition="exempt",
        required_log_level="WARNING",
        reason=(
            "Execution failures degrade to the legacy explainer path. The release gate "
            "must keep tracking the logger-only signal until the runtime contract is "
            "tightened."
        ),
    ),
    FallbackSiteSpec(
        site_id="feature_filter_chain_read_skipped",
        rel_path="core/calibrated_explainer.py",
        context="_enforce_feature_filter_plugin_preferences",
        message_pattern=r"feature filter enforcement skipped",
        disposition="exempt",
        required_log_level="WARNING",
        reason=(
            "ADR-027 currently allows FAST feature-filter enforcement to fail open. "
            "This exemption is recorded so the warning-policy gate reports the site "
            "instead of missing it."
        ),
    ),
    FallbackSiteSpec(
        site_id="deepcopy_shallow_copy_fallback",
        rel_path="core/calibrated_explainer.py",
        context="__deepcopy__",
        message_pattern=r"fallback to shallow copy",
        disposition="exempt",
        required_log_level="NONE",
        reason=(
            "Deep-copy fallback is still implemented through exception suppression with "
            "no runtime signal. It remains an explicit exemption so the gate does not "
            "pretend the path is covered by visible warning policy."
        ),
        notes="No logger signal exists yet; tracked as a silent best-effort fallback.",
    ),
    FallbackSiteSpec(
        site_id="reset_runtime_helper_suppression",
        rel_path="core/calibrated_explainer.py",
        context="reset",
        message_pattern=r"clear_(?:explanation_plugin_instances|explanation_plugin_identifiers|bridge_monitors)",
        disposition="exempt",
        required_log_level="NONE",
        reason=(
            "Reset helper cleanup still suppresses teardown errors silently. The site is "
            "explicitly recorded here so the gate reports the exemption instead of "
            "claiming full fallback visibility."
        ),
        notes="Static suppression-only exemption.",
    ),
)


DEPRECATION_ONLY_PATHS: Final[frozenset[str]] = frozenset(
    {
        "utils/deprecations.py",
        "utils/deprecation.py",
    }
)


class WarnSite(NamedTuple):
    """A ``warnings.warn`` call site."""

    rel_path: str
    line: int
    message_snippet: str
    context: str


class LogSite(NamedTuple):
    """A logger call site."""

    rel_path: str
    line: int
    level: str
    message_snippet: str
    context: str


@dataclass(frozen=True)
class FallbackSiteResult:
    """Evaluation result for one fallback registry entry."""

    site_id: str
    rel_path: str
    context: str
    disposition: str
    required_log_level: str
    matched_warning: bool
    matched_log: bool
    reason: str
    notes: str
    status: str
    violations: tuple[str, ...]


def _extract_message_snippet(node: ast.Call) -> str:
    """Return a short message snippet from the first positional argument."""
    if not node.args:
        return ""
    first = node.args[0]
    # ``"template %s" % args`` keeps its literal template as the snippet.
    if (
        isinstance(first, ast.BinOp)
        and isinstance(first.op, ast.Mod)
        and isinstance(first.left, ast.Constant)
    ):
        first = first.left
    if isinstance(first, ast.Constant):
        return str(first.value)[:160]
    if isinstance(first, ast.JoinedStr):
        return "<f-string>"
    return ""


class _SignalVisitor(ast.NodeVisitor):
    """Collect warning and logger signals while tracking function context."""

    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.context_stack: list[str] = []
        self.warn_sites: list[WarnSite] = []
        self.log_sites: list[LogSite] = []

    def _context(self) -> str:
        return ".".join(self.context_stack) if self.context_stack else "<module>"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.context_stack.append(node.name)
        self.generic_visit(node)
        self.context_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.context_stack.append(node.name)
        self.generic_visit(node)
        self.context_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.context_stack.append(node.name)
        self.generic_visit(node)
        self.context_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        context = self._context()
        if isinstance(func, ast.Attribute) and func.attr == "warn":
            if isinstance(func.value, ast.Name) and func.value.id == "warnings":
                self.warn_sites.append(
                    WarnSite(self.rel_path, node.lineno, _extract_message_snippet(node), context)
                )
        elif isinstance(func, ast.Name) and func.id == "warn":
            self.warn_sites.append(
                WarnSite(self.rel_path, node.lineno, _extract_message_snippet(node), context)
            )

        log_level = _extract_log_level(func)
        if log_level is not None:
            self.log_sites.append(
                LogSite(
                    self.rel_path,
                    node.lineno,
                    log_level,
                    _extract_message_snippet(node),
                    context,
                )
            )
        self.generic_visit(node)


def _extract_log_level(func: ast.AST) -> str | None:
    """Return the logger level for a call expression when it looks like a log call."""
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr not in {"debug", "info", "warning", "error", "exception"}:
        return None
    return func.attr.upper()


def _parse_signals(src_root: Path) -> tuple[list[WarnSite], list[LogSite]]:
    """Walk source files and collect warning and log call sites."""
    warn_sites: list[WarnSite] = []
    log_sites: list[LogSite] = []
    for py_file in sorted(src_root.rglob("*.py")):
        rel = py_file.relative_to(src_root).as_posix()
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue
        visitor = _SignalVisitor(rel)
        visitor.visit(tree)
        warn_sites.extend(visitor.warn_sites)
        log_sites.extend(visitor.log_sites)
    return warn_sites, log_sites


def classify_warning_site(site: WarnSite) -> str:
    """Return a coarse category for a warning call site."""
    if site.rel_path in DEPRECATION_ONLY_PATHS:
        return "DEPRECATION"
    if re.search("|".join(FALLBACK_MESSAGE_PATTERNS), site.message_snippet, flags=re.IGNORECASE):
        return "FALLBACK_VISIBLE"
    return "USER_CONTRACT"


def evaluate_fallback_registry(
    warn_sites: list[WarnSite],
    log_sites: list[LogSite],
    registry: tuple[FallbackSiteSpec, ...] = FALLBACK_SITE_REGISTRY,
) -> list[FallbackSiteResult]:
    """Evaluate fallback registry entries against static warning/log signals."""
    results: list[FallbackSiteResult] = []
    for spec in registry:
        context_pattern = re.compile(re.escape(spec.context))
        message_pattern = re.compile(spec.message_pattern, flags=re.IGNORECASE)
        matched_warning = any(
            site.rel_path == spec.rel_path
            and context_pattern.search(site.context)
            and message_pattern.search(site.message_snippet)
            for site in warn_sites
        )
        matched_log = any(
            site.rel_path == spec.rel_path
            and context_pattern.search(site.context)
            and message_pattern.search(site.message_snippet)
            and (spec.required_log_level == "NONE" or site.level == spec.required_log_level)
            for site in log_sites
        )
        violations: list[str] = []
        if spec.disposition == "user_visible":
            if not matched_warning:
                violations.append("missing UserWarning signal")
            if not matched_log:
                violations.append(f"missing {spec.required_log_level} log signal")
        elif spec.disposition == "exempt":
            if not spec.reason.strip():
                violations.append("missing exemption reason")
            if spec.required_log_level != "NONE" and not matched_log:
                violations.append(f"missing exempted {spec.required_log_level} log signal")
        else:
            violations.append(f"unknown disposition: {spec.disposition}")
        results.append(
            FallbackSiteResult(
                site_id=spec.site_id,
                rel_path=spec.rel_path,
                context=spec.context,
                disposition=spec.disposition,
                required_log_level=spec.required_log_level,
                matched_warning=matched_warning,
                matched_log=matched_log,
                reason=spec.reason,
                notes=spec.notes,
                status="pass" if not violations else "fail",
                violations=tuple(violations),
            )
        )
    return results


def build_payload() -> dict[str, object]:
    """Build the full warning-policy report payload."""
    warn_sites, log_sites = _parse_signals(SRC_ROOT)
    classified = [
        {
            "file": site.rel_path,
            "line": site.line,
            "context": site.context,
            "message_snippet": site.message_snippet,
            "category": classify_warning_site(site),
        }
        for site in warn_sites
    ]
    fallback_results = evaluate_fallback_registry(warn_sites, log_sites)
    fallback_failures = [result for result in fallback_results if result.status == "fail"]
    category_counts: dict[str, int] = {}
    for entry in classified:
        category = str(entry["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "policy": VISIBLE_FALLBACK_POLICY,
        "warnings": {
            "total": len(warn_sites),
            "by_category": category_counts,
            "sites": classified,
        },
        "fallback_registry": {
            "total": len(fallback_results),
            "failures": len(fallback_failures),
            "sites": [asdict(result) for result in fallback_results],
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the warning-policy check."""
    parser = argparse.ArgumentParser(
        description="Warning-policy inventory and fallback-site coverage check (ADR-028 / STD-005)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any fallback registry entry is uncovered or malformed.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Write JSON report to this path.",
    )
    args = parser.parse_args(argv)

    payload = build_payload()
    warning_total = int(payload["warnings"]["total"])  # type: ignore[index]
    category_counts = payload["warnings"]["by_category"]  # type: ignore[index]
    fallback_total = int(payload["fallback_registry"]["total"])  # type: ignore[index]
    fallback_failures = int(payload["fallback_registry"]["failures"])  # type: ignore[index]
    fallback_sites = payload["fallback_registry"]["sites"]  # type: ignore[index]

    print("ADR-028 / STD-005 Warning-policy check")
    print("=" * 50)
    print(VISIBLE_FALLBACK_POLICY)
    print()
    print(f"warnings.warn inventory sites: {warning_total}")
    for category in sorted(category_counts):
        print(f"  {category}: {category_counts[category]}")  # type: ignore[index]
    print(f"Fallback registry entries: {fallback_total}")
    print(f"Fallback registry failures: {fallback_failures}")

    if fallback_failures:
        print("\nFallback registry violations:")
        for site in fallback_sites:
            if site["status"] != "fail":
                continue
            joined = ", ".join(site["violations"])
            print(
                f"  {site['site_id']} ({site['file'] if 'file' in site else site['rel_path']}::{site['context']}): {joined}"
            )

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"\nReport written to: {args.report}")

    if fallback_failures == 0:
        print("\n[PASS] Warning-policy inventory complete and fallback registry covered.")
    else:
        print("\n[FAIL] Fallback registry contains uncovered or malformed sites.")

    if args.check and fallback_failures > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
