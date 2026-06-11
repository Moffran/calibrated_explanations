"""Content coverage tests for the v1.0.0 upgrade checklist and safe-defaults guide."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKLIST = ROOT / "docs" / "upgrade" / "v1.0.0-upgrade-checklist.md"
SAFE_DEFAULTS = ROOT / "docs" / "foundations" / "how-to" / "safe-defaults.md"


def load_known_env_keys() -> list[str]:
    """Parse CE_ keys from config_manager.py source without importing private attrs."""
    src = (ROOT / "src" / "calibrated_explanations" / "core" / "config_manager.py").read_text(
        encoding="utf-8"
    )
    m = re.search(r"_DEFAULT_KNOWN_ENV_KEYS\s*=\s*\((.*?)\)", src, re.DOTALL)
    assert m, "Could not locate _DEFAULT_KNOWN_ENV_KEYS in config_manager.py"
    return re.findall(r'"(CE_\w+|CI|GITHUB_ACTIONS)"', m.group(1))


def test_all_known_env_keys_in_checklist():
    text = CHECKLIST.read_text(encoding="utf-8")
    missing = [k for k in load_known_env_keys() if k not in text]
    assert not missing, f"Checklist missing CE_ keys: {missing}"


def extract_method_name(cell: str) -> str | None:
    """Extract a short method/attribute identifier from a deprecation table cell.

    Returns the bare name (no class prefix, no args) if it looks like a method
    or attribute (contains an underscore). Returns None for module paths and
    prose entries that don't yield a meaningful identifier.
    """
    m = re.search(r"`([^`]+)`", cell)
    if not m:
        return None
    raw = m.group(1)
    name = raw.split("(")[0]
    name = name.split(".")[-1]
    name = name.strip()
    if "_" in name and re.fullmatch(r"[A-Za-z_]\w*", name):
        return name
    return None


def test_removed_v0_11_symbols_in_checklist():
    ledger = (ROOT / "docs" / "migration" / "deprecations.md").read_text(encoding="utf-8")
    text = CHECKLIST.read_text(encoding="utf-8")
    in_removed = False
    missing = []
    for line in ledger.splitlines():
        if "### Removed deprecations" in line:
            in_removed = True
            continue
        if (
            in_removed
            and line.startswith("|")
            and not line.startswith("|---")
            and "Removed symbol" not in line
            and "Deprecated symbol" not in line
        ):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 4:
                removed_in = cells[3] if len(cells) > 3 else ""
                if removed_in.startswith("v0.11"):
                    name = extract_method_name(cells[0])
                    if name and name not in text:
                        missing.append(name)
    assert not missing, f"Checklist missing removed v0.11 symbols: {missing}"


def test_canonical_reject_policy_members_in_checklist():
    text = CHECKLIST.read_text(encoding="utf-8")
    for member in (
        "RejectPolicy.FLAG",
        "RejectPolicy.ONLY_REJECTED",
        "RejectPolicy.ONLY_ACCEPTED",
        "RejectPolicy.NONE",
    ):
        assert member in text, f"Missing canonical RejectPolicy member: {member}"


def test_guarded_deprecation_in_checklist():
    text = CHECKLIST.read_text(encoding="utf-8")
    assert "explain_guarded_factual" in text, "Checklist missing guarded deprecation migration"
    assert "explore_guarded_alternatives" in text, "Checklist missing guarded deprecation migration"


def test_safe_defaults_required_content():
    text = SAFE_DEFAULTS.read_text(encoding="utf-8")
    required = (
        "CE_DEBUG_TRUST_INVARIANTS",  # production warning must be present
        "CE_STRICT_OBSERVABILITY",  # observability key must be mentioned
        "CE_DEPRECATIONS=error",  # CI migration pattern must be present
        "get_process_config_manager",  # correct call pattern (not from_sources)
        "CE_DEPRECATIONS",  # must appear in safe-defaults
    )
    missing = [s for s in required if s not in text]
    assert not missing, f"safe-defaults.md missing required content: {missing}"


def test_explainer_builder_api_correctness():
    # Verify ExplainerBuilder/ExplainerConfig are importable and the documented
    # call patterns actually work without error.
    from calibrated_explanations import ExplainerBuilder, ExplainerConfig, WrapCalibratedExplainer
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(random_state=0)
    # Direct config construction pattern (topic 5, first example)
    config = ExplainerConfig(model=model, auto_encode=True)
    # from_config takes only the config — model is inside it
    wrapper = WrapCalibratedExplainer.from_config(config)
    assert isinstance(wrapper, WrapCalibratedExplainer)

    # Fluent builder pattern (topic 5, second example)
    config2 = (
        ExplainerBuilder(model)
        .perf_cache(True, max_items=256)
        .perf_parallel(True, workers=2, min_batch=8)
        .build_config()
    )
    wrapper2 = WrapCalibratedExplainer.from_config(config2)
    assert isinstance(wrapper2, WrapCalibratedExplainer)


def test_phantom_env_vars_absent():
    phantoms = (
        "CE_CACHE_MAX_ITEMS",
        "CE_CACHE_MAX_BYTES",
        "CE_CACHE_TTL",
        "CE_PARALLEL_WORKERS",
    )
    for f in (CHECKLIST, SAFE_DEFAULTS):
        text = f.read_text(encoding="utf-8")
        for phantom in phantoms:
            assert phantom not in text, f"{f.name} documents non-existent env var {phantom}"
