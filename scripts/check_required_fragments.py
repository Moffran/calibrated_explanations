"""Validate required shared documentation fragments are present."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# Load reusable shared fragments so we can assert their text presence where
# markdown doesn't use the templated include (e.g. README).
HERO_TEXT = (DOCS / "_shared" / "hero_calibrated_explanations.md").read_text(encoding="utf-8").strip()

README_CHECKS: dict[str, tuple[str, ...]] = {
    "README.md": (
        HERO_TEXT,
        "Backed by research",
        "docs/citing.md",
        "Optional extras",
    )
}

DOC_CHECKS: dict[str, tuple[str, ...]] = {
    "docs/index.md": (
        "{{ hero_calibrated_explanations }}",
        "{{ backed_by_research_banner }}",
        "{{ optional_extras_template }}",
    ),
    "docs/overview/index.md": (
        "{{ hero_calibrated_explanations }}",
        "{{ backed_by_research_banner }}",
        "{{ optional_extras_template }}",
    ),
    "docs/get-started/index.md": (
        "{{ hero_calibrated_explanations }}",
        "{{ backed_by_research_banner }}",
        "{{ optional_extras_template }}",
    ),
    "docs/concepts/index.md": (
        "{{ hero_calibrated_explanations }}",
        "{{ backed_by_research_banner }}",
        "{{ optional_extras_template }}",
    ),
    "docs/extending/index.md": (
        "{{ hero_calibrated_explanations }}",
        "{{ backed_by_research_banner }}",
        "{{ optional_extras_template }}",
    ),
    "docs/concepts/alternatives.md": (
        "{{ alternatives_triangular }}",
        "{{ optional_extras_template }}",
    ),
    "docs/get-started/quickstart_classification.md": (
        "{{ alternatives_triangular }}",
        "{{ optional_extras_template }}",
    ),
    "docs/get-started/quickstart_regression.md": (
        "{{ alternatives_triangular }}",
        "{{ optional_extras_template }}",
    ),
    "docs/external_plugins/index.md": (
        "{{ optional_extras_template }}",
    ),
}


def _check_file(path: Path, needles: tuple[str, ...]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    missing: list[str] = []
    for needle in needles:
        if needle not in text:
            missing.append(needle)
    return missing


def main() -> int:
    failures: list[str] = []

    for relative, needles in README_CHECKS.items():
        path = ROOT / relative
        missing = _check_file(path, needles)
        if missing:
            failures.append(
                f"{relative}: missing required fragment(s): {', '.join(missing)}"
            )

    for relative, needles in DOC_CHECKS.items():
        path = ROOT / relative
        missing = _check_file(path, needles)
        if missing:
            failures.append(
                f"{relative}: missing required fragment(s): {', '.join(missing)}"
            )

    if failures:
        for failure in failures:
            print(f"::error ::{failure}")
        print(
            "Documentation fragments check failed. "
            "Ensure hero, research banner, triangular plot, and optional extras includes are present."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
