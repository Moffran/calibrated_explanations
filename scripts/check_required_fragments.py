"""Validate required shared documentation fragments are present and ordered."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# Load reusable shared fragments so we can assert their text presence where
# markdown doesn't use the templated include (e.g. README).
HERO_TEXT = (DOCS / "_shared" / "hero_calibrated_explanations.md").read_text(encoding="utf-8").strip()

README_CHECKS: dict[str, tuple[str, ...]] = {
    "README.md": (
        HERO_TEXT,
        "docs/research/index.md",
        "docs/citing.md",
        "## Optional extras",
    )
}

DOC_CHECKS: dict[str, tuple[str, ...]] = {
    "docs/index.md": (
        "{{ hero_calibrated_explanations }}",
        "research/index",
        "{{ optional_extras_template }}",
    ),
    "docs/overview/index.md": (
        "{{ hero_calibrated_explanations }}",
        "research/index",
        "{{ optional_extras_template }}",
    ),
    "docs/get-started/index.md": (
        "{{ hero_calibrated_explanations }}",
        "research/index",
        "{{ optional_extras_template }}",
    ),
    "docs/concepts/index.md": (
        "{{ hero_calibrated_explanations }}",
        "research/index",
        "{{ optional_extras_template }}",
    ),
    "docs/concepts/probabilistic_regression.md": (
        "{{ optional_extras_template }}",
    ),
    "docs/extending/index.md": (
        "{{ hero_calibrated_explanations }}",
        "research/index",
        "{{ optional_extras_template }}",
    ),
    "docs/practitioner/index.md": (
        "{{ hero_calibrated_explanations }}",
        "{{ optional_extras_template }}",
    ),
    "docs/researcher/index.md": (
        "{{ hero_calibrated_explanations }}",
        "{{ optional_extras_template }}",
    ),
    "docs/contributor/index.md": (
        "{{ hero_calibrated_explanations }}",
        "{{ optional_extras_template }}",
    ),
    "docs/how-to/configure_telemetry.md": (
        "Optional telemetry extra",
        "{{ optional_extras_template }}",
    ),
    "docs/how-to/integrate_with_pipelines.md": (
        "Telemetry note:",
        "{{ optional_extras_template }}",
    ),
    "docs/governance/optional_telemetry.md": (
        "Optional telemetry extra",
        "{{ optional_extras_template }}",
    ),
    "docs/concepts/telemetry.md": (
        "Telemetry moved to governance",
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
    "docs/plugins.md": (
        "{{ optional_extras_template }}",
    ),
}

ORDER_RULES: dict[str, tuple[str, bool, bool]] = {
    "README.md": ("## Optional extras", False, True),
    "docs/index.md": ("{{ optional_extras_template }}", False, False),
    "docs/overview/index.md": ("{{ optional_extras_template }}", False, False),
    "docs/get-started/index.md": ("{{ optional_extras_template }}", False, False),
    "docs/concepts/index.md": ("{{ optional_extras_template }}", False, False),
    "docs/extending/index.md": ("{{ optional_extras_template }}", False, False),
    "docs/practitioner/index.md": ("{{ optional_extras_template }}", False, False),
    "docs/researcher/index.md": ("{{ optional_extras_template }}", False, False),
    "docs/contributor/index.md": ("{{ optional_extras_template }}", False, False),
    "docs/how-to/configure_telemetry.md": ("{{ optional_extras_template }}", False, False),
    "docs/how-to/integrate_with_pipelines.md": ("{{ optional_extras_template }}", False, False),
    "docs/governance/optional_telemetry.md": ("{{ optional_extras_template }}", False, False),
    "docs/concepts/telemetry.md": ("{{ optional_extras_template }}", False, False),
    "docs/concepts/alternatives.md": ("{{ optional_extras_template }}", False, False),
    "docs/concepts/probabilistic_regression.md": ("{{ optional_extras_template }}", False, False),
    "docs/get-started/quickstart_classification.md": ("{{ optional_extras_template }}", False, False),
    "docs/get-started/quickstart_regression.md": ("{{ optional_extras_template }}", False, False),
    "docs/external_plugins/index.md": ("{{ optional_extras_template }}", False, False),
    "docs/plugins.md": ("{{ optional_extras_template }}", False, False),
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

    for relative, (marker, allow_reference_defs, marker_is_heading) in ORDER_RULES.items():
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        occurrences = text.count(marker)
        if occurrences == 0:
            # Presence already reported above if required; skip order enforcement here.
            continue
        if occurrences > 1:
            failures.append(f"{relative}: '{marker}' appears {occurrences} times; expected exactly once")
            continue
        position = text.index(marker)
        trailing = text[position + len(marker):]
        if marker_is_heading:
            next_heading = re.search(r"\n#{1,6}\s", trailing)
            if next_heading:
                failures.append(
                    f"{relative}: contains additional heading(s) after the optional extras section"
                )
            continue
        if allow_reference_defs:
            trailing_lines = []
            for line in trailing.splitlines():
                if re.match(r"^\s*\[[^\]]+\]:", line):
                    continue
                trailing_lines.append(line)
            trailing = "\n".join(trailing_lines)
        if trailing.strip():
            failures.append(f"{relative}: contains additional content after the optional extras section")

    if failures:
        for failure in failures:
            print(f"::error ::{failure}")
        print(
            "Documentation fragments check failed. "
            "Ensure hero, research hub reference, triangular plot, and optional extras includes are present."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
