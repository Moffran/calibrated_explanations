from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def _resolve_doc(slug: str) -> Path:
    slug = slug.lstrip("./")
    candidate = DOCS / slug
    if candidate.suffix:
        return candidate
    md_candidate = candidate.with_suffix(".md")
    if md_candidate.exists():
        return md_candidate
    rst_candidate = candidate.with_suffix(".rst")
    return rst_candidate


def test_top_level_toctree_targets_exist() -> None:
    expected = (
        "overview/index",
        "get-started/index",
        "how-to/index",
        "concepts/index",
        "reference/index",
        "extending/index",
        "governance/index",
    )
    index_path = _resolve_doc("index")
    index = index_path.read_text()
    for slug in expected:
        assert slug in index, f"Missing toctree entry for {slug}"
        target = _resolve_doc(slug)
        assert target.exists(), f"Document referenced by {slug} is missing"


def test_crosswalk_covers_legacy_pages() -> None:
    crosswalk = (DOCS / "governance" / "nav_crosswalk.md").read_text()
    mapping: dict[str, tuple[str, ...]] = {
        "calibrated_explanations.rst": ("overview/index",),
        "getting_started.md": (
            "get-started/index",
            "get-started/installation",
            "get-started/quickstart_classification",
            "get-started/quickstart_regression",
            "get-started/troubleshooting",
        ),
        "viz_plotspec.md": ("how-to/plot_with_plotspec",),
        "architecture.md": ("concepts/index", "concepts/telemetry"),
        "pr_guide.md": ("governance/release_notes", "governance/section_owners"),
    }

    for legacy, replacements in mapping.items():
        assert legacy in crosswalk, f"Legacy page {legacy} missing from crosswalk"
        for slug in replacements:
            needle = slug.split("/", 1)[1] if slug.startswith("governance/") else slug
            assert (
                needle in crosswalk
            ), f"Replacement {slug} missing from crosswalk entry for {legacy}"
            target = _resolve_doc(slug)
            assert target.exists(), f"Replacement document {target} missing on disk"
