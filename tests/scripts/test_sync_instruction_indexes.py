"""Tests for generated instruction indexes."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.quality.sync_instruction_indexes import GENERATED_FILES, main


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_source_tree(root: Path) -> None:
    _write(
        root / "development/adrs/ADR-001-example.md",
        """---
id: "ADR-001"
title: "Example ADR"
status: "accepted"
description: "Governs example architecture."
supersedes: []
superseded_by: []
---

# ADR-001: Example ADR

## Decision

Example.
""",
    )
    _write(
        root / "development/standards/STD-001-example.md",
        """---
id: "STD-001"
title: "Example Standard"
status: "active"
description: "Governs example engineering."
supersedes: []
superseded_by: []
---

# Standard-001: Example Standard

Example.
""",
    )
    _write(
        root / "skills/ce-example/SKILL.md",
        """---
name: ce-example
description: >-
  Demonstrate deterministic skill discovery.
---

# CE Example
""",
    )


def _generated_contents(root: Path) -> dict[str, str]:
    return {rel_path: (root / rel_path).read_text(encoding="utf-8") for rel_path in GENERATED_FILES}


def test_should_generate_all_indexes_when_write_runs(tmp_path: Path) -> None:
    """Write mode creates every discovery index from canonical sources."""
    # Arrange
    _write_source_tree(tmp_path)

    # Act
    exit_code = main(["--root", str(tmp_path), "--write"])

    # Assert
    assert exit_code == 0
    generated = _generated_contents(tmp_path)
    assert all(
        expected_text in generated[path]
        for path, expected_text in {
            "development/adrs/README.md": "ADR-001",
            "development/standards/README.md": "STD-001",
            "skills/README.md": "ce-example",
            "docs/contributor/agent_skills.md": "ce-example",
        }.items()
    )


def test_should_reflect_source_changes_when_write_runs(tmp_path: Path) -> None:
    """Generated catalogues update when descriptions or inventory membership change."""
    # Arrange
    _write_source_tree(tmp_path)
    assert main(["--root", str(tmp_path), "--write"]) == 0
    _write(
        tmp_path / "skills/ce-second/SKILL.md",
        """---
name: ce-second
description: New source record.
---

# CE Second
""",
    )
    standard = tmp_path / "development/standards/STD-001-example.md"
    standard.write_text(
        standard.read_text(encoding="utf-8").replace(
            "Governs example engineering.",
            "Updated applicability description.",
        ),
        encoding="utf-8",
    )
    (tmp_path / "skills/ce-example/SKILL.md").unlink()
    (tmp_path / "skills/ce-example").rmdir()

    # Act
    exit_code = main(["--root", str(tmp_path), "--write"])

    # Assert
    generated = _generated_contents(tmp_path)
    assert exit_code == 0
    assert "ce-second" in generated["skills/README.md"]
    assert "ce-example" not in generated["docs/contributor/agent_skills.md"]
    assert "Updated applicability description." in generated["development/standards/README.md"]


def test_should_be_deterministic_and_idempotent_when_write_repeats(tmp_path: Path) -> None:
    """Repeated generation leaves byte-identical files."""
    # Arrange
    _write_source_tree(tmp_path)
    assert main(["--root", str(tmp_path), "--write"]) == 0
    first = _generated_contents(tmp_path)

    # Act
    exit_code = main(["--root", str(tmp_path), "--write"])

    # Assert
    assert exit_code == 0
    assert _generated_contents(tmp_path) == first


def test_should_fail_without_mutating_when_generated_index_is_stale(tmp_path: Path) -> None:
    """Check mode reports drift without repairing it."""
    # Arrange
    _write_source_tree(tmp_path)
    assert main(["--root", str(tmp_path), "--write"]) == 0
    index = tmp_path / "development/adrs/README.md"
    index.write_text("stale\n", encoding="utf-8")
    before = _generated_contents(tmp_path)

    # Act
    exit_code = main(["--root", str(tmp_path), "--check"])

    # Assert
    assert exit_code == 1
    assert _generated_contents(tmp_path) == before


def test_should_reject_duplicate_record_identifiers(tmp_path: Path) -> None:
    """Duplicate ADR identifiers fail before any output is generated."""
    # Arrange
    _write_source_tree(tmp_path)
    _write(
        tmp_path / "development/adrs/ADR-001-duplicate.md",
        """---
id: "ADR-001"
title: "Duplicate"
status: "proposed"
description: "Duplicates the stable identifier."
supersedes: []
superseded_by: []
---
""",
    )

    # Act
    exit_code = main(["--root", str(tmp_path), "--write"])

    # Assert
    assert exit_code == 1
    assert not any((tmp_path / path).exists() for path in GENERATED_FILES)


@pytest.mark.parametrize(
    ("source_path", "content", "expected_message"),
    [
        (
            "development/adrs/ADR-002-invalid.md",
            """---
id: "ADR-002"
title: "Missing description"
status: "accepted"
supersedes: []
superseded_by: []
---
""",
            "description",
        ),
        (
            "skills/ce-invalid/SKILL.md",
            """---
name: ce-other
description: The name does not match the directory.
---
""",
            "directory",
        ),
    ],
)
def test_should_reject_invalid_metadata_when_checking(
    tmp_path: Path, source_path: str, content: str, expected_message: str, capsys
) -> None:
    """Invalid ADR or skill metadata produces a precise validation error."""
    # Arrange
    _write_source_tree(tmp_path)
    _write(tmp_path / source_path, content)

    # Act
    exit_code = main(["--root", str(tmp_path), "--check"])

    # Assert
    assert exit_code == 1
    assert expected_message in capsys.readouterr().out.lower()


def test_should_separate_superseded_decisions_from_active_records(tmp_path: Path) -> None:
    """Historical ADRs are listed separately from active and proposed records."""
    # Arrange
    _write_source_tree(tmp_path)
    _write(
        tmp_path / "development/adrs/superseded ADR-000-old.md",
        """---
id: "ADR-000"
title: "Old ADR"
status: "superseded"
description: "Historical architecture."
supersedes: []
superseded_by: ["ADR-001"]
---

# ADR-000: Old ADR
""",
    )

    # Act
    exit_code = main(["--root", str(tmp_path), "--write"])

    # Assert
    index = (tmp_path / "development/adrs/README.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert "## Active decisions" in index
    assert "## Proposed decisions" in index
    assert "## Historical decisions" in index
    assert index.index("ADR-000") > index.index("## Historical decisions")
    assert "superseded%20ADR-000-old.md" in index


def test_should_reject_invalid_supersession_references(tmp_path: Path) -> None:
    """Supersession links must target another record or an existing path."""
    # Arrange
    _write_source_tree(tmp_path)
    _write(
        tmp_path / "development/standards/STD-002-invalid.md",
        """---
id: "STD-002"
title: "Invalid relationship"
status: "superseded"
description: "References a missing successor."
supersedes: []
superseded_by: ["STD-999"]
---
""",
    )

    # Act
    exit_code = main(["--root", str(tmp_path), "--check"])

    # Assert
    assert exit_code == 1


def test_should_reject_skill_directory_without_canonical_source(tmp_path: Path) -> None:
    """Every canonical skill directory must contain a SKILL.md source."""
    # Arrange
    _write_source_tree(tmp_path)
    (tmp_path / "skills/ce-incomplete").mkdir()

    # Act
    exit_code = main(["--root", str(tmp_path), "--check"])

    # Assert
    assert exit_code == 1
