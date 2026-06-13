from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


SCRIPT_PATH = Path("scripts/quality/check_parameter_naming.py")
PARAMETER_REFERENCE_DOC = Path("docs/foundations/concepts/parameter-reference.md")
REQUIRED_PARAM_HEADINGS = [
    "## `threshold`",
    "## `low_high_percentiles`",
    "## `confidence`",
    "## `confidence_level`",
    "## `significance`",
]


def run_checker(root: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT_PATH), "--root", str(root)]
    if check:
        cmd.append("--check")
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def test_no_banned_names_in_public_signatures() -> None:
    from scripts.quality.check_parameter_naming import BANNED_PUBLIC_PARAM_NAMES

    assert len(BANNED_PUBLIC_PARAM_NAMES) > 0, (
        "BANNED_PUBLIC_PARAM_NAMES is empty — _load_banned_names() failed to parse "
        "REMOVED_ALIAS_MAP from api/params.py. The check would pass vacuously."
    )
    src_root = Path("src/calibrated_explanations")
    result = run_checker(src_root, check=True)
    assert result.returncode == 0, (
        f"Banned parameter names found in src/:\n{result.stdout}\n{result.stderr}"
    )


def test_checker_detects_banned_name_in_synthetic_module(tmp_path: Path) -> None:
    module = tmp_path / "bad_module.py"
    module.write_text(
        textwrap.dedent("""\
            def explain(alpha=0.05):
                pass
        """),
        encoding="utf-8",
    )
    result = run_checker(tmp_path, check=True)
    assert result.returncode == 1
    assert "alpha" in result.stdout
    assert "explain()" in result.stdout


def test_checker_passes_on_clean_module(tmp_path: Path) -> None:
    module = tmp_path / "clean_module.py"
    module.write_text(
        textwrap.dedent("""\
            def explain(low_high_percentiles=(5, 95)):
                pass
        """),
        encoding="utf-8",
    )
    result = run_checker(tmp_path, check=True)
    assert result.returncode == 0, result.stdout


def test_parameter_reference_doc_exists() -> None:
    assert PARAMETER_REFERENCE_DOC.exists(), (
        f"Parameter reference doc not found: {PARAMETER_REFERENCE_DOC}"
    )


def test_parameter_reference_covers_required_parameters() -> None:
    content = PARAMETER_REFERENCE_DOC.read_text(encoding="utf-8")
    missing = [h for h in REQUIRED_PARAM_HEADINGS if h not in content]
    assert not missing, (
        f"Parameter reference doc is missing headings: {missing}"
    )
