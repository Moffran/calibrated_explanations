from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples" / "use_cases"


def run_script(path: Path) -> str:
    process = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return process.stdout


def test_minimal_quickstart() -> None:
    output = run_script(EXAMPLES_DIR / "minimal_quickstart.py")
    payload = json.loads(output)
    assert "factual_table" in payload
    assert "probability_interval" in payload
    interval = payload["probability_interval"]
    assert "low" in interval and "high" in interval


def test_metadata_json_schema() -> None:
    metadata_path = REPO_ROOT / "METADATA.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["name"] == "calibrated-explanations"
    assert metadata["public_api"] == ["WrapCalibratedExplainer"]
    assert metadata["required_calibration"] is True


def test_examples_index_entries() -> None:
    index_path = REPO_ROOT / "EXAMPLES_INDEX.json"
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    assert entries, "EXAMPLES_INDEX.json should list example scripts"
    for entry in entries:
        example_path = EXAMPLES_DIR / entry["file"].split("use_cases/")[-1]
        assert example_path.exists(), f"Missing example: {example_path}"
        content = example_path.read_text(encoding="utf-8")
        assert "WrapCalibratedExplainer" in content


def test_tool_description_yaml() -> None:
    tool_path = REPO_ROOT / ".ai" / "tool_description.yaml"
    tool = yaml.safe_load(tool_path.read_text(encoding="utf-8"))
    assert tool["tool_name"] == "calibrated_explanations"
    assert "WrapCalibratedExplainer" in tool["public_api"]
    assert tool["required_calibration"] is True


def test_quick_api_mentions_wrap() -> None:
    quick_api_path = REPO_ROOT / "QUICK_API.md"
    content = quick_api_path.read_text(encoding="utf-8")
    assert "WrapCalibratedExplainer" in content
