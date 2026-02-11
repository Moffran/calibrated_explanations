from __future__ import annotations
import json
import runpy
import io
import contextlib
from pathlib import Path
import yaml
import pkgutil
import importlib
import warnings
import calibrated_explanations

# Import all submodules where possible to exercise top-level code and
# increase coverage for the examples CI job. Failures are warned, not
# fatal — optional extras may be missing in the examples runner.
for finder, mod_name, ispkg in pkgutil.walk_packages(
    calibrated_explanations.__path__, calibrated_explanations.__name__ + "."
):
    # Skip deprecated shims to avoid noise in test logs
    if any(
        mod_name.startswith(p)
        for p in [
            "calibrated_explanations.core.calibration.",
            "calibrated_explanations.perf.",
        ]
    ):
        continue
    try:
        importlib.import_module(mod_name)
    except Exception as exc:  # pragma: no cover - best-effort imports
        warnings.warn(f"examples: failed to import {mod_name}: {exc}", UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples" / "use_cases"


def run_script(path: Path) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(path), run_name="__main__")
    return buf.getvalue()


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
