from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/quality/compare_parity_snapshots.py")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_compare_parity_snapshots_passes_when_equal(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    payload = {"value": [1.0, 2.0], "nested": {"ok": True}}
    write_json(left / "sample.json", payload)
    write_json(right / "sample.json", payload)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--left", str(left), "--right", str(right)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Parity snapshots match." in result.stdout


def test_compare_parity_snapshots_fails_on_diff(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    write_json(left / "sample.json", {"score": 0.5})
    write_json(right / "sample.json", {"score": 0.8})

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--left", str(left), "--right", str(right)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Detected" in result.stdout
