from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/quality/check_no_local_paths_in_reports.py")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_checker(root: Path, *paths: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    report = root / "reports" / "quality" / "no_local_paths_report.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--repo-root",
        str(root),
        "--report",
        str(report),
        *paths,
    ]
    if check:
        command.append("--check")
    return subprocess.run(command, check=False, capture_output=True, text=True)


def test_checker_passes_for_relative_and_http_paths(tmp_path: Path) -> None:
    write(
        tmp_path / "reports" / "quality" / "ok.json",
        json.dumps(
            {
                "path": "src/calibrated_explanations/plugins/registry.py",
                "url": "https://moffran.example/files/report.json",
            },
            indent=2,
        ),
    )

    result = run_checker(tmp_path)

    assert result.returncode == 0
    assert "No local absolute paths detected" in result.stdout


def test_checker_fails_for_windows_drive_paths(tmp_path: Path) -> None:
    write(
        tmp_path / "reports" / "quality" / "bad.json",
        json.dumps({"package_root": "C:/Users/example/repo/src/calibrated_explanations"}, indent=2),
    )

    result = run_checker(tmp_path)

    assert result.returncode == 1
    report = json.loads((tmp_path / "reports/quality/no_local_paths_report.json").read_text(encoding="utf-8"))
    assert report["total_violations"] == 1
    assert report["violations"][0]["category"] == "windows_drive"
    assert report["violations"][0]["json_path"] == "$.package_root"


def test_checker_fails_for_unc_paths(tmp_path: Path) -> None:
    write(
        tmp_path / "reports" / "quality" / "bad.txt",
        r"artifact=\\server\share\team\report.json",
    )

    result = run_checker(tmp_path)

    assert result.returncode == 1
    assert "[unc_path]" in result.stdout


def test_checker_fails_for_unix_absolute_paths(tmp_path: Path) -> None:
    write(
        tmp_path / "reports" / "quality" / "bad.json",
        json.dumps({"cwd": "/home/runner/work/repo"}, indent=2),
    )

    result = run_checker(tmp_path)

    assert result.returncode == 1
    report = json.loads((tmp_path / "reports/quality/no_local_paths_report.json").read_text(encoding="utf-8"))
    assert report["violations"][0]["category"] == "unix_absolute"


def test_checker_ignores_non_path_colons(tmp_path: Path) -> None:
    write(tmp_path / "reports" / "quality" / "ok.txt", "status: ready\nratio: 1:2\n")

    result = run_checker(tmp_path)

    assert result.returncode == 0


def test_checker_scans_tracked_debug_artifact_outside_reports(tmp_path: Path) -> None:
    write(tmp_path / ".pytest_matplotlib_debug.json", json.dumps({"sys_path": [r"C:\Users\alice\repo"]}, indent=2))

    result = run_checker(tmp_path)

    assert result.returncode == 1
    report = json.loads((tmp_path / "reports/quality/no_local_paths_report.json").read_text(encoding="utf-8"))
    assert report["violations"][0]["artifact"] == ".pytest_matplotlib_debug.json"
