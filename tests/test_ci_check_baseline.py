"""Tests for scripts/ci_check_baseline.py baseline gating.

We load the script as a module via importlib to avoid depending on 'scripts'
being an installed package, and we stub its tool runners to avoid invoking
external binaries (ruff/mypy) in test.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(unique_name: str = "ci_check_baseline_tests"):
    """Dynamically load the CI helper script as a Python module.

    We avoid importing external tools (ruff/mypy) by stubbing the script's
    functions later; this loader only executes the file so we can call its
    pure-Python helpers (`ruff_diagnostics`, `mypy_diagnostics`, `main`).
    """
    script = Path("scripts/ci_check_baseline.py").resolve()
    spec = importlib.util.spec_from_file_location(unique_name, script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_ruff_diagnostics_parsing(monkeypatch):
    """Parse a minimal ruff JSON payload into stable diagnostic keys.

    Verifies that the helper:
      - extracts code, filename, line and column
      - normalizes paths relative to repo root
      - returns a set of strings suitable for baseline comparison
    """
    mod = _load_module("ci_check_baseline_ruff_parse")

    # Stub run_cmd to return a ruff JSON payload with two diagnostics
    payload = (
        "[\n"
        "  {\n"
        "    \"filename\": \"src/foo.py\",\n"
        "    \"code\": \"E402\",\n"
        "    \"location\": {\"row\": 10, \"column\": 5}\n"
        "  },\n"
        "  {\n"
        "    \"filename\": \"src/bar.py\",\n"
        "    \"code\": \"F401\",\n"
        "    \"location\": {\"row\": 1, \"column\": 1}\n"
        "  }\n"
        "]\n"
    )
    monkeypatch.setattr(mod, "run_cmd", lambda cmd: payload)

    root = Path.cwd()
    # paths list is passed through to ruff, but for parsing we only care about JSON
    out = mod.ruff_diagnostics(root, ["src"])  # paths are not used by parser
    assert f"E402:src/foo.py:10:5" in out
    assert f"F401:src/bar.py:1:1" in out


def test_mypy_diagnostics_parsing(monkeypatch):
    """Parse a minimal mypy JSON payload into stable diagnostic keys.

    Ensures we include code, relative path, line, column and text, because
    mypy can emit multiple messages per file and we want stable keys.
    """
    mod = _load_module("ci_check_baseline_mypy_parse")

    # Minimal mypy JSON format with one file having two messages
    payload = (
        "{\n"
        "  \"errors\": [\n"
        "    {\n"
        "      \"path\": \"src/pkg/file.py\",\n"
        "      \"messages\": [\n"
        "        {\n"
        "          \"code\": \"arg-type\",\n"
        "          \"line\": 3, \"column\": 7,\n"
        "          \"message\": \"Argument 1 to f has incompatible type\"\n"
        "        },\n"
        "        {\n"
        "          \"code\": \"return-value\",\n"
        "          \"line\": 5, \"column\": 1,\n"
        "          \"message\": \"Function is missing return statement\"\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )
    monkeypatch.setattr(mod, "run_cmd", lambda cmd: payload)
    root = Path.cwd()
    out = mod.mypy_diagnostics(root, ["src/pkg/file.py"])
    assert any(s.startswith("arg-type:src/pkg/file.py:3:7:") for s in out)
    assert any(s.startswith("return-value:src/pkg/file.py:5:1:") for s in out)


def test_main_creates_baseline_when_missing(monkeypatch, capsys):
    """First run: writes baseline file and exits non-zero to prompt commit.

    We stub diagnostics to a fixed set and ensure the script creates the
    requested baseline file and prints an informational message to stderr.
    """
    mod = _load_module("ci_check_baseline_main_create")
    # Stub diagnostics to a deterministic set
    monkeypatch.setattr(mod, "ruff_diagnostics", lambda root, paths: {"E123:a.py:1:1"})

    baseline_rel = Path("ci/test_baselines/ruff_create.txt")
    baseline_path = (Path(__file__).resolve().parents[1] / baseline_rel).resolve()
    if baseline_path.exists():
        baseline_path.unlink()
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    if baseline_path.exists():
        baseline_path.unlink()

    monkeypatch.setattr(
        mod.sys, "argv",
        [
            "prog", "--tool", "ruff", "--baseline", str(baseline_rel), "--paths", "src",
        ],
    )
    rc = mod.main()
    stderr = capsys.readouterr().err
    assert rc == 1
    assert baseline_path.exists()
    assert "Baseline created" in stderr
    # Cleanup
    baseline_path.unlink(missing_ok=True)


def test_main_detects_new_issues_and_writes_current(monkeypatch, capsys):
    """Regression: non-empty diff from baseline fails and writes snapshot.

    We seed a baseline with one issue and have diagnostics return an extra
    one. The script should write `current_ruff.txt` for debugging and
    exit with non-zero status.
    """
    mod = _load_module("ci_check_baseline_main_regress")
    # Existing + new issue
    current = {"E123:a.py:1:1", "E124:a.py:2:1"}
    monkeypatch.setattr(mod, "ruff_diagnostics", lambda root, paths: current)

    baseline_rel = Path("ci/test_baselines/ruff_regress.txt")
    baseline_path = (Path(__file__).resolve().parents[1] / baseline_rel).resolve()
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text("E123:a.py:1:1\n")

    monkeypatch.setattr(
        mod.sys, "argv",
        ["prog", "--tool", "ruff", "--baseline", str(baseline_rel), "--paths", "src"],
    )
    rc = mod.main()
    stderr = capsys.readouterr().err
    assert rc == 1
    assert "New ruff issues detected" in stderr
    # current snapshot should be written
    current_path = baseline_path.parent / "current_ruff.txt"
    assert current_path.exists()
    text = current_path.read_text()
    for line in sorted(current):
        assert line in text
    # Cleanup
    baseline_path.unlink(missing_ok=True)
    current_path.unlink(missing_ok=True)


def test_main_passes_when_no_new_issues(monkeypatch, capsys):
    """No regressions: subset of baseline passes with zero exit code.

    If current diagnostics are equal to or a subset of the baseline, the
    script should not fail the run.
    """
    mod = _load_module("ci_check_baseline_main_pass")
    current = {"E123:a.py:1:1"}
    monkeypatch.setattr(mod, "ruff_diagnostics", lambda root, paths: current)

    baseline_rel = Path("ci/test_baselines/ruff_pass.txt")
    baseline_path = (Path(__file__).resolve().parents[1] / baseline_rel).resolve()
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text("E123:a.py:1:1\nE999:legacy.py:1:1\n")

    monkeypatch.setattr(
        mod.sys, "argv",
        ["prog", "--tool", "ruff", "--baseline", str(baseline_rel), "--paths", "src"],
    )
    rc = mod.main()
    assert rc == 0
    # Cleanup
    baseline_path.unlink(missing_ok=True)
