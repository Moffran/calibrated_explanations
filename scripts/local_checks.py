"""Run local verification profiles in the current environment.

This runner preserves focused standalone lanes (ADR-030 ratification,
deprecation closure, workflow run-block smoke) and adds explicit local
verification profiles for quick, task, PR, full, and release validation.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
import venv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 (dev extras provide tomli)
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class Step:
    """A single local check step."""

    name: str
    command: list[str]
    optional: bool = False


@dataclass(frozen=True)
class ProfilePlan:
    """A concrete local-check execution plan for one profile."""

    profile: str
    steps: list[Step]
    skipped_heavy_gates: list[dict[str, str]]
    task: int | None = None


def _python_cmd(*args: str) -> list[str]:
    return [sys.executable, *args]


def _dependency_audit_command() -> list[str]:
    """Return the preferred dependency-audit command for this environment."""
    if shutil.which("pip-audit") is not None:
        return [
            "pip-audit",
            "-r",
            "requirements.txt",
            "-r",
            "docs/requirements-doc.txt",
            "--ignore-vuln",
            "GHSA-xm59-rqc7-hhvf",
        ]
    if importlib.util.find_spec("pip_audit") is not None:
        return _python_cmd(
            "-m",
            "pip_audit",
            "-r",
            "requirements.txt",
            "-r",
            "docs/requirements-doc.txt",
            "--ignore-vuln",
            "GHSA-xm59-rqc7-hhvf",
        )
    return [
        "pip-audit",
        "-r",
        "requirements.txt",
        "-r",
        "docs/requirements-doc.txt",
        "--ignore-vuln",
        "GHSA-xm59-rqc7-hhvf",
    ]


def _is_python_build_module_step(command: list[str]) -> bool:
    """Return True when a step invokes the PyPA ``build`` module."""
    return len(command) >= 3 and command[0] == sys.executable and command[1:3] == ["-m", "build"]


def _is_pre_commit_step(step: Step) -> bool:
    if not step.command:
        return False
    head = Path(step.command[0]).name.lower()
    return head in {"pre-commit", "pre-commit.exe"}


def _mypy_strict_override_modules() -> list[str]:
    """Return the dotted module names declared in the pyproject strict-mypy override.

    This is the single source of truth for the "Phase 1B" mypy scope; both
    :func:`_mypy_targets` and CI derive their file list from it so the
    declared and enforced scopes cannot drift (v0.11.6 Task 58 / pre-v5 M2).
    """
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    for override in payload.get("tool", {}).get("mypy", {}).get("overrides", []):
        modules = override.get("module")
        if modules:
            return list(modules)
    return []


def _mypy_targets() -> list[str]:
    candidates = [
        "src/" + module.replace(".", "/") + ".py" for module in _mypy_strict_override_modules()
    ]
    return [path for path in candidates if Path(path).is_file()]


def _local_checks_coverage_file() -> str:
    """Return the isolated coverage-data filename shared by writer and reader steps.

    Stable for the lifetime of this process, so any step that writes or reads
    coverage data during this run resolves to the same path.
    """
    return f".coverage.local_checks.{os.getpid()}"


_COVERAGE_WRITER_STEPS = {
    "Core tests with coverage",
    "Over-testing coverage contexts",
}
_COVERAGE_READER_STEPS = {
    "Redundant tests report",
}


def _run_step(step: Step) -> int:
    repo_root = Path.cwd()
    command = list(step.command)
    run_cwd: Path | None = None
    if _is_python_build_module_step(command):
        # Avoid importing the repository's top-level ``build/`` directory as a namespace
        # package instead of the installed PyPA ``build`` tool.
        if len(command) == 3:
            command.append(str(repo_root))
        run_cwd = Path(tempfile.mkdtemp(prefix="ce-local-check-build-"))
    cmd_text = " ".join(command)
    print(f"\n[{step.name}]")
    print(f"$ {cmd_text}")
    env = dict(os.environ)
    env.setdefault("PRE_COMMIT_HOME", str(Path(".cache/pre-commit").resolve()))
    if step.name in _COVERAGE_WRITER_STEPS or step.name in _COVERAGE_READER_STEPS:
        env.setdefault("COVERAGE_FILE", _local_checks_coverage_file())
    if step.name in {
        "Core tests (no viz/no cov)",
        "Core tests with coverage",
        "Unit tests (fast/no viz/no slow/no cov)",
        "All non-viz tests (no coverage)",
    }:
        env.pop("CE_DEPRECATIONS", None)
    result: subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]
    try:
        if _is_pre_commit_step(step):
            result = subprocess.run(  # noqa: S603
                command,
                check=False,
                env=env,
                cwd=run_cwd,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.stderr:
                print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
        else:
            result = subprocess.run(command, check=False, env=env, cwd=run_cwd)  # noqa: S603
    except FileNotFoundError as exc:
        if step.optional:
            print(f"Step skipped (optional command unavailable): {step.name} ({exc})")
            return 0
        print(f"ERROR: required command unavailable for step '{step.name}': {exc}")
        return 127
    if result.returncode == 0:
        return 0
    if _is_pre_commit_step(step):
        combined = f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}"
        if _is_network_fetch_failure(combined):
            print(
                "Pre-commit could not fetch hook repos (offline/network-restricted). Continuing with stacked checks."
            )
            return 0
    if step.optional:
        print(f"Step failed but is advisory/optional: {step.name} (rc={result.returncode})")
        return 0
    return result.returncode


def _venv_python(venv_path: Path) -> Path:
    """Return the Python executable path for a virtual environment."""
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _run_timed_command(command: list[str]) -> tuple[int, int]:
    """Run a command and return ``(returncode, elapsed_seconds)``."""
    start = time.monotonic()
    result = subprocess.run(command, check=False)  # noqa: S603
    elapsed = int(round(time.monotonic() - start))
    return result.returncode, elapsed


def _run_uv_install_smoke() -> int:
    """Reproduce the CI uv install smoke and timing lane locally."""
    uv_binary = shutil.which("uv")
    if uv_binary is None:
        print("ERROR: uv not found. Install uv before running the optional uv install smoke.")
        return 127

    run_dir = Path(tempfile.mkdtemp(prefix="ce-uv-install-smoke-")).resolve()
    pip_venv = run_dir / "venv-pip"
    uv_venv = run_dir / "venv-uv"
    timing_report = Path("reports/ci/uv_install_timing.txt")
    smoke_python = "3.11"

    print("\n[uv install smoke and timing]")
    print(f"Working directory: {run_dir}")
    print(f"Provisioning smoke envs with Python {smoke_python} via uv")
    timing_report.parent.mkdir(parents=True, exist_ok=True)

    for venv_path in (pip_venv, uv_venv):
        rc = subprocess.run(  # noqa: S603
            [uv_binary, "venv", "--python", smoke_python, "--seed", str(venv_path)],
            check=False,
        ).returncode
        if rc != 0:
            return rc

    pip_python = _venv_python(pip_venv)
    uv_python = _venv_python(uv_venv)
    binary_only = "--only-binary=numpy,scipy,scikit-learn"

    pip_rc, pip_seconds = _run_timed_command(
        [
            str(pip_python),
            "-m",
            "pip",
            "install",
            binary_only,
            "-e",
            ".[dev]",
            "-c",
            "constraints.txt",
        ]
    )
    if pip_rc != 0:
        return pip_rc

    uv_rc, uv_seconds = _run_timed_command(
        [
            uv_binary,
            "pip",
            "install",
            "--python",
            str(uv_python),
            binary_only,
            "-e",
            ".[dev]",
            "-c",
            "constraints.txt",
        ]
    )
    if uv_rc != 0:
        return uv_rc

    smoke = subprocess.run(  # noqa: S603
        [
            str(uv_python),
            "-c",
            (
                "from importlib.metadata import version; "
                "import calibrated_explanations; "
                "from calibrated_explanations import WrapCalibratedExplainer; "
                "print(calibrated_explanations.__name__); "
                "print(WrapCalibratedExplainer.__name__); "
                "print(version('calibrated-explanations'))"
            ),
        ],
        check=False,
    )
    if smoke.returncode != 0:
        return smoke.returncode

    timing_report.write_text(
        f"pip_install_seconds={pip_seconds}\nuv_install_seconds={uv_seconds}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(timing_report.read_text(encoding="utf-8"), end="")
    return 0


def adr030_ratification_steps() -> list[Step]:
    """Return the focused ADR-030 ratification gate sequence."""
    return [
        Step(
            "Private-member scan",
            ["python", "scripts/anti-pattern-analysis/scan_private_usage.py", "tests", "--check"],
        ),
        Step(
            "ADR-030 anti-pattern detector",
            [
                "python",
                "scripts/anti-pattern-analysis/detect_test_anti_patterns.py",
                "--tests-dir",
                "tests",
                "--check",
                "--output",
                "reports/anti-pattern-analysis/test_anti_pattern_report.csv",
                "--report",
                "reports/anti-pattern-analysis/test_quality_report.json",
                "--baseline",
                ".github/test-quality-baseline.json",
            ],
        ),
        Step(
            "ADR-030 test-helper export guard",
            [
                "python",
                "scripts/quality/check_no_test_helper_exports.py",
                "--root",
                "src/calibrated_explanations",
                "--report",
                "reports/anti-pattern-analysis/test_helper_wrapper_report.json",
            ],
        ),
        Step(
            "ADR-030 marker hygiene",
            [
                "python",
                "scripts/quality/check_marker_hygiene.py",
                "--check",
                "--report",
                "reports/marker-hygiene/marker_hygiene_report.json",
                "--baseline",
                ".github/marker-hygiene-baseline.json",
            ],
        ),
        Step(
            "Generated report local-path guard",
            [
                "python",
                "scripts/quality/check_no_local_paths_in_reports.py",
                "--check",
                "--report",
                "reports/quality/no_local_paths_report.json",
            ],
        ),
        Step(
            "Class-surface allowlist gate",
            [
                "python",
                "scripts/quality/check_class_surface_allowlist.py",
                "--check",
                "--report",
                "reports/quality/class_surface_allowlist_report.json",
            ],
        ),
    ]


def adr030_expected_reports() -> list[Path]:
    """Return reports that the ADR-030 ratification lane must produce."""
    return [
        Path("reports/anti-pattern-analysis/private_usage_scan.csv"),
        Path("reports/anti-pattern-analysis/test_anti_pattern_report.csv"),
        Path("reports/anti-pattern-analysis/test_quality_report.json"),
        Path("reports/anti-pattern-analysis/test_helper_wrapper_report.json"),
        Path("reports/marker-hygiene/marker_hygiene_report.json"),
        Path("reports/quality/no_local_paths_report.json"),
        Path("reports/quality/class_surface_allowlist_report.json"),
    ]


def _utc_now_iso() -> str:
    """Return the current UTC timestamp for generated local-check reports."""
    return datetime.now(timezone.utc).isoformat()


def _command_text(command: list[str]) -> str:
    """Return a repo-relative command string for timing reports."""
    display_command = list(command)
    if display_command and Path(display_command[0]) == Path(sys.executable):
        display_command[0] = "python"
    return " ".join(display_command)


def _write_adr030_timing_report(
    records: list[dict[str, object]], started_at: float, output_path: Path
) -> None:
    """Write the focused ADR-030 ratification timing report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "steps": records,
        "total_elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def _active_deprecation_rows(ledger_path: Path) -> list[dict[str, str]]:
    """Return data rows from the Active deprecations table."""
    text = ledger_path.read_text(encoding="utf-8")
    try:
        active_section = text.split("### Active deprecations", 1)[1].split(
            "### Removed deprecations (history)",
            1,
        )[0]
    except IndexError:
        raise RuntimeError("Could not locate Active deprecations ledger section") from None

    rows: list[dict[str, str]] = []
    for line in active_section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("|---") or "Deprecated symbol" in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 5:
            continue
        rows.append(
            {
                "deprecated_symbol": cells[0],
                "replacement": cells[1],
                "deprecated_since": cells[2],
                "removal_eta": cells[3],
                "notes": cells[4],
            }
        )
    return rows


def _ledger_version(text: str) -> tuple[int, ...] | None:
    """Return the numeric release tuple of a ledger cell such as ``v1.3.0``."""
    try:
        from packaging.version import InvalidVersion, Version  # noqa: PLC0415
    except ImportError:  # pragma: no cover - packaging ships with pip/setuptools
        return None
    try:
        return Version(text.strip().strip("`").lstrip("vV")).release
    except InvalidVersion:
        return None


def _current_package_version() -> tuple[int, ...] | None:
    """Return the release tuple of the working tree, or of the installed package."""
    raw: str | None = None
    if Path("pyproject.toml").is_file():
        raw = _pyproject_release_version()
    else:
        try:
            from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

            raw = version("calibrated-explanations")
        except PackageNotFoundError:
            return None
    return _ledger_version(raw)


def _deprecation_blocking_reason(
    row: dict[str, str], current: tuple[int, ...] | None
) -> str | None:
    """Return why an Active ledger row blocks under ADR-011, or ``None``.

    v1.0.0 shipped with zero active deprecations (ADR-011 finalization
    exception). After it the default policy applies: a row needs a parseable
    ``Deprecated since`` and ``Removal ETA``; the ETA must leave at least two
    minor releases (or be a later major); and the ETA must not be overdue for
    the current version.
    """
    since = _ledger_version(row["deprecated_since"])
    eta = _ledger_version(row["removal_eta"])
    if since is None or eta is None:
        return "unparsable 'Deprecated since' or 'Removal ETA' version"
    since_major, since_minor = (since + (0, 0))[:2]
    eta_major, eta_minor = (eta + (0, 0))[:2]
    if eta_major == since_major and eta_minor < since_minor + 2:
        return "removal ETA is inside the two-minor deprecation window"
    if eta_major < since_major:
        return "removal ETA precedes the deprecation"
    if current is None:
        return "current package version is unknown"
    if current >= eta:
        return "removal ETA has been reached; remove the deprecated surface"
    return None


def _write_active_deprecations_report(
    rows: list[dict[str, str]],
    output_path: Path,
    current_version: tuple[int, ...] | None = None,
) -> int:
    """Write the active-deprecation ledger artifact and return its gate code.

    Active rows are reported. A row blocks only when it breaks the ADR-011
    lifecycle (see :func:`_deprecation_blocking_reason`), so deprecations
    introduced after v1.0.0 can stay active until their removal ETA.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reasons = {
        row["deprecated_symbol"]: reason
        for row in rows
        if (reason := _deprecation_blocking_reason(row, current_version)) is not None
    }
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "status": "pass" if not reasons else "fail",
        "active_rows_count": len(rows),
        "active_symbols": [r["deprecated_symbol"] for r in rows],
        "blocking_rows_count": len(reasons),
        "blocking_symbols": list(reasons),
        "blocking_reasons": reasons,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    for row in rows:
        print(f"Active deprecation: {row['deprecated_symbol']} (ETA: {row['removal_eta']})")
    if reasons:
        print("ERROR: Active deprecations break the ADR-011 lifecycle:")
        for symbol, reason in reasons.items():
            print(f"  {symbol}: {reason}")
        return 1
    return 0


def _write_deprecation_closure_timing_report(
    records: list[dict[str, object]],
    started_at: float,
    output_path: Path,
) -> None:
    """Write the pre-v1.0 deprecation-closure timing report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "steps": records,
        "total_elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def deprecation_closure_steps() -> list[Step]:
    """Return the pre-v1.0 deprecation-closure validation sequence."""
    return [
        Step(
            "Focused deprecation closure tests",
            _python_cmd(
                "-m",
                "pytest",
                "tests/",
                "-k",
                "deprecat or lime or shap or reject or plugin or parallel or calibration",
                "-v",
                "--no-cov",
            ),
        ),
        Step(
            "ADR-030 ratification lane",
            _python_cmd("scripts/local_checks.py", "--adr030-ratification"),
        ),
    ]


def run_deprecation_ledger_gate() -> int:
    """Fail when an active deprecation breaks the ADR-011 lifecycle.

    Folded into the PR profile by v0.11.6 Task 60 (it replaces the removed
    ``deprecation-check.yml`` workflow). Until v1.0.0 it enforced zero active
    deprecations; from v1.1.0 active rows are allowed while they keep a valid
    two-minor removal ETA that has not been reached. The heavier focused-test
    lane remains available via ``--deprecation-closure``.
    """
    try:
        rows = _active_deprecation_rows(Path("docs/migration/deprecations.md"))
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    return _write_active_deprecations_report(
        rows,
        Path("reports/deprecations/active_deprecations_check.json"),
        _current_package_version(),
    )


def run_deprecation_closure() -> int:
    """Run the pre-v1.0 deprecation-closure lane and emit timing evidence."""
    ledger_report = Path("reports/deprecations/active_deprecations_check.json")
    timing_report = Path("reports/deprecations/deprecation_closure_timing.json")
    records: list[dict[str, object]] = []
    started_at = time.monotonic()

    step_started_at = time.monotonic()
    try:
        rows = _active_deprecation_rows(Path("docs/migration/deprecations.md"))
        rc = _write_active_deprecations_report(rows, ledger_report, _current_package_version())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        rc = 1
    records.append(
        {
            "name": "Active deprecations ledger check",
            "command": "parse docs/migration/deprecations.md",
            "exit_code": rc,
            "elapsed_seconds": round(time.monotonic() - step_started_at, 3),
        }
    )
    _write_deprecation_closure_timing_report(records, started_at, timing_report)
    if rc != 0:
        return rc

    for step in deprecation_closure_steps():
        step_started_at = time.monotonic()
        rc = _run_step(step)
        records.append(
            {
                "name": step.name,
                "command": _command_text(step.command),
                "exit_code": rc,
                "elapsed_seconds": round(time.monotonic() - step_started_at, 3),
            }
        )
        _write_deprecation_closure_timing_report(records, started_at, timing_report)
        if rc != 0:
            return rc

    return 0


def _validate_adr030_ratification_outputs(timing_report: Path) -> int:
    """Validate required ADR-030 ratification artifacts exist and are parseable."""
    missing_reports = [path.as_posix() for path in adr030_expected_reports() if not path.exists()]
    if missing_reports:
        print("ERROR: ADR-030 ratification lane did not produce expected reports:")
        for path in missing_reports:
            print(f"  {path}")
        return 1
    try:
        json.loads(timing_report.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: ADR-030 timing report is not valid JSON: {exc}")
        return 1
    return 0


def run_adr030_ratification() -> int:
    """Run the focused ADR-030 ratification lane and emit timing evidence."""
    timing_report = Path("reports/anti-pattern-analysis/adr030_ratification_timing.json")
    records: list[dict[str, object]] = []
    started_at = time.monotonic()

    for step in adr030_ratification_steps():
        step_started_at = time.monotonic()
        rc = _run_step(step)
        records.append(
            {
                "name": step.name,
                "command": _command_text(step.command),
                "exit_code": rc,
                "elapsed_seconds": round(time.monotonic() - step_started_at, 3),
            }
        )
        _write_adr030_timing_report(records, started_at, timing_report)
        if rc != 0:
            return rc

    return _validate_adr030_ratification_outputs(timing_report)


def _is_network_fetch_failure(stderr: str) -> bool:
    text = (stderr or "").lower()
    if "unable to access 'https://github.com" in text:
        return True
    if "failed to connect to github.com" in text:
        return True
    return "could not connect to server" in text


def _pytest_supports_no_cov() -> bool:
    """Return True if pytest supports the --no-cov option."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--help"], check=False, capture_output=True, text=True
    )
    return "--no-cov" in (result.stdout or "")


def _changed_python_targets() -> list[str]:
    """Return changed Python files for fast lint/format checks.

    Falls back to broad repo paths when git metadata is unavailable or when no
    changed Python files are detected.
    """
    default_targets = ["src", "tests", "scripts"]
    git = shutil.which("git")
    if git is None:
        return default_targets

    commands = [
        [
            git,
            "diff",
            "--name-only",
            "--relative",
            "--diff-filter=d",
            "HEAD",
            "--",
            "src",
            "tests",
            "scripts",
        ],
        [git, "ls-files", "--others", "--exclude-standard", "--", "src", "tests", "scripts"],
    ]
    changed: list[str] = []
    for command in commands:
        result = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603
        if result.returncode != 0:
            return default_targets
        for line in (result.stdout or "").splitlines():
            path = line.strip()
            if path.endswith(".py") and path not in changed:
                changed.append(path)
    return changed or default_targets


def _ruff_check_command(*paths: str) -> list[str]:
    """Return the Ruff lint command for the provided paths."""
    return _python_cmd("-m", "ruff", "check", *paths)


def _requires_ruff_preview_for_format(*paths: str) -> bool:
    """Return whether Ruff format should enable preview mode for the paths."""
    return any(Path(path).suffix.lower() == ".md" for path in paths)


def _ruff_format_check_command(*paths: str) -> list[str]:
    """Return the Ruff format-check command for the provided paths."""
    command = ["-m", "ruff", "format", "--check"]
    if _requires_ruff_preview_for_format(*paths):
        command.append("--preview")
    return _python_cmd(*command, *paths)


def _pytest_no_cov_command(*args: str) -> list[str]:
    """Return a pytest command and add ``--no-cov`` when supported."""
    command = [sys.executable, "-m", "pytest", *args]
    if _pytest_supports_no_cov():
        command.append("--no-cov")
    return command


def _quick_pytest_command() -> list[str]:
    """Return the inner-loop pytest command."""
    return _pytest_no_cov_command(
        "-q",
        "tests/unit",
        "-m",
        "not viz and not slow",
        "--maxfail=1",
        "-o",
        "addopts=",
    )


def _all_non_viz_pytest_command() -> list[str]:
    """Return the PR-scope pytest command."""
    return _pytest_no_cov_command("-q", "-m", "not viz", "-o", "addopts=")


def _quick_steps(mypy_targets: list[str], lint_targets: list[str]) -> list[Step]:
    """Return the inner-loop verification steps."""
    steps: list[Step] = []
    if lint_targets:
        steps.extend(
            [
                Step("Ruff check", _ruff_check_command(*lint_targets)),
                Step("Ruff format check", _ruff_format_check_command(*lint_targets)),
            ]
        )
    if mypy_targets:
        steps.append(
            Step(
                "Mypy (Phase 1B scope)",
                _python_cmd("-m", "mypy", *mypy_targets, "--config-file", "pyproject.toml"),
            )
        )
    steps.extend(
        [
            Step("ADR-001 boundary check", _python_cmd("scripts/quality/check_import_graph.py")),
            Step(
                "ADR-002 compliance check",
                _python_cmd("scripts/quality/check_adr002_compliance.py"),
            ),
            Step(
                "Instruction index consistency",
                _python_cmd("scripts/quality/sync_instruction_indexes.py", "--check"),
            ),
            Step(
                "Skill package consistency",
                _python_cmd("scripts/quality/sync_skill_packages.py", "--check"),
            ),
            Step("Unit tests (fast/no viz/no slow/no cov)", _quick_pytest_command()),
        ]
    )
    return steps


def _pr_steps(
    mypy_targets: list[str], lint_targets: list[str], *, pre_commit_available: bool
) -> list[Step]:
    """Return the PR-scope verification steps."""
    steps = list(_quick_steps(mypy_targets, lint_targets))
    steps.extend(
        [
            Step(
                "CI policy (full inventory)",
                _python_cmd("scripts/quality/validate_ci_policy.py", "--full-inventory"),
            ),
            Step(
                "Ruff naming enforcement (STD-001)",
                _python_cmd("-m", "ruff", "check", "--select", "N"),
            ),
            Step(
                "Notebook naming lint (STD-002 notebooks)",
                _python_cmd("-m", "nbqa", "ruff", "notebooks", "--select", "N"),
            ),
            Step(
                "pydocstyle (STD-002 baseline)",
                _python_cmd("-m", "pydocstyle", "src", "tests"),
            ),
            Step(
                "Agent instruction consistency",
                _python_cmd("scripts/quality/check_agent_instruction_consistency.py"),
            ),
            Step(
                "Deprecation ledger gate (ADR-011)",
                _python_cmd("scripts/local_checks.py", "--deprecation-ledger"),
            ),
            Step(
                "evaluation/ freeze guard (tracked files)",
                _python_cmd(
                    "-c",
                    (
                        "import subprocess, sys; "
                        "out = subprocess.run(['git', 'ls-files', 'evaluation'], "
                        "capture_output=True, text=True, check=True).stdout; "
                        "bad = [line for line in out.splitlines() "
                        "if line and not line.endswith('README.md')]; "
                        "print('\\n'.join(bad)); sys.exit(1 if bad else 0)"
                    ),
                ),
            ),
            Step(
                "Docstring coverage",
                _python_cmd("scripts/quality/check_docstring_coverage.py", "--fail-under", "94.0"),
            ),
            Step(
                "STD-005 logger domain enforcement",
                _python_cmd(
                    "scripts/quality/check_logging_domains.py",
                    "--root",
                    "src/calibrated_explanations",
                    "--report",
                    "reports/quality/logging_domain_report.json",
                ),
            ),
            Step(
                "STD-001 nomenclature enforcement",
                _python_cmd(
                    "scripts/quality/check_std001_nomenclature.py",
                    "--root",
                    "src/calibrated_explanations",
                    "--report",
                    "reports/nomenclature_violation_inventory.json",
                    "--check",
                ),
            ),
            Step(
                "Parameter naming CI guard (removed aliases)",
                _python_cmd(
                    "scripts/quality/check_parameter_naming.py",
                    "--root",
                    "src/calibrated_explanations",
                    "--check",
                ),
            ),
            Step(
                "Capability-chain validator",
                _python_cmd("scripts/quality/validate_capability_chain.py", "--check"),
            ),
            Step(
                "Raw evidence structural validation",
                _python_cmd("scripts/generate_tif_evidence.py", "--validate-existing"),
            ),
            Step(
                "Private-member scan",
                _python_cmd(
                    "scripts/anti-pattern-analysis/scan_private_usage.py", "tests", "--check"
                ),
            ),
            Step(
                "ADR-030 anti-pattern detector",
                _python_cmd(
                    "scripts/anti-pattern-analysis/detect_test_anti_patterns.py",
                    "--tests-dir",
                    "tests",
                    "--check",
                    "--output",
                    "reports/anti-pattern-analysis/test_anti_pattern_report.csv",
                    "--report",
                    "reports/anti-pattern-analysis/test_quality_report.json",
                    "--baseline",
                    ".github/test-quality-baseline.json",
                ),
            ),
            Step(
                "ADR-030 test-helper export guard",
                _python_cmd(
                    "scripts/quality/check_no_test_helper_exports.py",
                    "--root",
                    "src/calibrated_explanations",
                    "--report",
                    "reports/anti-pattern-analysis/test_helper_wrapper_report.json",
                ),
            ),
            Step(
                "Class-surface allowlist gate",
                _python_cmd(
                    "scripts/quality/check_class_surface_allowlist.py",
                    "--check",
                    "--report",
                    "reports/quality/class_surface_allowlist_report.json",
                ),
            ),
            Step(
                "ADR-006 trust-mutation primitive guard",
                _python_cmd(
                    "scripts/quality/check_trust_mutation_primitive.py",
                    "--root",
                    "src/calibrated_explanations",
                    "--report",
                    "reports/trust_mutation_inventory.json",
                    "--check",
                ),
            ),
            Step(
                "ADR-034 ConfigManager usage guard",
                _python_cmd(
                    "scripts/quality/check_config_manager_usage.py",
                    "--root",
                    "src/calibrated_explanations",
                    "--scope",
                    "runtime",
                    "--report",
                    "reports/quality/config_manager_usage_report.json",
                    "--check",
                ),
            ),
            Step(
                "ADR-030 marker hygiene",
                _python_cmd(
                    "scripts/quality/check_marker_hygiene.py",
                    "--check",
                    "--report",
                    "reports/marker-hygiene/marker_hygiene_report.json",
                    "--baseline",
                    ".github/marker-hygiene-baseline.json",
                ),
            ),
            Step(
                "Generated report local-path guard",
                _python_cmd(
                    "scripts/quality/check_no_local_paths_in_reports.py",
                    "--check",
                    "--report",
                    "reports/quality/no_local_paths_report.json",
                ),
            ),
            Step(
                "Forbidden doc-pattern gates",
                _python_cmd(
                    "scripts/quality/check_forbidden_doc_patterns.py",
                    "--all-checks",
                ),
            ),
            Step(
                "Release-profile meta-test guard",
                _python_cmd(
                    "scripts/quality/check_no_release_profile_meta_tests.py",
                    "--report",
                    "reports/quality/meta_test_guard.json",
                ),
            ),
            Step("All non-viz tests (no coverage)", _all_non_viz_pytest_command()),
        ]
    )
    if pre_commit_available:
        steps.append(Step("Pre-commit", ["pre-commit", "run", "--all-files"]))
    return steps


def _full_steps(
    mypy_targets: list[str], lint_targets: list[str], *, pre_commit_available: bool
) -> list[Step]:
    """Return the heavier main/full verification steps."""
    steps = list(_pr_steps(mypy_targets, lint_targets, pre_commit_available=pre_commit_available))
    steps.extend(
        [
            Step(
                "Docs build (HTML)",
                [sys.executable, "-m", "sphinx", "-b", "html", "docs", "docs/_build/html"],
            ),
            Step(
                "Core tests with coverage",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-o",
                    "addopts=",
                    "--cov=src/calibrated_explanations",
                    "--cov-config=pyproject.toml",
                    "--cov-report=xml:coverage.xml",
                    "--cov-context=test",
                    "--cov-fail-under=90",
                ],
            ),
            Step(
                "Per-module coverage gates",
                _python_cmd("scripts/quality/check_coverage_gates.py", "coverage.xml"),
            ),
            Step("Micro benchmark", _python_cmd("scripts/perf/micro_bench_perf.py")),
            Step(
                "Perf thresholds",
                _python_cmd(
                    "scripts/perf/check_perf_micro.py",
                    "tests/benchmarks/micro_current.json",
                    "tests/benchmarks/perf_thresholds.json",
                ),
            ),
            Step(
                "Over-testing coverage contexts",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-o",
                    "addopts=",
                    "--cov=src/calibrated_explanations",
                    "--cov-config=pyproject.toml",
                    "--cov-context=test",
                    "--no-cov-on-fail",
                ],
            ),
            Step(
                "Over-testing report",
                _python_cmd(
                    "scripts/over_testing/over_testing_report.py",
                    "--require-multiple-contexts",
                    "--coverage-file",
                    _local_checks_coverage_file(),
                    "--output-lines",
                    "reports/over_testing/line_coverage_counts.csv",
                    "--output-blocks",
                    "reports/over_testing/block_coverage_counts.csv",
                    "--output-summary",
                    "reports/over_testing/summary.json",
                    "--output-metadata",
                    "reports/over_testing/metadata.json",
                ),
            ),
            Step(
                "Redundant tests report",
                _python_cmd("scripts/over_testing/detect_redundant_tests.py"),
            ),
        ]
    )
    return steps


def _release_steps(
    mypy_targets: list[str], lint_targets: list[str], *, pre_commit_available: bool
) -> list[Step]:
    """Return the release-oriented verification steps."""
    steps = list(_full_steps(mypy_targets, lint_targets, pre_commit_available=pre_commit_available))
    steps.extend(
        [
            Step(
                "Dependency audit",
                _dependency_audit_command(),
            ),
            Step(
                "Notebook audit",
                _python_cmd(
                    "scripts/quality/audit_notebook_api.py",
                    "notebooks",
                    "--json",
                    "artifacts/notebook_audit.json",
                ),
            ),
            Step(
                "Notebook execution report",
                _python_cmd(
                    "scripts/docs/run_notebooks.py",
                    "--mode",
                    "advisory",
                    "--output",
                    "reports/docs/notebook_execution_report.json",
                ),
            ),
            Step(
                "Strict docs build",
                [sys.executable, "-m", "sphinx", "-W", "--keep-going", "docs", "docs/_build/html"],
            ),
            Step(
                "Docs linkcheck",
                [
                    sys.executable,
                    "-m",
                    "sphinx",
                    "-b",
                    "linkcheck",
                    "-D",
                    "nbsphinx_execute=never",
                    "docs",
                    "docs/_build/linkcheck",
                ],
            ),
            Step(
                "Warning policy",
                _python_cmd(
                    "scripts/quality/check_warning_policy.py",
                    "--check",
                    "--report",
                    "reports/quality/warning_policy.json",
                ),
            ),
            Step(
                "Deprecation closure",
                _python_cmd("scripts/local_checks.py", "--deprecation-closure"),
            ),
            Step(
                "Version alignment",
                _python_cmd(
                    "scripts/quality/check_version_alignment.py",
                    "--check",
                    "--allow-normalized",
                ),
            ),
            Step("Release packaging build", _python_cmd("-m", "build")),
            Step(
                "Release packaging artifact smoke",
                _python_cmd(
                    "scripts/quality/check_packaging_artifacts.py",
                    "--clean-dist",
                    "--report",
                    "reports/packaging/license_artifact_check.json",
                ),
            ),
            Step(
                "Capability evidence refresh",
                _python_cmd("scripts/generate_tif_evidence.py", "--check-current"),
            ),
            Step("Public API snapshot", _python_cmd("scripts/quality/snapshot_public_api.py")),
        ]
    )
    return steps


RELEASE_PREFLIGHT_REPORT = Path("reports/local_checks/release_preflight_report.json")
RELEASE_TRANSIENT_ASYNC_LOGS: tuple[Path, ...] = (
    Path("reports/local_checks/release_preflight_async.log"),
    Path("reports/local_checks/release_preflight_async.err.log"),
)
RELEASE_MANUAL_STEP_RANGE = tuple(range(11, 14))
RELEASE_POSTCOMMIT_STEP_RANGE = tuple(range(14, 18))
RELEASE_STANDARD_NOTEBOOK_PATTERNS: tuple[str, ...] = (
    "notebooks/quickstart.ipynb",
    "notebooks/quickstart_guarded.ipynb",
    "notebooks/quickstart_tiny.ipynb",
    "notebooks/core_demos/*.ipynb",
    "notebooks/miscellaneous/*.ipynb",
    "notebooks/paper_based/*.ipynb",
    "notebooks/advanced/demo_conditional.ipynb",
    "notebooks/advanced/demo_config_management.ipynb",
    "notebooks/advanced/demo_narrative_explanations.ipynb",
    "notebooks/advanced/demo_plugin_wiring.ipynb",
    "notebooks/advanced/demo_reject.ipynb",
    "notebooks/advanced/demo_under_the_hood.ipynb",
)
RELEASE_SLOW_NOTEBOOK_PATTERNS: tuple[str, ...] = (
    "notebooks/advanced/fast_feature_filtering_demo.ipynb",
)


def _resolve_release_notebook_paths(patterns: tuple[str, ...]) -> list[str]:
    """Resolve release notebook globs into deterministic repo-relative paths."""
    resolved_paths: list[str] = []
    missing_patterns: list[str] = []
    for pattern in patterns:
        if any(token in pattern for token in "*?["):
            matches = sorted(Path().glob(pattern))
        else:
            candidate = Path(pattern)
            matches = [candidate] if candidate.exists() else []
        if not matches:
            missing_patterns.append(pattern)
            continue
        for match in matches:
            resolved_paths.append(match.as_posix())
    if missing_patterns:
        joined = ", ".join(sorted(missing_patterns))
        raise RuntimeError(f"Release notebook patterns did not resolve to files: {joined}")
    return resolved_paths


def _purge_release_transient_async_logs() -> None:
    """Remove stale async release-preflight capture logs before governed checks.

    These files are ad hoc workspace captures rather than governed release
    artifacts. If left behind, they can leak local machine paths into the
    report-scanning lane and invalidate an otherwise clean release handoff.
    """
    for path in RELEASE_TRANSIENT_ASYNC_LOGS:
        if not path.exists():
            continue
        with suppress(OSError):
            path.unlink()
            print(f"Removed stale transient release log: {path.as_posix()}")
            continue
        with suppress(OSError):
            path.write_text(
                "Removed by scripts/local_checks.py before release validation.\n",
                encoding="utf-8",
                newline="\n",
            )
            print(f"Sanitized stale transient release log: {path.as_posix()}")


def _release_notebook_steps() -> list[Step]:
    """Return the strict in-place notebook execution steps for releases."""
    standard_paths = _resolve_release_notebook_paths(RELEASE_STANDARD_NOTEBOOK_PATTERNS)
    slow_paths = _resolve_release_notebook_paths(RELEASE_SLOW_NOTEBOOK_PATTERNS)
    steps = [
        Step(
            "Release notebooks (saved in-place)",
            _python_cmd(
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=600",
                *standard_paths,
            ),
        )
    ]
    if slow_paths:
        steps.append(
            Step(
                "Release slow notebook (saved in-place)",
                _python_cmd(
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    "--ExecutePreprocessor.timeout=5400",
                    *slow_paths,
                ),
            )
        )
    return steps


def _git_text(*args: str) -> str | None:
    """Return trimmed stdout from a git command, or ``None`` when unavailable."""
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(  # noqa: S603
        [git, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def _current_git_branch() -> str | None:
    """Return the current git branch, if available."""
    return _git_text("branch", "--show-current")


def _current_git_status_porcelain() -> str | None:
    """Return the current git status snapshot used by release-finalize."""
    return _git_text("status", "--short")


# The release-preflight report records its own change set inside itself, so
# its own path must be excluded from that change set or recording the
# fingerprint would change the file whose fingerprint was just recorded.
_FINGERPRINT_EXCLUDED_PATHS = frozenset({"reports/local_checks/release_preflight_report.json"})


def _parse_git_status_paths(status_text: str) -> set[str]:
    """Return changed/untracked paths from ``git status --porcelain`` output.

    Rename entries (``old -> new``) resolve to the new path. The
    release-preflight report's own path is dropped (see
    ``_FINGERPRINT_EXCLUDED_PATHS``).
    """
    paths: set[str] = set()
    for line in status_text.splitlines():
        if len(line) < 4:
            continue
        raw_path = line[3:].strip()
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        raw_path = raw_path.strip('"')
        if raw_path and raw_path not in _FINGERPRINT_EXCLUDED_PATHS:
            paths.add(raw_path)
    return paths


def _file_content_hash(path: str) -> str | None:
    """Return the sha256 hex digest of ``path``'s current bytes, or ``None`` if absent."""
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        return hashlib.sha256(file_path.read_bytes()).hexdigest()
    except OSError:
        return None


def _worktree_changed_file_hashes() -> dict[str, str | None]:
    """Return ``{path: content_hash}`` for every changed/untracked file.

    Unlike ``git status --short`` text, this survives a file moving between
    untracked/modified/staged/committed as long as its bytes do not change,
    which is what ``release-finalize`` needs to tolerate committing exactly
    what ``release-preflight`` produced before step 11.
    """
    status_text = _git_text("status", "--porcelain", "--untracked-files=all") or ""
    return {path: _file_content_hash(path) for path in sorted(_parse_git_status_paths(status_text))}


def _changed_paths_since(recorded_head_sha: str | None) -> set[str]:
    """Return every path touched since ``recorded_head_sha``, committed or not.

    Comparing only the *currently dirty* status against the recorded set
    misses changes that get committed on top of exactly what
    release-preflight produced: once committed, the tree is clean again and
    there is no dirty-status line left to flag as unexpected. Diffing from
    the ``HEAD`` recorded at preflight completion catches paths touched by
    any commit made since, in addition to whatever is still uncommitted.
    """
    paths = set(
        _parse_git_status_paths(_git_text("status", "--porcelain", "--untracked-files=all") or "")
    )
    current_head_sha = _git_text("rev-parse", "HEAD")
    if recorded_head_sha and current_head_sha and recorded_head_sha != current_head_sha:
        committed_diff = _git_text("diff", "--name-only", recorded_head_sha, current_head_sha) or ""
        paths.update(
            line.strip()
            for line in committed_diff.splitlines()
            if line.strip() and line.strip() not in _FINGERPRINT_EXCLUDED_PATHS
        )
    return paths


def _pyproject_release_version() -> str:
    """Return the raw project version from pyproject.toml."""
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def _canonical_version(version: str) -> str:
    """Return the PEP 440 canonical form of ``version``."""
    try:
        from packaging.version import Version  # noqa: PLC0415

        return str(Version(version))
    except Exception as exc:  # noqa: BLE001 - converted to a release-workflow error
        raise RuntimeError(f"Invalid release version {version!r}.") from exc


def _version_lineage(version: str) -> tuple[tuple[int, ...], str | None]:
    """Return the numeric release tuple and prerelease family for ``version``."""
    try:
        from packaging.version import Version  # noqa: PLC0415

        parsed = Version(version)
    except Exception as exc:  # noqa: BLE001 - converted to a release-workflow error
        raise RuntimeError(f"Invalid release version {version!r}.") from exc
    prerelease_family = parsed.pre[0] if parsed.pre is not None else None
    return parsed.release, prerelease_family


_PLAN_RELEASE_VERSION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\*\*Release version:\*\*\s*`(?P<version>[^`]+)`", re.IGNORECASE),
    re.compile(r"\*\*Release version:\s*`(?P<version>[^`]+)`", re.IGNORECASE),
    re.compile(r"\*\*RC version string:\s*`(?P<version>[^`]+)`", re.IGNORECASE),
)


def _release_version_from_plan(plan_path: Path) -> str:
    """Resolve the exact release version declared by a version plan."""
    text = plan_path.read_text(encoding="utf-8")
    for pattern in _PLAN_RELEASE_VERSION_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return _canonical_version(match.group("version"))

    filename_match = re.fullmatch(r"v(?P<version>.+)_plan\.md", plan_path.name)
    if filename_match is not None:
        return _canonical_version(filename_match.group("version"))

    project_version = _canonical_version(_pyproject_release_version())
    try:
        from packaging.version import Version  # noqa: PLC0415

        return str(Version(project_version).base_version)
    except Exception as exc:  # pragma: no cover - canonicalization above already validates
        raise RuntimeError("Could not infer the release version from the active plan.") from exc


def _release_date_from_clock() -> str:
    """Return the UTC release date used for deterministic release-file updates."""
    return _utc_now_iso().split("T", 1)[0]


_INCLUDED_WORK_HEADER_RE = re.compile(r"(?m)^##\s+Included work\s*$")
_NEXT_HEADING_RE = re.compile(r"(?m)^##\s+\S")
_TABLE_ROW_RE = re.compile(r"(?m)^\|(?P<cells>.+)\|\s*$")


def _release_included_work_state(
    plan_path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    """Parse the ``## Included work`` table rows from the active version plan."""
    text = plan_path.read_text(encoding="utf-8")

    header_match = _INCLUDED_WORK_HEADER_RE.search(text)
    if header_match is None:
        return [], [f"Could not locate an '## Included work' section in {plan_path.as_posix()}."]

    remainder = text[header_match.end() :]
    next_heading = _NEXT_HEADING_RE.search(remainder)
    section = remainder[: next_heading.start()] if next_heading else remainder

    rows = [
        [cell.strip() for cell in match.group("cells").split("|")]
        for match in _TABLE_ROW_RE.finditer(section)
    ]
    if len(rows) < 2:
        return [], [f"'## Included work' table has no rows in {plan_path.as_posix()}."]

    header_cells = [cell.lower() for cell in rows[0]]
    try:
        id_index = header_cells.index("id")
        status_index = header_cells.index("status")
    except ValueError:
        return [], [
            "'## Included work' table is missing an 'ID' or 'Status' column in "
            f"{plan_path.as_posix()}."
        ]

    items: list[dict[str, str]] = []
    for row in rows[2:]:
        if len(row) <= max(id_index, status_index) or not any(cell for cell in row):
            continue
        items.append({"id": row[id_index], "status": row[status_index]})

    if not items:
        return [], [f"'## Included work' table has no data rows in {plan_path.as_posix()}."]
    return items, []


def _evaluate_release_plan_readiness(
    plan_path: Path,
    *,
    release_version: str | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Evaluate whether release-handoff prerequisites are satisfied.

    Every row of the active plan's ``## Included work`` table must be marked
    ``Done`` before a release handoff can proceed. Readiness derives solely
    from that table plus the executable checks below; the plan's optional
    ``## Release decision`` section (if present) is non-authoritative summary
    prose and is not parsed.
    """
    included_work, parse_errors = _release_included_work_state(plan_path)
    branch = _current_git_branch()
    pyproject_version = _pyproject_release_version()
    target_version = release_version or _release_version_from_plan(plan_path)
    observed: dict[str, object] = {
        "plan_path": plan_path.as_posix(),
        "branch": branch,
        "pyproject_version": pyproject_version,
        "release_version": target_version,
        "included_work": included_work,
        "parse_errors": parse_errors,
    }

    errors = list(parse_errors)
    if branch is None:
        errors.append("Git is unavailable; release-preflight requires a git checkout.")
    elif branch != "main":
        errors.append(
            f"Release-preflight must run from the main branch; current branch is {branch!r}."
        )

    if _version_lineage(pyproject_version) != _version_lineage(target_version):
        errors.append(
            "pyproject.toml and the active release plan describe different release lines: "
            f"{pyproject_version!r} vs {target_version!r}."
        )

    for item in included_work:
        status = str(item.get("status", "")).strip().lower()
        if status not in {"done", "complete", "completed"}:
            errors.append(
                f"Included work item {item.get('id', '?')!r} is not done "
                f"(status: {item.get('status', '')!r})."
            )

    return observed, errors


def _run_release_readiness_guard(
    plan_path: Path,
    *,
    release_version: str | None = None,
) -> tuple[int, dict[str, object]]:
    """Run the release-readiness guard and return the observed state."""
    observed, errors = _evaluate_release_plan_readiness(
        plan_path,
        release_version=release_version,
    )
    print("\n[Release readiness guard]")
    print(f"Active release plan: {plan_path.as_posix()}")
    if errors:
        print("ERROR: release handoff prerequisites are not satisfied:")
        for error in errors:
            print(f"- {error}")
        return 1, observed
    print("PASS: release plan summary, branch, and version lineage are ready for release handoff.")
    return 0, observed


_INIT_VERSION_FALLBACK_RE = re.compile(r'(?m)^(?P<indent>[ \t]+)return "[^"]+"[ \t]*$')
_CHANGELOG_VERSION_HEADING_RE = re.compile(
    r"^## \[v?(?P<version>[^\]]+)\].*$",
    re.MULTILINE,
)
RELEASE_PREPARED_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "src/calibrated_explanations/__init__.py",
    "CITATION.cff",
    "docs/citing.md",
    "METADATA.json",
    "CHANGELOG.md",
)


def _write_text_if_changed(path: Path, text: str) -> bool:
    """Write ``text`` with repository newlines and report whether it changed."""
    original = path.read_text(encoding="utf-8")
    if original.endswith(("\n", "\r")) and not text.endswith("\n"):
        text += "\n"
    if original == text:
        return False
    path.write_text(text, encoding="utf-8", newline="\n")
    return True


def _replace_required(
    path: Path,
    pattern: re.Pattern[str],
    replacement: str,
    *,
    description: str,
) -> bool:
    """Replace one required release field and report whether the file changed."""
    text = path.read_text(encoding="utf-8")
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not locate {description} in {path.as_posix()}.")
    return _write_text_if_changed(path, updated)


def _write_pyproject_version(version: str) -> bool:
    """Set the first project version declaration in ``pyproject.toml``."""
    return _replace_required(
        Path("pyproject.toml"),
        re.compile(r'(?m)^version\s*=\s*"[^"]+"\s*$'),
        f'version = "{version}"',
        description="[project].version",
    )


def _write_init_version_fallback(version: str) -> bool:
    """Set the source-tree fallback used when package metadata is unavailable."""
    path = Path("src/calibrated_explanations/__init__.py")
    text = path.read_text(encoding="utf-8")
    updated, count = _INIT_VERSION_FALLBACK_RE.subn(
        lambda match: f'{match.group("indent")}return "{version}"',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not locate the __version__ fallback string in __init__.py.")
    return _write_text_if_changed(path, updated)


def _changelog_release_content(text: str) -> str:
    """Return release-note content with generated compare links removed."""
    lines = [
        line
        for line in text.strip().splitlines()
        if not line.strip().startswith("[Full changelog](")
    ]
    return "\n".join(lines).strip()


def _merge_changelog_release_content(*contents: str) -> str:
    """Merge release-note category sections without duplicating headings."""
    preamble: list[str] = []
    section_order: list[str] = []
    section_bodies: dict[str, list[str]] = {}
    heading_pattern = re.compile(r"(?m)^###\s+.+$")

    for content in contents:
        headings = list(heading_pattern.finditer(content))
        if not headings:
            if content.strip():
                preamble.append(content.strip())
            continue
        leading = content[: headings[0].start()].strip()
        if leading:
            preamble.append(leading)
        for index, heading in enumerate(headings):
            title = heading.group(0).strip()
            end = headings[index + 1].start() if index + 1 < len(headings) else len(content)
            body = content[heading.end() : end].strip()
            if title not in section_bodies:
                section_order.append(title)
                section_bodies[title] = []
            if body:
                section_bodies[title].append(body)

    merged = list(preamble)
    for title in section_order:
        section = title
        if section_bodies[title]:
            section += "\n\n" + "\n\n".join(section_bodies[title])
        merged.append(section)
    return "\n\n".join(merged)


def _prepare_changelog_release(version: str, release_date: str) -> bool:
    """Move Unreleased notes into an idempotent versioned CHANGELOG section."""
    path = Path("CHANGELOG.md")
    text = path.read_text(encoding="utf-8")
    unreleased_match = re.search(r"^## \[Unreleased\]\s*$", text, re.MULTILINE)
    if unreleased_match is None:
        raise RuntimeError("Could not locate the CHANGELOG [Unreleased] section.")

    headings = list(_CHANGELOG_VERSION_HEADING_RE.finditer(text, unreleased_match.end()))
    if not headings:
        raise RuntimeError("Could not locate a prior version section in CHANGELOG.md.")
    target_heading = next(
        (match for match in headings if _canonical_version(match.group("version")) == version),
        None,
    )
    first_heading = headings[0]
    unreleased_body = _changelog_release_content(
        text[unreleased_match.end() : first_heading.start()]
    )

    if target_heading is not None:
        later_headings = [match for match in headings if match.start() > target_heading.start()]
        if not later_headings:
            raise RuntimeError("The target CHANGELOG section has no prior-version section.")
        prior_heading = later_headings[0]
        target_body = _changelog_release_content(text[target_heading.end() : prior_heading.start()])
        suffix = text[prior_heading.start() :]
    else:
        prior_heading = first_heading
        target_body = ""
        suffix = text[first_heading.start() :]

    prior_version = _canonical_version(prior_heading.group("version"))
    combined_body = _merge_changelog_release_content(unreleased_body, target_body)
    release_section = (
        f"## [v{version}](https://github.com/Moffran/calibrated_explanations/releases/tag/v{version}) "
        f"- {release_date}\n\n"
        "[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/"
        f"v{prior_version}...v{version})"
    )
    if combined_body:
        release_section += f"\n\n{combined_body}"

    prefix = text[: unreleased_match.start()].rstrip()
    updated = (
        f"{prefix}\n\n## [Unreleased]\n\n"
        "[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/"
        f"v{version}...main)\n\n{release_section}\n\n{suffix.lstrip()}"
    )
    return _write_text_if_changed(path, updated)


def _prepare_software_citation(version: str, release_datetime: datetime) -> bool:
    """Update only the software BibTeX block in ``docs/citing.md``."""
    path = Path("docs/citing.md")
    text = path.read_text(encoding="utf-8")
    marker = "To cite this software"
    marker_index = text.find(marker)
    if marker_index < 0:
        raise RuntimeError(f"Could not locate the software citation in {path.as_posix()}.")
    fence_start = text.find("```", marker_index)
    if fence_start < 0:
        raise RuntimeError(f"Could not locate the software BibTeX fence in {path.as_posix()}.")
    fence_end = text.find("```", fence_start + 3)
    if fence_end < 0:
        raise RuntimeError(f"Could not locate the software BibTeX fence in {path.as_posix()}.")
    block = text[fence_start:fence_end]
    replacements: tuple[tuple[re.Pattern[str], str, str], ...] = (
        (
            re.compile(r"(?m)^(\s*version\s*=\s*)\{[^}]+\}(,?)\s*$"),
            rf"\g<1>{{v{version}}}\g<2>",
            "software BibTeX version",
        ),
        (
            re.compile(r"(?m)^(\s*month\s*=\s*)\{[^}]+\}(,?)\s*$"),
            rf"\g<1>{{{release_datetime.strftime('%B')}}}\g<2>",
            "software BibTeX month",
        ),
        (
            re.compile(r"(?m)^(\s*year\s*=\s*)\{[^}]+\}(,?)\s*$"),
            rf"\g<1>{{{release_datetime.year}}}\g<2>",
            "software BibTeX year",
        ),
    )
    for pattern, replacement, description in replacements:
        block, count = pattern.subn(replacement, block, count=1)
        if count != 1:
            raise RuntimeError(f"Could not locate {description} in {path.as_posix()}.")
    updated = text[:fence_start] + block + text[fence_end:]
    return _write_text_if_changed(path, updated)


def _prepare_release_files(
    *,
    release_version: str,
    release_date: str,
) -> list[str]:
    """Update every deterministic release file owned by release.md steps 3-4."""
    version = _canonical_version(release_version)
    release_datetime = datetime.strptime(release_date, "%Y-%m-%d")
    if release_datetime.strftime("%Y-%m-%d") != release_date:
        raise ValueError(f"Release date must use YYYY-MM-DD form, got {release_date!r}.")
    changed: list[str] = []

    def record(path: str, did_change: bool) -> None:
        if did_change:
            changed.append(path)

    record("pyproject.toml", _write_pyproject_version(version))
    record(
        "src/calibrated_explanations/__init__.py",
        _write_init_version_fallback(version),
    )
    record(
        "CITATION.cff",
        _replace_required(
            Path("CITATION.cff"),
            re.compile(r"(?m)^version:\s*\S+\s*$"),
            f"version: v{version}",
            description="citation version",
        ),
    )
    record(
        "CITATION.cff",
        _replace_required(
            Path("CITATION.cff"),
            re.compile(r"(?m)^date-released:\s*['\"][^'\"]+['\"]\s*$"),
            f"date-released: '{release_date}'",
            description="citation release date",
        ),
    )

    record("docs/citing.md", _prepare_software_citation(version, release_datetime))

    metadata_path = Path("METADATA.json")
    record(
        "METADATA.json",
        _replace_required(
            metadata_path,
            re.compile(r'(?m)^(\s*"version"\s*:\s*)"[^"]+"(,?)\s*$'),
            rf'\g<1>"{version}"\g<2>',
            description="repository metadata version",
        ),
    )
    record("CHANGELOG.md", _prepare_changelog_release(version, release_date))

    docs_conf = Path("docs/conf.py")
    if not docs_conf.exists():
        raise RuntimeError("docs/conf.py is missing; dynamic release metadata cannot be verified.")
    print(
        f"Prepared release files for v{version} ({release_date}); docs/conf.py derives its "
        "release/version from installed package metadata."
    )
    if changed:
        print("Updated release files: " + ", ".join(dict.fromkeys(changed)))
    else:
        print("Release files were already up to date.")
    return list(dict.fromkeys(changed))


def run_release_prepare_files(
    *,
    plan_path: Path | None = None,
    release_version: str | None = None,
    release_date: str | None = None,
) -> int:
    """Prepare deterministic release files without requiring checklist closure.

    This focused lane exists so version-dependent release tasks can align and
    verify package metadata before the strict release-preflight readiness guard
    runs. The full preflight repeats the same preparation idempotently and
    remains the only command that can produce a publication handoff report.
    """
    try:
        resolved_plan = _resolved_plan_path(plan_path, version=release_version)
        target_version = _canonical_version(
            release_version or _release_version_from_plan(resolved_plan)
        )
        target_date = release_date or _release_date_from_clock()
        branch = _current_git_branch()
        if branch is None:
            raise RuntimeError("Git is unavailable; release-file preparation requires a checkout.")
        if branch != "main":
            raise RuntimeError(
                "Release-file preparation must run from the main branch; "
                f"current branch is {branch!r}."
            )
        project_version = _pyproject_release_version()
        if _version_lineage(project_version) != _version_lineage(target_version):
            raise RuntimeError(
                "pyproject.toml and the active release plan describe different release lines: "
                f"{project_version!r} vs {target_version!r}."
            )
        changed = _prepare_release_files(
            release_version=target_version,
            release_date=target_date,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: Release-file preparation failed: {exc}")
        return 1

    if changed:
        print("PASS: Deterministic release files prepared for task verification.")
    else:
        print("PASS: Deterministic release files already match the requested release.")
    print(
        "This command does not create a release handoff. Run `make release-preflight` "
        "after all prerequisite task checklists are closed."
    )
    return 0


def _write_release_preflight_report(
    records: list[dict[str, object]],
    started_at: float,
    output_path: Path,
    *,
    observed: dict[str, object],
    exit_status: int,
    git_status_porcelain: str | None,
    prepared_release_files: list[str] | None = None,
    completed: bool = False,
) -> None:
    """Write the strict release-preflight report consumed by release-finalize."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_exit_status = exit_status if completed or exit_status != 0 else None
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "plan_path": observed.get("plan_path"),
        "branch": observed.get("branch"),
        "pyproject_version": observed.get("pyproject_version"),
        "release_version": observed.get("release_version"),
        "included_work": observed.get("included_work", []),
        "parse_errors": observed.get("parse_errors", []),
        "steps": records,
        "exit_status": final_exit_status,
        "preflight_passed": completed and exit_status == 0,
        "git_status_porcelain": git_status_porcelain,
        "worktree_changed_files": _worktree_changed_file_hashes(),
        "head_sha": _git_text("rev-parse", "HEAD"),
        "automated_release_steps": list(range(1, 11)),
        "manual_release_steps": list(RELEASE_MANUAL_STEP_RANGE),
        "postcommit_release_steps": list(RELEASE_POSTCOMMIT_STEP_RANGE),
        "prepared_release_files": (
            list(RELEASE_PREPARED_FILES) if prepared_release_files is not None else []
        ),
        "changed_release_files": prepared_release_files or [],
        "verified_release_files": (
            ["docs/conf.py", str(observed.get("plan_path"))]
            if prepared_release_files is not None
            else []
        ),
        "total_elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def _run_release_twine_check() -> int:
    """Validate built release artifacts with twine."""
    artifacts = sorted(path.as_posix() for path in Path("dist").glob("*"))
    if not artifacts:
        print("ERROR: No build artifacts found under dist/ for twine validation.")
        return 1
    return _run_step(
        Step("Release artifact validation", _python_cmd("-m", "twine", "check", *artifacts))
    )


def _run_release_wheel_smoke() -> int:
    """Smoke-test the built wheel in an isolated temporary virtual environment."""
    wheel_paths = sorted(
        Path("dist").glob("*.whl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not wheel_paths:
        print("ERROR: No wheel artifacts found under dist/ for the release smoke test.")
        return 1
    wheel_path = wheel_paths[0].resolve()

    run_dir = Path(tempfile.mkdtemp(prefix="ce-release-wheel-smoke-")).resolve()
    venv_dir = run_dir / "venv-wheel"
    builder = venv.EnvBuilder(with_pip=True, clear=True)
    builder.create(venv_dir)
    venv_python = _venv_python(venv_dir)
    smoke_steps = [
        Step(
            "Wheel smoke: upgrade pip",
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        ),
        Step(
            "Wheel smoke: install release artifact",
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                str(wheel_path),
            ],
        ),
        Step(
            "Wheel smoke: import release artifact",
            [
                str(venv_python),
                "-c",
                "import calibrated_explanations as ce; print(ce.__version__)",
            ],
        ),
    ]
    for step in smoke_steps:
        rc = _run_step(step)
        if rc != 0:
            return rc
    return 0


def run_release_preflight(
    *,
    plan_path: Path | None = None,
    release_version: str | None = None,
    release_date: str | None = None,
) -> int:
    """Run the strict pre-step-11 release gate and persist a handoff snapshot."""
    _purge_release_transient_async_logs()
    resolved_plan = _resolved_plan_path(plan_path, version=release_version)
    target_version = _canonical_version(
        release_version or _release_version_from_plan(resolved_plan)
    )
    target_date = release_date or _release_date_from_clock()
    records: list[dict[str, object]] = []
    prepared_release_files: list[str] | None = None
    started_at = time.monotonic()

    def record_step(name: str, command: str, rc: int, step_started_at: float) -> None:
        records.append(
            {
                "name": name,
                "command": command,
                "exit_code": rc,
                "elapsed_seconds": round(time.monotonic() - step_started_at, 3),
            }
        )

    readiness_started_at = time.monotonic()
    readiness_rc, observed = _run_release_readiness_guard(
        resolved_plan,
        release_version=target_version,
    )
    record_step(
        "Release readiness guard",
        f"parse {resolved_plan.as_posix()} release gate summary",
        readiness_rc,
        readiness_started_at,
    )
    git_status_porcelain = _current_git_status_porcelain()
    _write_release_preflight_report(
        records,
        started_at,
        RELEASE_PREFLIGHT_REPORT,
        observed=observed,
        exit_status=readiness_rc,
        git_status_porcelain=git_status_porcelain,
    )
    if readiness_rc != 0:
        return readiness_rc

    preparation_started_at = time.monotonic()
    try:
        prepared_release_files = _prepare_release_files(
            release_version=target_version,
            release_date=target_date,
        )
        preparation_rc = 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: Release file preparation failed: {exc}")
        preparation_rc = 1
    if preparation_rc == 0:
        observed["pyproject_version"] = _pyproject_release_version()
        observed["release_version"] = target_version
    record_step(
        "Release file preparation",
        f"prepare release.md files for v{target_version} ({target_date})",
        preparation_rc,
        preparation_started_at,
    )
    git_status_porcelain = _current_git_status_porcelain()
    _write_release_preflight_report(
        records,
        started_at,
        RELEASE_PREFLIGHT_REPORT,
        observed=observed,
        exit_status=preparation_rc,
        git_status_porcelain=git_status_porcelain,
        prepared_release_files=prepared_release_files,
    )
    if preparation_rc != 0:
        return preparation_rc

    steps = [
        Step("Full pytest suite", _python_cmd("-m", "pytest", "-q")),
        Step(
            "Editable install (release tree)", _python_cmd("-m", "pip", "install", "-e", ".[dev]")
        ),
        Step(
            "Editable install version smoke",
            _python_cmd("-c", "import calibrated_explanations as ce; print(ce.__version__)"),
        ),
        *_release_notebook_steps(),
        Step("Release profile", _python_cmd("scripts/local_checks.py", "--profile", "release")),
    ]
    for step in steps:
        step_started_at = time.monotonic()
        rc = _run_step(step)
        record_step(step.name, _command_text(step.command), rc, step_started_at)
        git_status_porcelain = _current_git_status_porcelain()
        _write_release_preflight_report(
            records,
            started_at,
            RELEASE_PREFLIGHT_REPORT,
            observed=observed,
            exit_status=rc,
            git_status_porcelain=git_status_porcelain,
            prepared_release_files=prepared_release_files,
        )
        if rc != 0:
            return rc

    twine_started_at = time.monotonic()
    twine_rc = _run_release_twine_check()
    record_step(
        "Release artifact validation",
        "python -m twine check dist/*",
        twine_rc,
        twine_started_at,
    )
    git_status_porcelain = _current_git_status_porcelain()
    _write_release_preflight_report(
        records,
        started_at,
        RELEASE_PREFLIGHT_REPORT,
        observed=observed,
        exit_status=twine_rc,
        git_status_porcelain=git_status_porcelain,
        prepared_release_files=prepared_release_files,
    )
    if twine_rc != 0:
        return twine_rc

    wheel_started_at = time.monotonic()
    wheel_rc = _run_release_wheel_smoke()
    record_step(
        "Release wheel smoke",
        "create temp venv; install calibrated_explanations from dist; import version",
        wheel_rc,
        wheel_started_at,
    )
    git_status_porcelain = _current_git_status_porcelain()
    _write_release_preflight_report(
        records,
        started_at,
        RELEASE_PREFLIGHT_REPORT,
        observed=observed,
        exit_status=wheel_rc,
        git_status_porcelain=git_status_porcelain,
        prepared_release_files=prepared_release_files,
        completed=True,
    )
    return wheel_rc


def run_release_finalize(*, plan_path: Path | None = None) -> int:
    """Verify that release-preflight is still valid before manual release steps."""
    _purge_release_transient_async_logs()
    resolved_plan = _resolved_plan_path(plan_path)
    print("\n[Release finalize guard]")
    if not RELEASE_PREFLIGHT_REPORT.exists():
        print(
            "ERROR: No release-preflight report found. Run `make release-preflight` before the manual release phase."
        )
        return 1

    try:
        payload = json.loads(RELEASE_PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: release-preflight report is not valid JSON: {exc}")
        return 1

    exit_status = payload.get("exit_status")
    if not payload.get("preflight_passed"):
        if exit_status in (None, 0):
            steps = payload.get("steps")
            last_step = steps[-1] if isinstance(steps, list) and steps else {}
            last_step_name = (
                last_step.get("name", "none") if isinstance(last_step, dict) else "unknown"
            )
            last_step_exit = (
                last_step.get("exit_code", "unknown") if isinstance(last_step, dict) else "unknown"
            )
            print(
                "ERROR: The latest release-preflight is incomplete. "
                f"Last recorded step: {last_step_name!r} (exit code {last_step_exit}). "
                "A successful checkpoint does not mean the complete preflight passed. "
                "Re-run `make release-preflight` and let it finish before finalizing."
            )
            return 1
        print(
            "ERROR: The latest release-preflight run failed with exit status "
            f"{exit_status!r}. Re-run `make release-preflight`."
        )
        return 1
    if exit_status != 0:
        print(
            "ERROR: The release-preflight report is inconsistent: it is marked passed "
            f"but records exit status {exit_status!r}. Re-run `make release-preflight`."
        )
        return 1

    try:
        reported_version = _canonical_version(str(payload.get("release_version", "")))
        current_version = _canonical_version(_pyproject_release_version())
        plan_version = _release_version_from_plan(resolved_plan)
    except RuntimeError as exc:
        print(f"ERROR: Could not validate the release-preflight version snapshot: {exc}")
        return 1
    if reported_version != current_version or reported_version != plan_version:
        print(
            "ERROR: The release-preflight version no longer matches the project and active plan. "
            f"Report: {reported_version!r}; project: {current_version!r}; plan: {plan_version!r}."
        )
        return 1

    readiness_rc, _ = _run_release_readiness_guard(resolved_plan)
    if readiness_rc != 0:
        return readiness_rc

    current_branch = _current_git_branch()
    if current_branch != payload.get("branch"):
        print(
            "ERROR: The current branch no longer matches the successful preflight snapshot. "
            f"Current branch: {current_branch!r}; preflight branch: {payload.get('branch')!r}."
        )
        return 1

    current_status = _current_git_status_porcelain()
    recorded_status = payload.get("git_status_porcelain")
    recorded_files = payload.get("worktree_changed_files")
    if current_status != recorded_status:
        if not isinstance(recorded_files, dict):
            # Report predates the per-file content-hash field; fall back to
            # the strict status-string comparison this replaces.
            print(
                "ERROR: The worktree changed after release-preflight. Re-run `make release-preflight` before step 11."
            )
            return 1
        changed_paths = _changed_paths_since(
            payload.get("head_sha") if isinstance(payload.get("head_sha"), str) else None
        )
        unexpected_new = sorted(changed_paths - set(recorded_files))
        if unexpected_new:
            print(
                "ERROR: The worktree changed after release-preflight with file(s) outside the "
                f"preflight snapshot: {', '.join(unexpected_new)}. Re-run `make release-preflight` before step 11."
            )
            return 1
        content_mismatches = sorted(
            path
            for path, recorded_hash in recorded_files.items()
            if _file_content_hash(path) != recorded_hash
        )
        if content_mismatches:
            print(
                "ERROR: The following file(s) changed content after release-preflight (not "
                f"just commit status): {', '.join(content_mismatches)}. Re-run `make release-preflight` before step 11."
            )
            return 1
        print(
            "NOTE: The worktree's commit status changed since release-preflight (e.g. its "
            "output was committed), but every affected file's content is byte-identical to the "
            "successful preflight snapshot. Continuing."
        )

    print(
        "PASS: release-preflight is still valid. Continue with the manual release phase "
        f"(steps {RELEASE_MANUAL_STEP_RANGE[0]}-{RELEASE_MANUAL_STEP_RANGE[-1]} in the private runbook: "
        "commit/tag/push, publish docs on Read the Docs, upload to PyPI). "
        f"Run `make release-postcommit` afterward for steps {RELEASE_POSTCOMMIT_STEP_RANGE[0]}-"
        f"{RELEASE_POSTCOMMIT_STEP_RANGE[-1]}."
    )
    return 0


def _next_patch_version(version: str) -> str:
    """Return the next patch release version for a released ``X.Y.Z`` version."""
    try:
        from packaging.version import Version  # noqa: PLC0415

        release = Version(version).release
    except Exception as exc:  # noqa: BLE001 - converted to a release-workflow error
        raise RuntimeError(f"Invalid released version {version!r}.") from exc
    if len(release) != 3:
        raise RuntimeError(f"Released version must have X.Y.Z form, got {version!r}.")
    major, minor, patch = release
    return f"{major}.{minor}.{patch + 1}"


def _is_development_version(version: str) -> bool:
    """Return whether ``version`` is a PEP 440 development release."""
    try:
        from packaging.version import Version  # noqa: PLC0415

        return Version(version).is_devrelease
    except Exception as exc:  # noqa: BLE001 - converted to a release-workflow error
        raise RuntimeError(f"Invalid project version {version!r}.") from exc


def _run_release_pypi_page_check(version: str) -> int:
    """Verify that PyPI exposes the exact published release metadata.

    Only the JSON API is checked. The rendered project page is not: PyPI's
    CDN serves a bot-protection JS challenge page to plain scripted requests,
    so asserting on its body is unreliable regardless of release state, while
    the JSON API is the authoritative source for published version metadata.
    """
    endpoint = f"https://pypi.org/pypi/calibrated-explanations/{version}/json"
    script = (
        "import json, urllib.request; "
        f"payload = json.load(urllib.request.urlopen({endpoint!r}, timeout=30)); "
        "actual = payload['info']['version']; "
        f"expected = {version!r}; "
        "assert actual == expected, f'PyPI version mismatch: {actual!r} != {expected!r}'; "
        "print(actual)"
    )
    return _run_step(
        Step(
            "Postpublish smoke: verify PyPI project metadata",
            _python_cmd("-c", script),
        )
    )


def _run_release_pypi_install_smoke(version: str) -> int:
    """Smoke-test the published PyPI release in an isolated temporary virtual environment."""
    run_dir = Path(tempfile.mkdtemp(prefix="ce-release-postpublish-smoke-")).resolve()
    venv_dir = run_dir / "venv-postpublish"
    builder = venv.EnvBuilder(with_pip=True, clear=True)
    builder.create(venv_dir)
    venv_python = _venv_python(venv_dir)
    smoke_steps = [
        Step(
            "Postpublish smoke: upgrade pip",
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        ),
        Step(
            "Postpublish smoke: install published release",
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                f"calibrated-explanations=={version}",
            ],
        ),
        Step(
            "Postpublish smoke: import published release",
            [
                str(venv_python),
                "-c",
                (
                    "import calibrated_explanations as ce; "
                    f"expected = {version!r}; actual = ce.__version__; "
                    "assert actual == expected, "
                    "f'Installed version mismatch: {actual!r} != {expected!r}'; "
                    "print(actual)"
                ),
            ],
        ),
    ]
    for step in smoke_steps:
        rc = _run_step(step)
        if rc != 0:
            return rc
    return 0


def _next_development_version(released_version: str, requested_next_version: str | None) -> str:
    """Resolve the next development version after a release ships.

    Uses an explicitly supplied next-release label (``NEXT_VERSION``) when
    given; otherwise advances by one patch. This never infers a minor/major
    bump or consults a roadmap document — choosing the next milestone's scope
    is a maintainer decision made through a GitHub milestone, not something
    this command guesses.
    """
    label = (
        requested_next_version.removeprefix("v")
        if requested_next_version is not None
        else _next_patch_version(released_version)
    )
    return f"{_canonical_version(label)}-dev"


def _archive_release_plan(current_plan: Path) -> Path:
    """Move the completed version plan into ``development/finished-work``."""
    finished_dir = current_plan.parent.parent / "finished-work"
    finished_dir.mkdir(parents=True, exist_ok=True)
    target = finished_dir / current_plan.name
    if target.exists():
        raise RuntimeError(
            "Refusing to archive the release plan because both current-work and "
            f"finished-work contain {current_plan.name}: {target.as_posix()}"
        )
    shutil.move(str(current_plan), str(target))
    print(f"Archived released plan: {target.as_posix()}")
    return target


def run_release_postcommit(
    *,
    plan_path: Path | None = None,
    next_version: str | None = None,
    release_date: str | None = None,
) -> int:
    """Run the automatable post-publish steps (release.md steps 14-17).

    Steps 11-13 (commit/tag/push, Read the Docs publish, PyPI upload) remain
    manual and must already be done before this runs.
    """
    print(
        f"\n[Release postcommit: steps {RELEASE_POSTCOMMIT_STEP_RANGE[0]}-"
        f"{RELEASE_POSTCOMMIT_STEP_RANGE[-1]}]"
    )
    try:
        released_version = _canonical_version(_pyproject_release_version())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: Could not resolve the published release version: {exc}")
        return 1
    if _is_development_version(released_version):
        print(
            "ERROR: pyproject.toml still reports a -dev version "
            f"({released_version!r}). Run release-postcommit only after step 13 "
            "(PyPI upload) has published the release version."
        )
        return 1

    page_rc = _run_release_pypi_page_check(released_version)
    if page_rc != 0:
        return page_rc
    print(f"Step 14: PyPI exposes calibrated-explanations=={released_version}.")

    smoke_rc = _run_release_pypi_install_smoke(released_version)
    if smoke_rc != 0:
        return smoke_rc
    print(
        f"Step 15: installed calibrated-explanations=={released_version} from PyPI in a clean venv."
    )

    completed_date = release_date or _release_date_from_clock()
    try:
        completed_datetime = datetime.strptime(completed_date, "%Y-%m-%d")
        if completed_datetime.strftime("%Y-%m-%d") != completed_date:
            raise ValueError
    except ValueError:
        print(f"ERROR: Release date must use YYYY-MM-DD form, got {completed_date!r}.")
        return 1

    try:
        resolved_plan = _resolved_plan_path(plan_path, version=released_version)
        archive_target = resolved_plan.parent.parent / "finished-work" / resolved_plan.name
        if archive_target.exists():
            print(
                "ERROR: release-plan archive collision: both current-work and finished-work "
                f"contain {resolved_plan.name}. Resolve the duplicate before rerunning postcommit."
            )
            return 1
        development_version = _next_development_version(released_version, next_version)
        archived_plan = _archive_release_plan(resolved_plan)

        _write_pyproject_version(development_version)
        _write_init_version_fallback(_canonical_version(development_version))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: Release plan handoff or development-version bump failed: {exc}")
        return 1
    print(
        f"Step 16: released plan archived at {archived_plan.as_posix()}. No next version plan is "
        "created automatically — open one under development/current-work/ once maintainers select "
        "the next GitHub milestone (see the ce-release-planner skill)."
    )

    print(
        f"Step 17: bumped the working version to {development_version} in pyproject.toml "
        "and its canonical fallback in __init__.py."
    )
    return 0


def _common_skipped_heavy_gates() -> list[dict[str, str]]:
    """Return the heavy gates excluded from quick/task/pr by default."""
    return [
        {"gate": "coverage", "reason": "Reserved for full/release validation."},
        {"gate": "docs_html", "reason": "Reserved for full/release validation."},
        {"gate": "docs_linkcheck", "reason": "Reserved for release validation."},
        {"gate": "notebook_audit", "reason": "Reserved for release validation."},
        {"gate": "notebook_execution", "reason": "Reserved for release validation."},
        {"gate": "dependency_audit", "reason": "Reserved for release validation."},
        {"gate": "performance", "reason": "Reserved for full/release validation."},
        {"gate": "capability_evidence_refresh", "reason": "Reserved for release validation."},
    ]


_TASK_VERIFICATION_BLOCK_RE = re.compile(
    r"```toml ce-task-verification\r?\n(?P<body>.*?)```",
    re.DOTALL,
)


def _load_task_verification_config(plan_path: Path) -> dict[str, object]:
    """Parse the task-verification TOML block from a release plan."""
    plan_text = plan_path.read_text(encoding="utf-8")
    match = _TASK_VERIFICATION_BLOCK_RE.search(plan_text)
    if match is None:
        raise ValueError(
            f"Could not find a ```toml ce-task-verification``` block in {plan_path.as_posix()}."
        )
    config = tomllib.loads(match.group("body"))
    if config.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported ce-task-verification schema in {plan_path.as_posix()}: "
            f"{config.get('schema_version')!r}."
        )
    task_table = config.get("task")
    if not isinstance(task_table, dict) or not task_table:
        raise ValueError(f"No task mappings found in {plan_path.as_posix()}.")
    return config


def _resolved_plan_path(plan_path: Path | None, *, version: str | None = None) -> Path:
    """Return an explicit plan or discover the active plan from project version lineage."""
    if plan_path is not None:
        return plan_path
    project_version = version or _pyproject_release_version()
    project_lineage = _version_lineage(project_version)
    candidates: list[Path] = []
    for candidate in sorted(Path("development/current-work").glob("v*_plan.md")):
        try:
            if _version_lineage(_release_version_from_plan(candidate)) == project_lineage:
                candidates.append(candidate)
        except RuntimeError:
            continue
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "Could not discover an active version plan matching project version "
            f"{project_version!r}; pass --plan explicitly."
        )
    joined = ", ".join(path.as_posix() for path in candidates)
    raise ValueError(
        f"Multiple active version plans match project version {project_version!r}: {joined}. "
        "Pass --plan explicitly."
    )


def _task_verification_tasks(plan_path: Path | None = None) -> dict[int, dict[str, object]]:
    """Return normalized task-verification entries keyed by task id."""
    config = _load_task_verification_config(_resolved_plan_path(plan_path))
    raw_tasks = config["task"]
    assert isinstance(raw_tasks, dict)
    tasks: dict[int, dict[str, object]] = {}
    for raw_id, raw_payload in raw_tasks.items():
        if not isinstance(raw_payload, dict):
            raise ValueError(f"Task mapping for {raw_id!r} must be a table.")
        task_id = int(raw_id)
        tasks[task_id] = raw_payload
    return tasks


def supported_task_ids(*, plan_path: Path | None = None) -> frozenset[int]:
    """Return task ids mapped in the selected release plan."""
    return frozenset(_task_verification_tasks(plan_path).keys())


def _unsupported_task_mapping_error(task: int, plan_path: Path | None = None) -> ValueError:
    supported = sorted(supported_task_ids(plan_path=plan_path))
    return ValueError(f"Unsupported task profile mapping: {task}. Supported task ids: {supported}.")


def _command_from_plan(command: str) -> list[str]:
    """Parse a task-plan command string into an executable argument vector."""
    parts = shlex.split(command, posix=True)
    if not parts:
        raise ValueError("Task verification command entries must not be empty.")
    if parts[0] == "python":
        return _python_cmd(*parts[1:])
    return parts


def _step_from_plan(step_config: dict[str, object]) -> Step:
    """Build a local-check step from one plan step table."""
    name = step_config.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Each task-verification step must define a non-empty 'name'.")
    if "pytest" in step_config:
        pytest_paths = step_config["pytest"]
        pytest_args = step_config.get("pytest_args", [])
        if not isinstance(pytest_paths, list) or not all(
            isinstance(path, str) and path for path in pytest_paths
        ):
            raise ValueError(f"{name}: pytest steps must define a non-empty string list.")
        if not isinstance(pytest_args, list) or not all(
            isinstance(arg, str) for arg in pytest_args
        ):
            raise ValueError(f"{name}: pytest_args must be a string list when present.")
        return Step(
            name,
            _pytest_no_cov_command("-q", *pytest_paths, *pytest_args, "-o", "addopts="),
        )
    command = step_config.get("command")
    if isinstance(command, str) and command:
        return Step(name, _command_from_plan(command))
    raise ValueError(f"{name}: task-verification steps must define either pytest or command.")


def _task_specific_steps(task: int, *, plan_path: Path | None = None) -> list[Step]:
    """Return the focused validation steps for a mapped release-plan task."""
    tasks = _task_verification_tasks(plan_path)
    if task not in tasks:
        raise _unsupported_task_mapping_error(task, plan_path)
    raw_steps = tasks[task].get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"Task {task} has no step definitions in the verification block.")
    return [
        _step_from_plan(step_config) for step_config in raw_steps if isinstance(step_config, dict)
    ]


def _task_specific_lint_targets(task: int, *, plan_path: Path | None = None) -> list[str]:
    """Return lint/format targets scoped to one mapped release-plan task."""
    tasks = _task_verification_tasks(plan_path)
    if task not in tasks:
        raise _unsupported_task_mapping_error(task, plan_path)
    task_config = tasks[task]
    if "lint_targets" not in task_config:
        return _changed_python_targets()
    lint_targets = task_config["lint_targets"]
    if not isinstance(lint_targets, list) or not all(
        isinstance(target, str) and target for target in lint_targets
    ):
        raise ValueError(f"Task {task} lint_targets must be a string list when present.")
    return lint_targets


def task_specific_steps(task: int, *, plan_path: Path | None = None) -> list[Step]:
    """Public accessor for the focused validation steps of one task."""
    return _task_specific_steps(task, plan_path=plan_path)


def task_specific_lint_targets(task: int, *, plan_path: Path | None = None) -> list[str]:
    """Public accessor for the lint/format targets of one task."""
    return _task_specific_lint_targets(task, plan_path=plan_path)


def _step_runs_docs_html(step: Step) -> bool:
    """Return True when a task step already runs a Sphinx HTML build."""
    command = step.command
    if len(command) < 2:
        return False
    if command[0] != sys.executable:
        return False
    if "-m" not in command:
        return False
    try:
        module_name = command[command.index("-m") + 1]
    except IndexError:
        return False
    return module_name == "sphinx" and "docs/_build/html" in command


def build_profile_plan(
    profile: str,
    *,
    task: int | None,
    mypy_targets: list[str],
    lint_targets: list[str],
    pre_commit_available: bool,
    plan_path: Path | None = None,
) -> ProfilePlan:
    """Build the local-check step plan for the requested profile."""
    skipped_heavy = _common_skipped_heavy_gates()
    if profile == "quick":
        return ProfilePlan(
            profile="quick",
            steps=_quick_steps(mypy_targets, lint_targets),
            skipped_heavy_gates=skipped_heavy,
        )
    if profile == "task":
        if task is None:
            raise ValueError("The task profile requires --task / TASK=<n>.")
        task_steps = _task_specific_steps(task, plan_path=plan_path)
        task_skips = [
            entry
            for entry in skipped_heavy
            if not (
                entry["gate"] == "docs_html"
                and any(_step_runs_docs_html(step) for step in task_steps)
            )
        ]
        return ProfilePlan(
            profile="task",
            task=task,
            steps=[*_quick_steps(mypy_targets, lint_targets), *task_steps],
            skipped_heavy_gates=task_skips,
        )
    if profile == "pr":
        return ProfilePlan(
            profile="pr",
            steps=_pr_steps(mypy_targets, lint_targets, pre_commit_available=pre_commit_available),
            skipped_heavy_gates=skipped_heavy,
        )
    if profile == "full":
        return ProfilePlan(
            profile="full",
            steps=_full_steps(
                mypy_targets, lint_targets, pre_commit_available=pre_commit_available
            ),
            skipped_heavy_gates=[],
        )
    if profile == "release":
        return ProfilePlan(
            profile="release",
            steps=_release_steps(
                mypy_targets, lint_targets, pre_commit_available=pre_commit_available
            ),
            skipped_heavy_gates=[],
        )
    raise ValueError(f"Unknown profile: {profile}")


def _write_task_profile_report(
    plan: ProfilePlan,
    *,
    commands_run: list[str],
    exit_status: int,
    output_path: Path,
    requested_paths: list[str] | None = None,
) -> None:
    """Write the machine-readable task-profile execution report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "selected_profile": plan.profile,
        "task_id": plan.task,
        "commands_run": commands_run,
        "skipped_heavy_gates": plan.skipped_heavy_gates,
        "requested_paths": requested_paths or [],
        "exit_status": exit_status,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def run_profile_plan(
    plan: ProfilePlan,
    *,
    task_report_path: Path | None = None,
    requested_paths: list[str] | None = None,
) -> int:
    """Run a planned profile and optionally persist a task-profile report."""
    if plan.profile == "release":
        _purge_release_transient_async_logs()
    commands_run: list[str] = []
    exit_status = 0
    for step in plan.steps:
        commands_run.append(_command_text(step.command))
        rc = _run_step(step)
        if rc != 0:
            exit_status = rc
            break
    if task_report_path is not None:
        _write_task_profile_report(
            plan,
            commands_run=commands_run,
            exit_status=exit_status,
            output_path=task_report_path,
            requested_paths=requested_paths,
        )
    return exit_status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local verification profiles in the current environment."
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "task", "pr", "full", "release"),
        help="Verification profile to run. Defaults to 'full' unless --skip-main is used.",
    )
    parser.add_argument("--task", type=int, help="Focused release-plan task id for --profile task.")
    parser.add_argument(
        "--plan",
        type=Path,
        help="Optional release-plan override containing the ce-task-verification block.",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional changed-path hints for task-focused reporting/routing.",
    )
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help="Compatibility alias for --profile pr.",
    )
    parser.add_argument(
        "--instruction-index-sync",
        action="store_true",
        help="Regenerate derived ADR, standard and skill indexes and platform skill packages.",
    )
    parser.add_argument(
        "--ci-parity",
        action="store_true",
        help="Run the local workflow run-block smoke runner (not an exact GitHub Actions emulator).",
    )
    parser.add_argument(
        "--uv-install-smoke",
        action="store_true",
        help="Run the optional uv install smoke and pip-vs-uv install timing lane.",
    )
    parser.add_argument(
        "--adr030-ratification",
        action="store_true",
        help="Run the focused ADR-030 ratification lane and timing report.",
    )
    parser.add_argument(
        "--deprecation-closure",
        action="store_true",
        help="Run the pre-v1.0 deprecation-closure lane and timing report.",
    )
    parser.add_argument(
        "--deprecation-ledger",
        action="store_true",
        help="Run only the ADR-011 deprecation ledger gate (PR-profile check).",
    )
    parser.add_argument(
        "--release-prepare-files",
        action="store_true",
        help=(
            "Prepare deterministic release files before checklist closure; the strict "
            "release-preflight must still run before publication."
        ),
    )
    parser.add_argument(
        "--release-preflight",
        action="store_true",
        help="Run the strict pre-step-11 release gate and write the handoff report.",
    )
    parser.add_argument(
        "--release-finalize",
        action="store_true",
        help="Validate the latest release-preflight snapshot before manual release steps.",
    )
    parser.add_argument(
        "--release-postcommit",
        action="store_true",
        help=(
            "Run the automatable post-publish steps (release.md steps 14-17): PyPI page "
            "verification, clean-venv install smoke test, plan archive, dev version bump."
        ),
    )
    parser.add_argument(
        "--release-version",
        default=None,
        help=(
            "Optional exact target version for release-preflight; otherwise discover the active "
            "plan from the project version and infer its declared release version."
        ),
    )
    parser.add_argument(
        "--next-version",
        default=None,
        help=(
            "Optional next release label/version for release-postcommit; otherwise the next "
            "patch version is used. Never invents a minor/major bump."
        ),
    )
    parser.add_argument(
        "--release-date",
        default=None,
        help="Optional YYYY-MM-DD release date override for reproducible release-file updates.",
    )
    parser.add_argument(
        "--print-mypy-targets",
        action="store_true",
        help=(
            "Print the declared-scope mypy target files (one per line), derived from "
            "the pyproject.toml strict-mypy override, and exit. Lets CI consume the "
            "same source of truth as the local gate (v0.11.6 Task 58 / pre-v5 M2)."
        ),
    )
    args = parser.parse_args()

    if args.print_mypy_targets:
        for target in _mypy_targets():
            print(target)
        return 0
    if args.instruction_index_sync:
        # Indexes and platform skill packages derive from the same canonical
        # sources, so one local regeneration command refreshes both.
        return subprocess.call(  # noqa: S603
            _python_cmd("scripts/quality/sync_instruction_indexes.py", "--write")
        ) or subprocess.call(  # noqa: S603
            _python_cmd("scripts/quality/sync_skill_packages.py", "--write")
        )
    if args.uv_install_smoke:
        return _run_uv_install_smoke()
    if args.adr030_ratification:
        return run_adr030_ratification()
    if args.deprecation_closure:
        return run_deprecation_closure()
    if args.deprecation_ledger:
        return run_deprecation_ledger_gate()
    if args.release_prepare_files:
        return run_release_prepare_files(
            plan_path=args.plan,
            release_version=args.release_version,
            release_date=args.release_date,
        )
    if args.release_preflight:
        return run_release_preflight(
            plan_path=args.plan,
            release_version=args.release_version,
            release_date=args.release_date,
        )
    if args.release_finalize:
        return run_release_finalize(plan_path=args.plan)
    if args.release_postcommit:
        return run_release_postcommit(
            plan_path=args.plan,
            next_version=args.next_version,
            release_date=args.release_date,
        )

    if args.ci_parity:
        print("Run-block smoke mode: delegating to scripts/run_ci_locally.py")
        shell_arg = "bash"
        if os.name == "nt":
            shell_arg = "bash"
        with suppress(Exception):
            for cov in Path(".").glob(".coverage*"):
                with suppress(Exception):
                    cov.unlink()
        return subprocess.call([sys.executable, "scripts/run_ci_locally.py", "--shell", shell_arg])  # noqa: S603

    selected_profile = args.profile or ("pr" if args.skip_main else "full")
    mypy_available = (
        shutil.which("mypy") is not None
        or subprocess.call(
            [sys.executable, "-m", "mypy", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )
    if not mypy_available:
        print("ERROR: mypy not found in current environment.")
        return 2

    pre_commit_available = shutil.which("pre-commit") is not None
    if selected_profile in {"pr", "full", "release"} and not pre_commit_available:
        print("WARNING: pre-commit not found in current environment; skipping the pre-commit step.")

    if not _pytest_supports_no_cov():
        print("WARNING: pytest-cov/--no-cov unavailable; running pytest commands without --no-cov.")

    mypy_targets = _mypy_targets()
    if not mypy_targets:
        print("No mypy target files found; skipping mypy step.")
    try:
        lint_targets = (
            _task_specific_lint_targets(args.task, plan_path=args.plan)
            if selected_profile == "task" and args.task is not None
            else _changed_python_targets()
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    try:
        build_kwargs = {
            "task": args.task,
            "mypy_targets": mypy_targets,
            "lint_targets": lint_targets,
            "pre_commit_available": pre_commit_available,
        }
        if args.plan is not None:
            build_kwargs["plan_path"] = args.plan
        plan = build_profile_plan(
            selected_profile,
            **build_kwargs,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    task_report_path = (
        Path("reports/local_checks/task_profile_report.json")
        if selected_profile == "task"
        else None
    )
    rc = run_profile_plan(plan, task_report_path=task_report_path, requested_paths=args.paths)
    if rc != 0:
        return rc

    print(f"\nLocal checks completed ({selected_profile} profile).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
