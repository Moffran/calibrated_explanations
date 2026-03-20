"""Run stacked CI checks in the current local environment.

This runner mirrors the command-level checks from CI workflows but deliberately
skips environment/bootstrap steps (no virtualenv creation, no pip install).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    """A single local check step."""

    name: str
    command: list[str]
    optional: bool = False


def _python_cmd(*args: str) -> list[str]:
    return [sys.executable, *args]


def _is_pre_commit_step(step: Step) -> bool:
    if not step.command:
        return False
    head = Path(step.command[0]).name.lower()
    return head in {"pre-commit", "pre-commit.exe"}


def _mypy_targets() -> list[str]:
    candidates = [
        "src/calibrated_explanations/core/exceptions.py",
        "src/calibrated_explanations/core/validation.py",
        "src/calibrated_explanations/api/params.py",
    ]
    return [path for path in candidates if Path(path).is_file()]


def _run_step(step: Step) -> int:
    cmd_text = " ".join(step.command)
    print(f"\n[{step.name}]")
    print(f"$ {cmd_text}")
    env = dict(os.environ)
    env.setdefault("PRE_COMMIT_HOME", str(Path(".cache/pre-commit").resolve()))
    if _is_pre_commit_step(step):
        result = subprocess.run(step.command, check=False, env=env, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    else:
        result = subprocess.run(step.command, check=False, env=env)
    if result.returncode == 0:
        return 0
    # Pre-commit may fail due to network fetch issues in restricted/offline
    # environments. Keep that case non-fatal so local stacks can still run.
    if _is_pre_commit_step(step):
        combined = f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}"
        if _is_network_fetch_failure(combined):
            print("Pre-commit could not fetch hook repos (offline/network-restricted). Continuing with stacked checks.")
            return 0
    if step.optional:
        print(f"Step failed but is advisory/optional: {step.name} (rc={result.returncode})")
        return 0
    return result.returncode


def _run_micro_benchmark() -> int:
    print("\n[Micro benchmark]")
    print(f"$ {sys.executable} scripts/perf/micro_bench_perf.py > tests/benchmarks/micro_current.json")
    env = dict(os.environ)
    env.setdefault("PRE_COMMIT_HOME", str(Path(".cache/pre-commit").resolve()))
    output = subprocess.run(
        _python_cmd("scripts/perf/micro_bench_perf.py"),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    Path("tests/benchmarks").mkdir(parents=True, exist_ok=True)
    Path("tests/benchmarks/micro_current.json").write_text(
        output.stdout,
        encoding="utf-8",
        newline="\n",
    )
    if output.returncode != 0:
        if output.stderr:
            print(output.stderr)
        return output.returncode
    return 0


def _is_network_fetch_failure(stderr: str) -> bool:
    text = (stderr or "").lower()
    if "unable to access 'https://github.com" in text:
        return True
    if "failed to connect to github.com" in text:
        return True
    if "could not connect to server" in text:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CI-equivalent checks locally in current env.")
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help="Run PR checks only (skip main-branch checks such as coverage/perf/over-testing).",
    )
    parser.add_argument(
        "--ci-parity",
        action="store_true",
        help="Run docs/linkcheck in strict CI parity mode (make linkcheck fatal).",
    )
    args = parser.parse_args()

    # CI-parity mode: dynamically read CI workflows and run them locally.
    if args.ci_parity:
        print("CI parity mode: delegating to scripts/run_ci_locally.py (dynamic workflow runner)")
        # Prefer bash to emulate GitHub Actions Linux runners; fall back to pwsh on Windows.
        shell_arg = "bash"
        if os.name == "nt":
            shell_arg = "bash"
        # Pre-clean any stale coverage database files which can cause sqlite schema
        # errors when pytest-cov writes coverage data from parallel runs.
        try:
            for cov in Path('.').glob('.coverage*'):
                try:
                    cov.unlink()
                except Exception:
                    pass
        except Exception:
            pass
        rc = subprocess.call([sys.executable, "scripts/run_ci_locally.py", "--shell", shell_arg])
        return rc

    if shutil.which("mypy") is None:
        print("ERROR: mypy not found in current environment.")
        return 2
    if shutil.which("pre-commit") is None:
        print("ERROR: pre-commit not found in current environment.")
        return 2

    mypy_targets = _mypy_targets()
    if not mypy_targets:
        print("No mypy target files found; skipping mypy step.")

    pr_steps: list[Step] = [
        Step("Pre-commit", ["pre-commit", "run", "--all-files"]),
        Step("Ruff naming", ["ruff", "check", "--select", "N"]),
        Step("Notebook naming lint", _python_cmd("-m", "nbqa", "ruff", "notebooks", "--select", "N")),
        Step("Pydocstyle", ["pydocstyle", "src", "tests"]),
        Step(
            "Docstring coverage",
            _python_cmd("scripts/quality/check_docstring_coverage.py", "--fail-under", "94.0"),
        ),
        Step("ADR-001 boundary check", _python_cmd("scripts/quality/check_import_graph.py")),
        Step("ADR-002 compliance check", _python_cmd("scripts/quality/check_adr002_compliance.py")),
        Step(
            "ADR-028 governance event schema check",
            _python_cmd("scripts/quality/check_governance_event_schema.py"),
        ),
        Step(
            "Agent instruction consistency",
            _python_cmd("scripts/quality/check_agent_instruction_consistency.py"),
        ),
        Step(
            "CI policy workflow validation (advisory)",
            _python_cmd(
                "scripts/quality/validate_ci_policy.py",
                "--base-sha",
                "HEAD~1",
                "--head-sha",
                "HEAD",
                "--advisory",
            ),
            optional=True,
        ),
        Step("Core tests (no viz/no cov)", ["pytest", "-q", "-m", "not viz", "--no-cov"]),
        Step("Private-member scan", _python_cmd("scripts/anti-pattern-analysis/scan_private_usage.py", "tests", "--check")),
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
    ]

    if mypy_targets:
        pr_steps.insert(
            6,
            Step(
                "Mypy (Phase 1B scope)",
                ["mypy", *mypy_targets, "--config-file", "pyproject.toml"],
            ),
        )

    # Additional PR-scoped CI checks mirrored from workflows
    optional_pr_steps: list[Step] = [
            # Deprecation-sensitive tests (treat deprecations as errors)
            Step(
                "Deprecation-sensitive tests",
                _python_cmd(
                    "-c",
                    (
                        "import os,sys,subprocess;"
                        "os.environ.setdefault('CE_DEPRECATIONS','error');"
                        "sys.exit(subprocess.call(['pytest','tests/unit','-m','not viz','-q','--maxfail=1','--no-cov']))"
                    ),
                ),
                optional=True,
            ),
            # Dependency audit (pip-audit) — advisory locally
            Step(
                "Dependency audit",
                [
                    "pip-audit",
                    "-r",
                    "requirements.txt",
                    "-r",
                    "docs/requirements-doc.txt",
                    "--ignore-vuln",
                    "GHSA-xm59-rqc7-hhvf",
                ],
                optional=True,
            ),
            # Notebook audit (advisory locally)
            Step(
                "Notebook audit",
                _python_cmd("scripts/quality/audit_notebook_api.py", "notebooks", "--json", "artifacts/notebook_audit.json"),
                optional=True,
            ),
            # Docs build (advisory locally)
            Step(
                "Docs build (HTML)",
                ["sphinx-build", "-b", "html", "docs", "docs/_build/html"],
                optional=not args.ci_parity,
            ),
            Step(
                "Docs linkcheck",
                [
                    "sphinx-build",
                    "-b",
                    "linkcheck",
                    "-D",
                    "nbsphinx_execute=never",
                    "docs",
                    "docs/_build/linkcheck",
                ],
                optional=not args.ci_parity,
            ),
        ]
    if Path("tests/examples").exists():
        optional_pr_steps.append(
            Step(
                "Examples smoke",
                ["pytest", "-q", "tests/examples"],
                optional=True,
            )
        )
    pr_steps.extend(optional_pr_steps)

    main_steps: list[Step] = [
        # Mirror the dependency audit and docs/notebook checks present on CI
        Step(
            "Dependency audit (main)",
            [
                "pip-audit",
                "-r",
                "requirements.txt",
                "-r",
                "docs/requirements-doc.txt",
                "--ignore-vuln",
                "GHSA-xm59-rqc7-hhvf",
            ],
            optional=True,
        ),
        Step(
            "Notebook audit (main)",
            _python_cmd("scripts/quality/audit_notebook_api.py", "notebooks", "--json", "artifacts/notebook_audit.json"),
            optional=True,
        ),
        Step(
            "Docs build (main)",
            ["sphinx-build", "-b", "html", "docs", "docs/_build/html"],
            optional=True,
        ),
        Step(
            "Core tests with coverage",
            [
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
        Step("Per-module coverage gates", _python_cmd("scripts/quality/check_coverage_gates.py", "coverage.xml")),
        Step(
            "Perf thresholds",
            _python_cmd(
                "scripts/perf/check_perf_micro.py",
                "tests/benchmarks/micro_current.json",
                "tests/benchmarks/perf_thresholds.json",
            ),
        ),
        Step(
            "Private-member scan (main)",
            _python_cmd("scripts/anti-pattern-analysis/scan_private_usage.py", "tests", "--check"),
        ),
        Step(
            "ADR-030 anti-pattern detector (main)",
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
            "ADR-030 test-helper export guard (main)",
            _python_cmd(
                "scripts/quality/check_no_test_helper_exports.py",
                "--root",
                "src/calibrated_explanations",
                "--report",
                "reports/anti-pattern-analysis/test_helper_wrapper_report.json",
            ),
        ),
        Step(
            "ADR-006 trust-mutation primitive guard (main)",
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
            "ADR-030 marker hygiene (main)",
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
            "ADR-028 governance event schema check (main)",
            _python_cmd("scripts/quality/check_governance_event_schema.py"),
        ),
        Step(
            "Over-testing coverage contexts",
            [
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "--cov=src/calibrated_explanations",
                "--cov-config=pyproject.toml",
                "--cov-context=test",
                "--no-cov-on-fail",
            ],
            optional=True,
        ),
        Step(
            "Over-testing report",
            _python_cmd(
                "scripts/over_testing/over_testing_report.py",
                "--require-multiple-contexts",
                "--output-lines",
                "reports/over_testing/line_coverage_counts.csv",
                "--output-blocks",
                "reports/over_testing/block_coverage_counts.csv",
                "--output-summary",
                "reports/over_testing/summary.json",
                "--output-metadata",
                "reports/over_testing/metadata.json",
            ),
            optional=True,
        ),
        Step("Redundant tests report", _python_cmd("scripts/over_testing/detect_redundant_tests.py"), optional=True),
    ]

    for step in pr_steps:
        rc = _run_step(step)
        if rc != 0:
            return rc

    if args.skip_main:
        final_precommit = Step("Pre-commit (final verification)", ["pre-commit", "run", "--all-files"])
        rc = _run_step(final_precommit)
        if rc != 0:
            return rc
        print("\nLocal checks completed (PR scope).")
        return 0

    rc = _run_micro_benchmark()
    if rc != 0:
        return rc

    for step in main_steps:
        rc = _run_step(step)
        if rc != 0:
            return rc

    final_precommit = Step("Pre-commit (final verification)", ["pre-commit", "run", "--all-files"])
    rc = _run_step(final_precommit)
    if rc != 0:
        return rc

    print("\nLocal checks completed (PR + main scope).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
