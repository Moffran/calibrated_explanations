"""Run GitHub Actions workflow shell steps locally.

This helper parses YAML workflow files under `.github/workflows/` and
extracts `run:` step bodies. It then executes them sequentially in a
local shell so you can catch CI failures before opening a PR.

Features:
- Dynamically discovers workflows in `.github/workflows/*.yml`.
- Extracts `run:` blocks from each job's `steps` and preserves `env`.
- Skips steps that use actions (e.g. `uses: actions/checkout`) but
  warns so you can run any required setup manually.
- Supports `--shell` (bash|pwsh) and `--dry-run`.
- By default runs all workflows; select workflows with `--workflow`.

Notes:
- Many workflow `run` blocks assume a Linux bash environment (set -e,
  Bash-specific operators). On Windows prefer running under WSL/Git-Bash
  and pass `--shell bash`.
- The script does not attempt to emulate `actions/setup-python`; it will
  only run the commands as-is in your current environment.

Usage examples:
  # Show the commands that would run
  python scripts/run_ci_locally.py --dry-run

  # Run lint and test workflows (may be slow)
  python scripts/run_ci_locally.py --workflow lint --workflow test

  # Run in PowerShell explicitly
  python scripts/run_ci_locally.py --shell pwsh
"""
from __future__ import annotations

import argparse
import glob
import os
import shlex
import subprocess
import sys
import tempfile
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception as exc:  # pragma: no cover - dev dependency expected
    print("PyYAML is required (package name: pyyaml). Please install it.")
    raise


def find_workflow_files(path: str = ".github/workflows") -> List[str]:
    pattern = os.path.join(path, "*.yml")
    files = glob.glob(pattern)
    pattern2 = os.path.join(path, "*.yaml")
    files += glob.glob(pattern2)
    files.sort()
    return files


def load_workflow(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def extract_run_steps(workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of dicts: {job, step_idx, name, env, run}.

    Only steps that have a `run` key are returned. Steps that use
    `uses:` are skipped (these are actions that run on CI).
    """
    out: List[Dict[str, Any]] = []
    jobs = workflow.get("jobs") or {}
    for job_name, job in jobs.items():
        steps = job.get("steps") or []
        job_env = job.get("env") or {}
        # Track whether this job used actions/setup-python earlier
        setup_python_found = False
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            # Detect `uses: actions/setup-python@...` so we can emulate it
            if "uses" in step and isinstance(step.get("uses"), str):
                if "actions/setup-python" in step.get("uses"):
                    setup_python_found = True
                # We don't add 'uses' steps to run list; continue scanning
                continue
            if "run" in step:
                # Merge job env, then step env
                step_env = dict(job_env)
                step_env.update(step.get("env") or {})
                out.append(
                    {
                        "job": job_name,
                        "step_idx": idx,
                        "name": step.get("name") or f"step_{idx}",
                        "env": step_env,
                        "run": step["run"],
                        "setup_python": setup_python_found,
                    }
                )
    return out


def collect_all_runs(workflow_files: List[str], selected: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    collected: Dict[str, List[Dict[str, Any]]] = {}
    for wf in workflow_files:
        base = os.path.basename(wf)
        name = os.path.splitext(base)[0]
        if selected and name not in selected:
            continue
        try:
            data = load_workflow(wf)
        except Exception as exc:
            print(f"Failed to parse {wf}: {exc}")
            continue
        steps = extract_run_steps(data)
        if steps:
            collected[name] = steps
    return collected


def run_script_block(script: str, env: Dict[str, str], shell: str, cwd: Optional[str] = None) -> int:
    # Write script to a temporary file and execute with chosen shell.
    # When running under Windows+bash, create the temp file inside the
    # requested working directory so we can invoke it by a relative path
    # (avoids path conversion/mount issues with MSYS/MinGW).
    temp_dir = cwd or os.getcwd()
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh", dir=temp_dir) as fh:
        fh.write(script)
        tmp = fh.name
    try:
        if shell == "bash":
            if os.name == "nt":
                # Use a relative path from the cwd so bash will resolve it
                # correctly when invoked with cwd set to the repo root.
                rel = os.path.relpath(tmp, start=temp_dir)
                cmd = ["bash", rel]
            else:
                cmd = ["bash", tmp]
        else:
            # PowerShell expects a .ps1 file. Rewrite extension and invoke pwsh.
            ps_tmp = tmp + ".ps1"
            os.rename(tmp, ps_tmp)
            tmp = ps_tmp
            cmd = ["pwsh", "-NoProfile", "-NonInteractive", "-File", tmp]

        print("-> Executing:", cmd)
        proc = subprocess.run(cmd, env={**os.environ, **env}, cwd=cwd)
        return proc.returncode
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run CI workflow steps locally")
    parser.add_argument("--workflow", action="append", help="Workflow basename to run (without extension)")
    default_shell = "pwsh" if os.name == "nt" else "bash"
    parser.add_argument("--shell", choices=("bash", "pwsh"), default=default_shell, help="Shell to use for running steps")
    parser.add_argument("--dry-run", action="store_true", help="Only print discovered steps without executing")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running other steps if one fails")
    parser.add_argument("--cwd", default=".", help="Working directory for commands (defaults to repo root)")
    args = parser.parse_args(argv)

    workflow_files = find_workflow_files()
    if not workflow_files:
        print("No workflow files found under .github/workflows/")
        return 2

    selected = args.workflow
    collected = collect_all_runs(workflow_files, selected)
    if not collected:
        print("No runnable steps found in the selected workflows.")
        return 0

    print("Discovered the following runnable steps:")
    for wf_name, steps in collected.items():
        print(f"\nWorkflow: {wf_name}")
        for s in steps:
            heading = f"  [{s['job']}] {s['name']}"
            print(heading)
            # Print the first line of the run block for brevity
            snippet = s["run"].splitlines()
            preview = snippet[0] if snippet else "<empty>"
            print("    ", preview)

    if args.dry_run:
        print("\nDry-run complete. No commands executed.")
        return 0

    print("\nRunning steps now. Press Ctrl-C to abort.")
    results: List[Dict[str, Any]] = []
    abort = False
    first_nonzero_rc = 0
    for wf_name, steps in collected.items():
        print(f"\n=== Workflow: {wf_name} ===")
        for s in steps:
            print(f"\n--- Job: {s['job']} | Step: {s['name']} ---")
            rc = run_script_block(
                s["run"], {k: str(v) for k, v in (s.get("env") or {}).items()}, args.shell, cwd=args.cwd
            )
            results.append({
                "workflow": wf_name,
                "job": s["job"],
                "step": s["name"],
                "rc": rc,
                "run": s["run"],
                "setup_python": s.get("setup_python", False),
            })
            if rc != 0:
                print(f"Step failed with exit code {rc}: {s['name']}")
                if first_nonzero_rc == 0:
                    first_nonzero_rc = rc
                if not args.continue_on_error:
                    abort = True
                    break
        if abort:
            break

    # Summary
    total = len(results)
    failed = [r for r in results if r["rc"] != 0]
    succeeded = [r for r in results if r["rc"] == 0]

    print("\n\nCI Local Run Summary")
    print("--------------------")
    print(f"Total steps run: {total}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailing steps:")
        for r in failed:
            preview = (r["run"] or "").splitlines()
            snippet = preview[0] if preview else "<empty>"
            print(f"- {r['workflow']} :: {r['job']} :: {r['step']} (rc={r['rc']})")
            print(f"    command: {snippet}")

        # Heuristic: highlight failing 'test' related steps that likely need attention
        test_keywords = [
            "pytest",
            "test-cov",
            "test",
            "pydocstyle",
            "mypy",
            "ruff",
            "nbqa",
            "check_coverage",
            "check_docstring",
            "micro_bench_perf",
        ]
        important = []
        for r in failed:
            run_lower = (r["run"] or "").lower()
            name_lower = (r["step"] or "").lower()
            if any(k in run_lower or k in name_lower for k in test_keywords):
                important.append(r)

        if important:
            print("\nCI test failures likely needing attention:")
            for r in important:
                print(f"- {r['workflow']} :: {r['job']} :: {r['step']} (rc={r['rc']})")
                print(f"    full command excerpt:\n      " + "\n      ".join((r['run'] or "").splitlines()[:5]))
    else:
        print("\nAll steps completed successfully.")

    # Write a machine-readable summary to `reports/ci_local_summary.json`.
    try:
        reports_dir = os.path.join(args.cwd or ".", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        summary_path = os.path.join(reports_dir, "ci_local_summary.json")
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "cwd": os.path.abspath(args.cwd or "."),
            "shell": args.shell,
            "workflows": list(collected.keys()),
            "total_steps": total,
            "succeeded": len(succeeded),
            "failed": len(failed),
            "failed_steps": [
                {
                    "workflow": r["workflow"],
                    "job": r["job"],
                    "step": r["step"],
                    "rc": r["rc"],
                    "command_preview": (r["run"] or "").splitlines()[:5],
                }
                for r in failed
            ],
            "important_failures": [
                {
                    "workflow": r["workflow"],
                    "job": r["job"],
                    "step": r["step"],
                    "rc": r["rc"],
                    "command_preview": (r["run"] or "").splitlines()[:10],
                }
                for r in (important if failed else [])
            ],
        }
        # Optionally run pre-commit and include its result in the summary.
        try:
            preproc = subprocess.run(["pre-commit", "run", "--all-files"], capture_output=True, text=True)
            pre_rc = preproc.returncode
            summary["pre_commit"] = {
                "rc": pre_rc,
                "stdout_tail": preproc.stdout.splitlines()[-50:],
                "stderr_tail": preproc.stderr.splitlines()[-50:],
            }
            print(f"\npre-commit exit code: {pre_rc}")
        except Exception as exc:  # pragma: no cover - best-effort
            summary["pre_commit"] = {"error": str(exc)}
            print(f"Failed to run pre-commit: {exc}")

        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote CI local summary to: {summary_path}")
    except Exception as exc:  # pragma: no cover - best-effort reporting
        print(f"Failed to write summary JSON: {exc}")

    return first_nonzero_rc


if __name__ == "__main__":
    raise SystemExit(main())
