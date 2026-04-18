"""Deterministic notebook execution driver for docs CI (ADR-012).

Executes notebooks discovered under a root directory, enforces per-cell and
per-notebook runtime ceilings, honours ``ce_skip`` metadata tags, and emits a
structured JSON report to ``reports/docs/notebook_execution_report.json``.

ADR compliance
--------------
- ADR-012: advisory on OSS/mainline, blocking on release/stable branches.
- ADR-010: requires ``[notebooks,viz]`` extras; no core dependency expansion.

Skip tags (set in notebook-level ``metadata.ce_skip``)
-------------------------------------------------------
- ``"noexec"`` — always skip; emits ``skipped_noexec``.
- ``"slow"``   — skip; emits ``skipped_slow``.  In blocking mode, slow-notebook
  inclusion/exclusion must follow documented branch policy.
- Any other value — policy violation in blocking mode; warn and execute in
  advisory mode.

Report fields per notebook
--------------------------
``notebook``, ``status``, ``elapsed_seconds``, ``errors``,
``invocation_id``, ``skip_reason``, ``mode``.

Status vocabulary
-----------------
``passed``, ``failed``, ``timed_out``, ``skipped_noexec``, ``skipped_slow``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

VALID_STATUSES: frozenset[str] = frozenset(
    {"passed", "failed", "timed_out", "skipped_noexec", "skipped_slow"}
)
KNOWN_SKIP_TAGS: frozenset[str] = frozenset({"noexec", "slow"})
REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "notebook",
    "status",
    "elapsed_seconds",
    "errors",
    "invocation_id",
    "skip_reason",
    "mode",
)


def discover_notebooks(root: Path) -> list[Path]:
    """Return ``.ipynb`` paths under *root*, sorted for deterministic traversal.

    Checkpoint files under ``.ipynb_checkpoints/`` directories are excluded.

    Parameters
    ----------
    root : Path
        Directory to search recursively.

    Returns
    -------
    list[Path]
        Sorted list of notebook paths.
    """
    return sorted(
        p for p in root.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts
    )


def read_skip_tag(nb_metadata: dict[str, Any]) -> str | None:
    """Return the ``ce_skip`` tag from notebook-level metadata, or ``None``.

    Parameters
    ----------
    nb_metadata : dict[str, Any]
        The top-level ``metadata`` dict of an ``nbformat`` notebook.

    Returns
    -------
    str | None
        The tag value (e.g. ``"noexec"``, ``"slow"``), or ``None`` if absent.
    """
    return nb_metadata.get("ce_skip")


def extract_errors(nb: Any) -> list[dict[str, Any]]:
    """Extract error output records from all cells of an executed notebook.

    Parameters
    ----------
    nb : Any
        An ``nbformat`` notebook node (duck-typed as a dict).

    Returns
    -------
    list[dict[str, Any]]
        List of ``{cell, etype, evalue}`` dicts, one per error output.
    """
    errors: list[dict[str, Any]] = []
    for i, cell in enumerate(nb.get("cells", [])):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    {
                        "cell": i,
                        "etype": output.get("ename", ""),
                        "evalue": output.get("evalue", ""),
                    }
                )
    return errors


def _execute_with_notebook_timeout(
    nb: Any,
    resources: dict[str, Any],
    preprocessor: Any,
    notebook_timeout: float,
) -> tuple[Any, dict[str, Any]]:
    """Execute *nb* through *preprocessor*, aborting after *notebook_timeout* seconds.

    Parameters
    ----------
    nb : Any
        Notebook node to execute.
    resources : dict[str, Any]
        nbconvert resource dict.
    preprocessor : Any
        Configured ``ExecutePreprocessor`` instance.
    notebook_timeout : float
        Maximum wall-clock seconds allowed for the whole notebook.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        ``(nb_out, resources_out)`` on success.

    Raises
    ------
    TimeoutError
        If the notebook-level wall-clock deadline is exceeded.
    Exception
        Any exception raised during cell execution (e.g.
        ``CellExecutionError``).
    """
    result: list[Any] = [None, None]
    exc_holder: list[BaseException | None] = [None]

    def _run() -> None:
        try:
            result[0], result[1] = preprocessor.preprocess(nb, resources)
        except BaseException as exc:  # noqa: BLE001
            exc_holder[0] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=notebook_timeout)

    if thread.is_alive():
        # Shut down the Jupyter kernel before raising so the subprocess does
        # not leak across subsequent notebooks or CI jobs.
        try:
            km = getattr(preprocessor, "km", None)
            if km is not None:
                km.shutdown_kernel(now=True)
        except Exception:  # noqa: BLE001
            pass
        raise TimeoutError(
            f"Notebook exceeded {notebook_timeout}s wall-clock timeout"
        )
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result[0], result[1]


def validate_report_schema(report: dict[str, Any]) -> list[str]:
    """Validate a report dict for required fields and closed status vocabulary.

    Parameters
    ----------
    report : dict[str, Any]
        Report dict as produced by :func:`run_notebooks`.

    Returns
    -------
    list[str]
        Validation error strings; empty list if the report is valid.
    """
    errs: list[str] = []
    for i, rec in enumerate(report.get("notebooks", [])):
        for field in REQUIRED_RECORD_FIELDS:
            if field not in rec:
                errs.append(f"Record {i}: missing required field '{field}'")
        status = rec.get("status", "")
        if status not in VALID_STATUSES:
            errs.append(
                f"Record {i}: unknown status '{status}'. "
                f"Valid statuses: {sorted(VALID_STATUSES)}"
            )
    return errs


def run_notebooks(
    notebooks_dir: Path,
    output_path: Path,
    cell_timeout: int,
    notebook_timeout: int,
    mode: str,
) -> int:
    """Execute notebooks and write a structured JSON execution report.

    Parameters
    ----------
    notebooks_dir : Path
        Root directory to discover ``*.ipynb`` files.
    output_path : Path
        Destination for the JSON execution report.
    cell_timeout : int
        Per-cell execution timeout in seconds.
    notebook_timeout : int
        Per-notebook wall-clock timeout in seconds.
    mode : str
        ``"advisory"`` or ``"blocking"``.  Blocking mode exits with code 1 on
        any failure; advisory mode always exits with code 0.

    Returns
    -------
    int
        ``0`` if all notebooks passed or were skipped; ``1`` if any failed
        and *mode* is ``"blocking"``.
    """
    try:
        import nbformat  # noqa: PLC0415
        from nbconvert.preprocessors import (  # noqa: PLC0415
            CellExecutionError,
            ExecutePreprocessor,
        )

        try:
            from nbconvert.preprocessors.execute import (  # noqa: PLC0415
                CellTimeoutError as _CellTimeoutError,
            )
        except ImportError:
            _CellTimeoutError = None
    except ImportError as exc:
        print(
            f"ERROR: nbconvert is not installed. "
            f"Install calibrated_explanations[notebooks]: {exc}",
            file=sys.stderr,
        )
        return 1

    invocation_id = str(uuid.uuid4())
    notebooks = discover_notebooks(notebooks_dir)

    if not notebooks:
        msg = f"No notebooks found under '{notebooks_dir}'."
        if mode == "blocking":
            print(f"ERROR: {msg}", file=sys.stderr)
            return 1
        warnings.warn(msg, UserWarning, stacklevel=2)

    records: list[dict[str, Any]] = []
    any_failed = False

    for nb_path in notebooks:
        if nb_path.is_absolute():
            try:
                rel_path = str(nb_path.relative_to(Path.cwd()))
            except ValueError:
                rel_path = str(nb_path)
        else:
            rel_path = str(nb_path)
        start = time.monotonic()

        # ------------------------------------------------------------------ #
        # Read notebook                                                        #
        # ------------------------------------------------------------------ #
        try:
            nb = nbformat.read(str(nb_path), as_version=4)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            records.append(
                {
                    "notebook": rel_path,
                    "status": "failed",
                    "elapsed_seconds": round(elapsed, 3),
                    "errors": [
                        {
                            "cell": -1,
                            "etype": type(exc).__name__,
                            "evalue": str(exc),
                        }
                    ],
                    "invocation_id": invocation_id,
                    "skip_reason": None,
                    "mode": mode,
                }
            )
            any_failed = True
            continue

        # ------------------------------------------------------------------ #
        # Skip-tag resolution                                                  #
        # ------------------------------------------------------------------ #
        nb_meta: dict[str, Any] = nb.get("metadata", {})
        skip_tag = read_skip_tag(nb_meta)

        if skip_tag is not None:
            if skip_tag not in KNOWN_SKIP_TAGS:
                if mode == "blocking":
                    elapsed = time.monotonic() - start
                    records.append(
                        {
                            "notebook": rel_path,
                            "status": "failed",
                            "elapsed_seconds": round(elapsed, 3),
                            "errors": [
                                {
                                    "cell": -1,
                                    "etype": "PolicyViolationError",
                                    "evalue": (
                                        f"Unknown ce_skip tag '{skip_tag}'. "
                                        "Only 'noexec' and 'slow' are valid. "
                                        "This is a policy violation in blocking mode."
                                    ),
                                }
                            ],
                            "invocation_id": invocation_id,
                            "skip_reason": f"unknown_tag:{skip_tag}",
                            "mode": mode,
                        }
                    )
                    any_failed = True
                    continue
                else:
                    warnings.warn(
                        f"Unknown ce_skip tag '{skip_tag}' in notebook "
                        f"'{nb_path.name}'. Proceeding with execution in "
                        "advisory mode.",
                        UserWarning,
                        stacklevel=2,
                    )
                    # Fall through to execution

            elif skip_tag == "noexec":
                elapsed = time.monotonic() - start
                records.append(
                    {
                        "notebook": rel_path,
                        "status": "skipped_noexec",
                        "elapsed_seconds": round(elapsed, 3),
                        "errors": [],
                        "invocation_id": invocation_id,
                        "skip_reason": "noexec",
                        "mode": mode,
                    }
                )
                continue

            else:  # skip_tag == "slow"
                elapsed = time.monotonic() - start
                records.append(
                    {
                        "notebook": rel_path,
                        "status": "skipped_slow",
                        "elapsed_seconds": round(elapsed, 3),
                        "errors": [],
                        "invocation_id": invocation_id,
                        "skip_reason": "slow",
                        "mode": mode,
                    }
                )
                continue

        # ------------------------------------------------------------------ #
        # Execute notebook                                                     #
        # ------------------------------------------------------------------ #
        ep = ExecutePreprocessor(timeout=cell_timeout, kernel_name="python3")
        resources = {"metadata": {"path": str(nb_path.parent)}}
        status = "passed"
        errors: list[dict[str, Any]] = []

        nb_out: Any = nb
        try:
            nb_out, _ = _execute_with_notebook_timeout(nb, resources, ep, notebook_timeout)
        except TimeoutError as exc:
            elapsed = time.monotonic() - start
            status = "timed_out"
            errors = [
                {
                    "cell": -1,
                    "etype": "TimeoutError",
                    "evalue": str(exc),
                }
            ]
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            is_cell_timeout = (
                _CellTimeoutError is not None
                and isinstance(exc, _CellTimeoutError)
            )
            if is_cell_timeout:
                status = "timed_out"
                errors = [
                    {
                        "cell": -1,
                        "etype": "CellTimeoutError",
                        "evalue": str(exc)[:500],
                    }
                ]
            else:
                status = "failed"
                errors = [
                    {
                        "cell": -1,
                        "etype": type(exc).__name__,
                        "evalue": str(exc)[:500],
                    }
                ]
        else:
            elapsed = time.monotonic() - start
            # Collect any error outputs left in cells by the executed notebook.
            cell_errors = extract_errors(nb_out)
            if cell_errors:
                status = "failed"
                errors = cell_errors

        if status in {"failed", "timed_out"}:
            any_failed = True

        records.append(
            {
                "notebook": rel_path,
                "status": status,
                "elapsed_seconds": round(elapsed, 3),
                "errors": errors,
                "invocation_id": invocation_id,
                "skip_reason": None,
                "mode": mode,
            }
        )

    summary: dict[str, Any] = {
        "invocation_id": invocation_id,
        "total": len(records),
        "passed": sum(1 for r in records if r["status"] == "passed"),
        "failed": sum(1 for r in records if r["status"] == "failed"),
        "timed_out": sum(1 for r in records if r["status"] == "timed_out"),
        "skipped_noexec": sum(
            1 for r in records if r["status"] == "skipped_noexec"
        ),
        "skipped_slow": sum(
            1 for r in records if r["status"] == "skipped_slow"
        ),
        "mode": mode,
    }

    report: dict[str, Any] = {"summary": summary, "notebooks": records}

    schema_errs = validate_report_schema(report)
    if schema_errs:
        for err in schema_errs:
            print(f"SCHEMA ERROR: {err}", file=sys.stderr)
        if mode == "blocking":
            return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Notebook execution report written to {output_path}")
    print(
        f"  total={summary['total']}  passed={summary['passed']}  "
        f"failed={summary['failed']}  timed_out={summary['timed_out']}  "
        f"skipped_noexec={summary['skipped_noexec']}  "
        f"skipped_slow={summary['skipped_slow']}"
    )

    if mode == "blocking" and any_failed:
        return 1
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Execute notebooks and emit a structured execution report. "
            "ADR-012 compliant: advisory by default, blocking on release branches."
        )
    )
    parser.add_argument(
        "--notebooks-dir",
        default="notebooks",
        help="Root directory to discover notebooks (default: notebooks).",
    )
    parser.add_argument(
        "--output",
        default="reports/docs/notebook_execution_report.json",
        help="Destination path for the JSON report.",
    )
    parser.add_argument(
        "--cell-timeout",
        type=int,
        default=30,
        help="Per-cell execution timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--notebook-timeout",
        type=int,
        default=300,
        help="Per-notebook wall-clock timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--mode",
        choices=["advisory", "blocking"],
        default="advisory",
        help=(
            "Execution mode: 'advisory' (exit 0 always) or "
            "'blocking' (exit 1 on failure). "
            "Use 'blocking' on release/stable branches per ADR-012."
        ),
    )
    return parser


def main() -> int:
    """Entry point for command-line use."""
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_notebooks(
        notebooks_dir=Path(args.notebooks_dir),
        output_path=Path(args.output),
        cell_timeout=args.cell_timeout,
        notebook_timeout=args.notebook_timeout,
        mode=args.mode,
    )


if __name__ == "__main__":
    sys.exit(main())
