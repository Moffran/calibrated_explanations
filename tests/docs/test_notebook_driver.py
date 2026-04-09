"""Tests for scripts/docs/run_notebooks.py (ADR-012 notebook execution driver).

Why a new test file?
--------------------
The notebook execution driver (scripts/docs/run_notebooks.py) is a standalone
script with CLI contract and JSON schema obligations distinct from any existing
SUT.  No existing test file covers script-level notebook execution behavior.
Scope: unit (pure helpers) + light integration (run_notebooks with mocked
nbconvert so tests stay fast and offline).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the SUT (namespace package — no __init__.py needed)
# ---------------------------------------------------------------------------
import scripts.docs.run_notebooks as driver
from scripts.docs.run_notebooks import (
    VALID_STATUSES,
    discover_notebooks,
    extract_errors,
    read_skip_tag,
    run_notebooks,
    validate_report_schema,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def nb_dict(
    cells: list[dict[str, Any]] | None = None, ce_skip: str | None = None
) -> dict[str, Any]:
    """Return a minimal notebook dict."""
    nb: dict[str, Any] = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": cells or [],
    }
    if ce_skip is not None:
        nb["metadata"]["ce_skip"] = ce_skip
    return nb


def error_output(ename: str = "RuntimeError", evalue: str = "oops") -> dict[str, Any]:
    return {"output_type": "error", "ename": ename, "evalue": evalue, "traceback": []}


def code_cell(outputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"cell_type": "code", "source": "1+1", "outputs": outputs or [], "metadata": {}}


def write_nb(path: Path, nb: dict[str, Any]) -> None:
    path.write_text(json.dumps(nb), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixture: inject mock nbformat + nbconvert into sys.modules
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_nbconvert() -> Any:
    """Patch sys.modules with minimal fake nbformat + nbconvert stubs.

    Yields a dict with handles to the mock instances so individual tests can
    configure desired behavior (return values, side effects).
    """
    CellExecutionError = type("CellExecutionError", (Exception,), {})
    CellTimeoutError = type("CellTimeoutError", (Exception,), {})

    mock_ep_instance = MagicMock(name="ep_instance")
    mock_ep_cls = MagicMock(name="ExecutePreprocessor", return_value=mock_ep_instance)

    mock_preprocessors = MagicMock(name="nbconvert.preprocessors")
    mock_preprocessors.CellExecutionError = CellExecutionError
    mock_preprocessors.ExecutePreprocessor = mock_ep_cls

    mock_execute_submod = MagicMock(name="nbconvert.preprocessors.execute")
    mock_execute_submod.CellTimeoutError = CellTimeoutError

    mock_nbconvert = MagicMock(name="nbconvert")
    mock_nbconvert.preprocessors = mock_preprocessors

    mock_nbformat = MagicMock(name="nbformat")

    patches = {
        "nbformat": mock_nbformat,
        "nbconvert": mock_nbconvert,
        "nbconvert.preprocessors": mock_preprocessors,
        "nbconvert.preprocessors.execute": mock_execute_submod,
    }

    with patch.dict(sys.modules, patches, clear=False):
        # Reload the driver so module-cached state picks up the mocked modules.
        importlib.reload(driver)
        yield {
            "nbformat": mock_nbformat,
            "ep_cls": mock_ep_cls,
            "ep_instance": mock_ep_instance,
            "CellExecutionError": CellExecutionError,
            "CellTimeoutError": CellTimeoutError,
        }

    # Reload clean after the test so later tests start from a fresh state.
    importlib.reload(driver)


# ===========================================================================
# Pure-helper tests (no nbconvert required)
# ===========================================================================


class TestReadSkipTag:
    def test_should_return_tag_value_when_ce_skip_present(self) -> None:
        # Arrange
        meta = {"ce_skip": "noexec", "kernelspec": {}}

        # Act
        result = read_skip_tag(meta)

        # Assert
        assert result == "noexec"

    def test_should_return_none_when_ce_skip_absent(self) -> None:
        # Arrange
        meta = {"kernelspec": {}}

        # Act
        result = read_skip_tag(meta)

        # Assert
        assert result is None

    def test_should_return_slow_tag_when_set(self) -> None:
        assert read_skip_tag({"ce_skip": "slow"}) == "slow"


class TestDiscoverNotebooks:
    def test_shoulddiscover_notebooks_in_sorted_order(self, tmp_path: Path) -> None:
        # Arrange — create notebooks in non-alphabetic order
        for name in ("c.ipynb", "a.ipynb", "b.ipynb"):
            (tmp_path / name).touch()

        # Act
        result = discover_notebooks(tmp_path)

        # Assert — sorted alphabetically for determinism
        assert [p.name for p in result] == ["a.ipynb", "b.ipynb", "c.ipynb"]

    def test_shoulddiscover_notebooks_recursively(self, tmp_path: Path) -> None:
        # Arrange
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "root.ipynb").touch()
        (subdir / "nested.ipynb").touch()

        # Act
        result = discover_notebooks(tmp_path)

        # Assert — both found
        names = {p.name for p in result}
        assert names == {"root.ipynb", "nested.ipynb"}

    def test_should_return_empty_list_when_no_notebooks(self, tmp_path: Path) -> None:
        assert discover_notebooks(tmp_path) == []


class TestExtractErrors:
    def test_shouldextract_errors_from_cell_outputs(self) -> None:
        # Arrange
        nb = nb_dict(cells=[code_cell(outputs=[error_output("ValueError", "bad")])])

        # Act
        errors = extract_errors(nb)

        # Assert
        assert len(errors) == 1
        assert errors[0] == {"cell": 0, "etype": "ValueError", "evalue": "bad"}

    def test_should_return_empty_list_when_no_errors(self) -> None:
        nb = nb_dict(cells=[code_cell(outputs=[{"output_type": "stream", "text": "ok"}])])
        assert extract_errors(nb) == []

    def test_shouldextract_errors_from_multiple_cells(self) -> None:
        # Arrange
        nb = nb_dict(
            cells=[
                code_cell(outputs=[error_output("E1", "v1")]),
                code_cell(outputs=[{"output_type": "stream"}]),
                code_cell(outputs=[error_output("E2", "v2")]),
            ]
        )

        # Act
        errors = extract_errors(nb)

        # Assert
        assert len(errors) == 2
        assert errors[0]["cell"] == 0
        assert errors[1]["cell"] == 2


class TestValidateReportSchema:
    def make_valid_record(self, notebook: str = "nb.ipynb") -> dict[str, Any]:
        return {
            "notebook": notebook,
            "status": "passed",
            "elapsed_seconds": 0.1,
            "errors": [],
            "invocation_id": "test-id",
            "skip_reason": None,
            "mode": "advisory",
        }

    def test_should_return_no_errors_for_valid_report(self) -> None:
        report = {"summary": {}, "notebooks": [self.make_valid_record()]}
        assert validate_report_schema(report) == []

    def test_should_flag_missing_required_field(self) -> None:
        rec = self.make_valid_record()
        del rec["invocation_id"]
        report = {"summary": {}, "notebooks": [rec]}

        errs = validate_report_schema(report)

        assert any("invocation_id" in e for e in errs)

    def test_should_flag_unknown_status_value(self) -> None:
        rec = self.make_valid_record()
        rec["status"] = "mystery_status"
        report = {"summary": {}, "notebooks": [rec]}

        errs = validate_report_schema(report)

        assert any("mystery_status" in e for e in errs)

    def test_should_accept_all_valid_statuses(self) -> None:
        for status in VALID_STATUSES:
            rec = self.make_valid_record()
            rec["status"] = status
            report = {"summary": {}, "notebooks": [rec]}
            assert validate_report_schema(report) == [], f"False error for status '{status}'"


# ===========================================================================
# run_notebooks integration tests (mocked nbconvert)
# ===========================================================================


class TestRunNotebooksSkipTags:
    """Skip-tag handling — exercises the pre-execution path."""

    def test_should_skip_noexec_and_emit_skipped_noexec_status(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange
        nb_path = tmp_path / "test.ipynb"
        write_nb(nb_path, nb_dict(ce_skip="noexec"))
        fake_nbconvert["nbformat"].read.return_value = nb_dict(ce_skip="noexec")
        output = tmp_path / "report.json"

        # Act
        rc = run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory")

        # Assert
        assert rc == 0
        report = json.loads(output.read_text())
        assert report["notebooks"][0]["status"] == "skipped_noexec"
        assert report["notebooks"][0]["skip_reason"] == "noexec"
        # ExecutePreprocessor should NOT be called for skipped notebooks
        fake_nbconvert["ep_instance"].preprocess.assert_not_called()

    def test_should_skip_slow_and_emit_skipped_slow_status(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange
        nb_path = tmp_path / "slow.ipynb"
        write_nb(nb_path, nb_dict(ce_skip="slow"))
        fake_nbconvert["nbformat"].read.return_value = nb_dict(ce_skip="slow")
        output = tmp_path / "report.json"

        # Act
        rc = run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory")

        # Assert
        assert rc == 0
        report = json.loads(output.read_text())
        assert report["notebooks"][0]["status"] == "skipped_slow"
        assert report["notebooks"][0]["skip_reason"] == "slow"

    def test_should_fail_with_policy_violation_when_unknown_tag_in_blocking_mode(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange
        nb_path = tmp_path / "bad.ipynb"
        write_nb(nb_path, nb_dict(ce_skip="undocumented"))
        fake_nbconvert["nbformat"].read.return_value = nb_dict(ce_skip="undocumented")
        output = tmp_path / "report.json"

        # Act
        rc = run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="blocking")

        # Assert
        assert rc == 1
        report = json.loads(output.read_text())
        rec = report["notebooks"][0]
        assert rec["status"] == "failed"
        assert "PolicyViolationError" in rec["errors"][0]["etype"]
        assert "undocumented" in rec["errors"][0]["evalue"]

    def test_should_warn_and_execute_when_unknown_tag_in_advisory_mode(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange
        nb_path = tmp_path / "bad.ipynb"
        write_nb(nb_path, nb_dict(ce_skip="undocumented"))
        fake_nbconvert["nbformat"].read.return_value = nb_dict(ce_skip="undocumented")
        fake_nbconvert["ep_instance"].preprocess.return_value = (
            nb_dict(),
            {},
        )
        output = tmp_path / "report.json"

        # Act
        with pytest.warns(UserWarning, match="Unknown ce_skip tag 'undocumented'"):
            rc = run_notebooks(
                tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory"
            )

        # Assert — advisory does NOT fail; execution proceeds
        assert rc == 0
        fake_nbconvert["ep_instance"].preprocess.assert_called_once()


class TestRunNotebooksExecution:
    """Execution outcomes — success, failure, and timeout."""

    def test_should_pass_when_notebook_executes_successfully(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange
        nb_path = tmp_path / "ok.ipynb"
        write_nb(nb_path, nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        fake_nbconvert["ep_instance"].preprocess.return_value = (nb_dict(), {})
        output = tmp_path / "report.json"

        # Act
        rc = run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory")

        # Assert
        assert rc == 0
        report = json.loads(output.read_text())
        assert report["notebooks"][0]["status"] == "passed"
        assert report["summary"]["passed"] == 1

    def test_should_fail_when_cell_execution_error_raised(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange
        nb_path = tmp_path / "fail.ipynb"
        write_nb(nb_path, nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        CellExecutionError = fake_nbconvert["CellExecutionError"]
        fake_nbconvert["ep_instance"].preprocess.side_effect = CellExecutionError("cell failure")
        output = tmp_path / "report.json"

        # Act
        rc = run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory")

        # Assert — advisory → exit 0 even on failure
        assert rc == 0
        report = json.loads(output.read_text())
        assert report["notebooks"][0]["status"] == "failed"

    def test_should_return_exit_1_in_blocking_mode_when_cell_error(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange
        nb_path = tmp_path / "fail.ipynb"
        write_nb(nb_path, nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        CellExecutionError = fake_nbconvert["CellExecutionError"]
        fake_nbconvert["ep_instance"].preprocess.side_effect = CellExecutionError("oops")
        output = tmp_path / "report.json"

        # Act
        rc = run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="blocking")

        # Assert
        assert rc == 1

    def test_should_return_exit_0_in_advisory_mode_on_failure(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange
        nb_path = tmp_path / "fail.ipynb"
        write_nb(nb_path, nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        CellExecutionError = fake_nbconvert["CellExecutionError"]
        fake_nbconvert["ep_instance"].preprocess.side_effect = CellExecutionError("fail")
        output = tmp_path / "report.json"

        # Act
        rc = run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory")

        # Assert
        assert rc == 0

    def test_should_report_timed_out_when_notebook_timeout_exceeded(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange — patch _execute_with_notebook_timeout to raise TimeoutError
        nb_path = tmp_path / "slow.ipynb"
        write_nb(nb_path, nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        output = tmp_path / "report.json"

        with patch.object(
            driver,
            "_execute_with_notebook_timeout",
            side_effect=TimeoutError("Notebook exceeded 0.001s wall-clock timeout"),
        ):
            rc = run_notebooks(
                tmp_path, output, cell_timeout=30, notebook_timeout=1, mode="advisory"
            )

        # Assert
        assert rc == 0
        report = json.loads(output.read_text())
        rec = report["notebooks"][0]
        assert rec["status"] == "timed_out"
        assert rec["errors"][0]["etype"] == "TimeoutError"

    def test_should_report_timed_out_when_cell_timeout_error_raised(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange — simulate nbconvert's own CellTimeoutError
        nb_path = tmp_path / "slow.ipynb"
        write_nb(nb_path, nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        CellTimeoutError = fake_nbconvert["CellTimeoutError"]

        with patch.object(
            driver,
            "_execute_with_notebook_timeout",
            side_effect=CellTimeoutError("cell timed out"),
        ):
            output = tmp_path / "report.json"
            rc = run_notebooks(
                tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory"
            )

        # Assert
        assert rc == 0
        report = json.loads(output.read_text())
        assert report["notebooks"][0]["status"] == "timed_out"

    def test_should_fail_when_notebook_read_raises(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange — nbformat.read raises (malformed JSON)
        nb_path = tmp_path / "bad.ipynb"
        nb_path.write_text("not valid json {{{", encoding="utf-8")
        fake_nbconvert["nbformat"].read.side_effect = ValueError("malformed notebook")
        output = tmp_path / "report.json"

        # Act
        rc = run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory")

        # Assert
        assert rc == 0
        report = json.loads(output.read_text())
        rec = report["notebooks"][0]
        assert rec["status"] == "failed"
        assert rec["errors"][0]["etype"] == "ValueError"


class TestRunNotebooksReport:
    """Report schema, ordering, and summary correctness."""

    def test_should_emit_report_with_all_required_fields(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange
        for name in ("a.ipynb", "b.ipynb"):
            write_nb(tmp_path / name, nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        fake_nbconvert["ep_instance"].preprocess.return_value = (nb_dict(), {})
        output = tmp_path / "report.json"

        # Act
        run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory")

        # Assert — all required fields present, all statuses valid
        report = json.loads(output.read_text())
        errs = validate_report_schema(report)
        assert errs == [], f"Schema violations: {errs}"

    def test_should_emit_report_in_deterministic_sorted_order(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange — create notebooks in non-alphabetic order
        for name in ("c.ipynb", "a.ipynb", "b.ipynb"):
            write_nb(tmp_path / name, nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        fake_nbconvert["ep_instance"].preprocess.return_value = (nb_dict(), {})
        output = tmp_path / "report.json"

        # Act
        run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory")

        # Assert — records appear in sorted path order
        report = json.loads(output.read_text())
        names = [Path(r["notebook"]).name for r in report["notebooks"]]
        assert names == sorted(names)

    def test_should_share_invocation_id_across_all_records(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange
        for name in ("a.ipynb", "b.ipynb"):
            write_nb(tmp_path / name, nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        fake_nbconvert["ep_instance"].preprocess.return_value = (nb_dict(), {})
        output = tmp_path / "report.json"

        # Act
        run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory")

        # Assert — same invocation_id for all records
        report = json.loads(output.read_text())
        ids = {r["invocation_id"] for r in report["notebooks"]}
        assert len(ids) == 1
        assert report["summary"]["invocation_id"] == next(iter(ids))

    def test_should_create_output_parent_directory_automatically(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange — deep path that does not yet exist
        nb_path = tmp_path / "nb.ipynb"
        write_nb(nb_path, nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        fake_nbconvert["ep_instance"].preprocess.return_value = (nb_dict(), {})
        output = tmp_path / "reports" / "docs" / "notebook_execution_report.json"

        # Act
        run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory")

        # Assert
        assert output.exists()

    def test_should_record_mode_field_in_each_record(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange
        write_nb(tmp_path / "nb.ipynb", nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        fake_nbconvert["ep_instance"].preprocess.return_value = (nb_dict(), {})
        output = tmp_path / "report.json"

        # Act
        run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="blocking")

        # Assert
        report = json.loads(output.read_text())
        assert report["notebooks"][0]["mode"] == "blocking"
        assert report["summary"]["mode"] == "blocking"


class TestRunNotebooksModeSemantics:
    """Advisory vs blocking mode exit-code behaviour (ADR-012 branch-gate contract)."""

    def test_should_use_advisory_mode_by_default(self) -> None:
        """Default arg parser mode must be 'advisory' (ADR-012)."""
        # Import the parser builder
        from scripts.docs.run_notebooks import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.mode == "advisory"

    def test_should_default_cell_timeout_to_30_seconds(self) -> None:
        from scripts.docs.run_notebooks import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.cell_timeout == 30

    def test_should_default_notebook_timeout_to_300_seconds(self) -> None:
        from scripts.docs.run_notebooks import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.notebook_timeout == 300

    def test_should_return_exit_1_in_blocking_mode_when_timed_out(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        """Blocking mode must exit 1 for timed_out, not just failed."""
        nb_path = tmp_path / "slow.ipynb"
        write_nb(nb_path, nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        output = tmp_path / "report.json"

        with patch.object(
            driver,
            "_execute_with_notebook_timeout",
            side_effect=TimeoutError("Notebook exceeded 1s wall-clock timeout"),
        ):
            rc = run_notebooks(
                tmp_path, output, cell_timeout=30, notebook_timeout=1, mode="blocking"
            )

        assert rc == 1
        report = json.loads(output.read_text())
        assert report["notebooks"][0]["status"] == "timed_out"


class TestDiscoverNotebooksCheckpoints:
    """Checkpoint files must be excluded from discovery."""

    def test_should_exclude_ipynb_checkpoints_directory(self, tmp_path: Path) -> None:
        # Arrange — a real notebook and a checkpoint copy
        (tmp_path / "real.ipynb").touch()
        checkpoints_dir = tmp_path / ".ipynb_checkpoints"
        checkpoints_dir.mkdir()
        (checkpoints_dir / "real-checkpoint.ipynb").touch()

        # Act
        result = discover_notebooks(tmp_path)

        # Assert — checkpoint is excluded
        names = [p.name for p in result]
        assert names == ["real.ipynb"]
        assert not any(".ipynb_checkpoints" in str(p) for p in result)


class TestRunNotebooksExtractErrors:
    """extract_errors must be called on successful execution output."""

    def test_should_fail_when_executed_notebook_contains_error_output(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        # Arrange — execution "succeeds" (no exception) but the returned
        # notebook node contains an error output in one cell.
        nb_with_error = nb_dict(cells=[code_cell(outputs=[error_output("RuntimeError", "boom")])])
        write_nb(tmp_path / "nb.ipynb", nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        # preprocess returns a notebook that has an error in cell 0
        fake_nbconvert["ep_instance"].preprocess.return_value = (nb_with_error, {})
        output = tmp_path / "report.json"

        rc = run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory")

        report = json.loads(output.read_text())
        rec = report["notebooks"][0]
        assert rec["status"] == "failed"
        assert rec["errors"][0]["etype"] == "RuntimeError"
        assert rec["errors"][0]["evalue"] == "boom"
        # advisory mode still exits 0 even on cell errors
        assert rc == 0

    def test_should_exit_1_in_blocking_mode_when_cell_error_output_present(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        nb_with_error = nb_dict(cells=[code_cell(outputs=[error_output("ValueError", "x")])])
        write_nb(tmp_path / "nb.ipynb", nb_dict())
        fake_nbconvert["nbformat"].read.return_value = nb_dict()
        fake_nbconvert["ep_instance"].preprocess.return_value = (nb_with_error, {})
        output = tmp_path / "report.json"

        rc = run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="blocking")

        assert rc == 1


class TestRunNotebooksZeroDir:
    """Zero-notebook directory behavior."""

    def test_should_warn_in_advisory_mode_when_no_notebooks_found(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        output = tmp_path / "report.json"

        with pytest.warns(UserWarning, match="No notebooks found"):
            rc = run_notebooks(
                tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="advisory"
            )

        assert rc == 0

    def test_should_return_exit_1_in_blocking_mode_when_no_notebooks_found(
        self, tmp_path: Path, fake_nbconvert: dict[str, Any]
    ) -> None:
        output = tmp_path / "report.json"

        rc = run_notebooks(tmp_path, output, cell_timeout=30, notebook_timeout=300, mode="blocking")

        assert rc == 1
