import pytest

from calibrated_explanations.viz import plots as plotting


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure environment variables are cleaned between tests."""
    monkeypatch.delenv("CE_PLOT_STYLE", raising=False)
    monkeypatch.delenv("CE_PLOT_STYLE_FALLBACKS", raising=False)


def test_read_plot_pyproject_extracts_nested_settings(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.calibrated_explanations.plots]
style = "py-style"
fallbacks = ["py-fallback", "legacy"]
"""
    )

    monkeypatch.chdir(tmp_path)

    config = plotting._read_plot_pyproject()

    assert config["style"] == "py-style"
    assert config["fallbacks"] == ["py-fallback", "legacy"]


def test_read_plot_pyproject_missing_file_returns_empty_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert plotting._read_plot_pyproject() == {}


def test_split_csv_handles_strings_sequences_and_invalid_values():
    assert plotting._split_csv(" a , b , , c ") == ("a", "b", "c")
    assert plotting._split_csv(["first", "", "second", 3]) == ("first", "second")
    assert plotting._split_csv(None) == ()
    assert plotting._split_csv(False) == ()


def test_require_matplotlib_raises_helpful_error(monkeypatch):
    from calibrated_explanations.core.exceptions import ConfigurationError
    monkeypatch.setattr(plotting, "plt", None)
    monkeypatch.setattr(plotting, "mcolors", None)
    monkeypatch.setattr(plotting, "_MATPLOTLIB_IMPORT_ERROR", RuntimeError("missing"))

    with pytest.raises(ConfigurationError) as excinfo:
        plotting.__require_matplotlib()

    message = str(excinfo.value)
    assert "Plotting requires matplotlib" in message
    assert "missing" in message
