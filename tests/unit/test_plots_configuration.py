import pytest

from calibrated_explanations.viz import plots as plotting


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Ensure environment variables are cleaned between tests."""
    monkeypatch.delenv("CE_PLOT_STYLE", raising=False)
    monkeypatch.delenv("CE_PLOT_STYLE_FALLBACKS", raising=False)


@pytest.fixture
def pyproject_factory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _create(content=None):
        if content is not None:
            (tmp_path / "pyproject.toml").write_text(content, encoding="utf-8")

    return _create


def test_read_plot_pyproject_missing_file(pyproject_factory):
    pyproject_factory(None)
    assert plotting._read_plot_pyproject() == {}


def test_read_plot_pyproject_valid_settings(pyproject_factory):
    pyproject_factory("""
[tool.calibrated_explanations.plots]
style = "py-style"
fallbacks = ["py-fallback", "legacy"]
""")
    config = plotting._read_plot_pyproject()
    assert config["style"] == "py-style"
    assert config["fallbacks"] == ["py-fallback", "legacy"]


def test_read_plot_pyproject_partial_sections(pyproject_factory):
    pyproject_factory("""
[tool.calibrated_explanations]
other = "value"
""")
    assert plotting._read_plot_pyproject() == {}


def test_read_plot_pyproject_empty_plots_section(pyproject_factory):
    pyproject_factory("""
[tool.calibrated_explanations.plots]
""")
    assert plotting._read_plot_pyproject() == {}
