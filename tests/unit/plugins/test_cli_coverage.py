"""Unit tests for coverage of cli.py."""

from types import SimpleNamespace

from unittest.mock import Mock, patch

from calibrated_explanations.plugins.cli import (
    coerce_string_tuple,
    emit_header,
    format_common_metadata,
    cmd_list,
    emit_explanation_descriptor,
    emit_interval_descriptor,
    emit_plot_descriptor,
    emit_plot_builder_descriptor,
    emit_plot_renderer_descriptor,
    emit_discovery_report,
    cmd_validate_plot,
    cmd_validate_interval,
    cmd_set_default,
    cmd_show,
    cmd_report,
)
from calibrated_explanations.plugins.intervals import IntervalCalibratorPlugin


class TestCliCoverage:
    def test_coerce_string_tuple(self):
        """Test coercion to tuple of strings."""
        assert coerce_string_tuple(None) == ()
        assert coerce_string_tuple("foo") == ("foo",)
        assert coerce_string_tuple(["a", "b"]) == ("a", "b")
        assert coerce_string_tuple(("c", None)) == ("c",)
        assert coerce_string_tuple([]) == ()
        assert coerce_string_tuple([" a ", ""]) == ("a",)  # stripped

    def test_emit_header(self, capsys):
        """Test simple header emission."""
        emit_header("Test Header")
        captured = capsys.readouterr()
        assert "Test Header" in captured.out
        assert "===========" in captured.out

    def test_format_common_metadata(self):
        """Test metadata formatting."""
        meta = {"name": "Test Plugin", "schema_version": "1.0.0"}
        assert "name=Test Plugin" in format_common_metadata(meta)
        assert "schema_version=1.0.0" in format_common_metadata(meta)

        meta_empty = {}
        assert "name=<unnamed>" in format_common_metadata(meta_empty)

    @patch("calibrated_explanations.plugins.cli.list_explanation_descriptors")
    def test_cmd_list_explanations(self, mock_list, capsys):
        """Test listing explanations."""
        args = Mock()
        args.kind = "explanations"
        args.trusted_only = False
        args.verbose = False
        args.plots = False
        args.include_skipped = False

        # Mock descriptor
        desc = Mock()
        desc.identifier = "test.explainer"
        desc.metadata = {
            "name": "Test Explainer",
            "modes": ["classification"],
            "tasks": ["binary"],
            "fallbacks": [],
        }
        desc.trusted = True

        mock_list.return_value = [desc]

        exit_code = cmd_list(args)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Explanation plugins" in captured.out
        assert "test.explainer" in captured.out

    @patch("calibrated_explanations.plugins.cli.list_explanation_descriptors")
    def test_cmd_list_empty(self, mock_list, capsys):
        """Test listing empty results."""
        args = Mock()
        args.kind = "explanations"
        args.trusted_only = False
        args.verbose = False
        args.plots = False
        args.include_skipped = False

        mock_list.return_value = []

        exit_code = cmd_list(args)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "<none>" in captured.out

    @patch("calibrated_explanations.plugins.cli.list_plot_renderer_descriptors")
    def test_cmd_list_plot_renderers_empty(self, mock_list, capsys):
        """Test listing empty plot renderers."""
        args = Mock()
        args.kind = "plot-renderers"
        args.trusted_only = False
        args.verbose = False
        args.plots = False
        args.include_skipped = False

        mock_list.return_value = []

        exit_code = cmd_list(args)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "<none>" in captured.out

    @patch("calibrated_explanations.plugins.registry.find_plot_builder")
    def test_cmd_validate_plot(self, mock_find, capsys):
        """Test validate plot command."""
        args = Mock()
        args.builder = "test.builder"

        # Case 1: builder not found
        mock_find.return_value = None
        exit_code = cmd_validate_plot(args)
        assert exit_code == 1

        # Case 2: build fail
        builder = Mock()
        builder.build.side_effect = Exception("Build error")
        mock_find.return_value = builder
        exit_code = cmd_validate_plot(args)
        assert exit_code == 2

        # Case 3: success
        builder.build.side_effect = None
        builder.build.return_value = {"version": 1, "data": []}  # Mock valid plotspec

        with patch("calibrated_explanations.viz.serializers.validate_plotspec") as mock_val:
            mock_val.return_value = None  # success
            exit_code = cmd_validate_plot(args)
            assert exit_code == 0  # Should count as success if no exception raised

        # Case 4: invalid plotspec
        builder.build.return_value = {"plot_spec": "invalid"}
        with patch("calibrated_explanations.viz.serializers.validate_plotspec") as mock_val:
            mock_val.side_effect = Exception("Invalid plotspec")
            exit_code = cmd_validate_plot(args)
            assert exit_code == 3

    def test_emit_descriptor_helpers_and_discovery(self, capsys):
        desc = SimpleNamespace(
            identifier="test.explainer",
            metadata={
                "name": "Test",
                "schema_version": "1.0",
                "modes": ["classification"],
                "tasks": ["binary"],
                "interval_dependency": ["interval.dep"],
                "plot_dependency": ["plot.dep"],
                "fallbacks": ["fallback.explainer"],
            },
            trusted=True,
        )
        interval_desc = SimpleNamespace(
            identifier="interval",
            metadata={
                "name": "Interval",
                "schema_version": "1.0",
                "modes": ["classification"],
                "dependencies": ["dep"],
            },
            trusted=False,
        )
        plot_desc = SimpleNamespace(
            identifier="plot.style",
            metadata={
                "builder_id": "builder",
                "renderer_id": "renderer",
                "fallbacks": ["fb"],
                "is_default": True,
                "legacy_compatible": False,
                "default_for": ["style.x"],
            },
            trusted=True,
        )
        builder_desc = SimpleNamespace(
            identifier="builder",
            metadata={
                "style": "style",
                "capabilities": ["cap"],
                "output_formats": ["png"],
                "dependencies": ["dep"],
                "legacy_compatible": True,
            },
            trusted=False,
        )
        renderer_desc = SimpleNamespace(
            identifier="renderer",
            metadata={
                "capabilities": ["cap"],
                "output_formats": ["png"],
                "dependencies": ["dep"],
                "supports_interactive": True,
            },
            trusted=True,
        )

        with patch("calibrated_explanations.plugins.cli.is_identifier_denied", return_value=True):
            emit_explanation_descriptor(desc)
        emit_interval_descriptor(interval_desc)
        emit_plot_descriptor(plot_desc)
        emit_plot_builder_descriptor(builder_desc)
        emit_plot_renderer_descriptor(renderer_desc)

        record = SimpleNamespace(
            identifier="skipped.plugin",
            metadata={"name": "Skipped"},
            provider="provider",
            source="source",
            details={"reason": "denied"},
        )
        report = SimpleNamespace(
            skipped_denied=[record],
            skipped_untrusted=[record],
            checksum_failures=[record],
        )
        emit_discovery_report(report)
        emit_discovery_report(None)

        captured = capsys.readouterr()
        assert "denied via CE_DENY_PLUGIN" in captured.out
        assert "Discovery skipped: denied" in captured.out
        assert "Provider" not in captured.out  # detail lines have provider label in lower-case

    def test_cmd_list_all_verbose_and_skipped(self):
        args = Mock(kind="all", trusted_only=False, verbose=True, plots=False, include_skipped=True)
        descriptor = SimpleNamespace(
            identifier="test.explainer",
            metadata={
                "name": "Test",
                "schema_version": "1.0",
                "modes": ["classification"],
                "tasks": ["binary"],
                "interval_dependency": [],
                "plot_dependency": [],
                "fallbacks": [],
            },
            trusted=True,
        )
        interval_desc = SimpleNamespace(
            identifier="interval",
            metadata={
                "name": "Interval",
                "schema_version": "1.0",
                "modes": ["regression"],
                "dependencies": [],
            },
            trusted=False,
        )
        builder_desc = SimpleNamespace(
            identifier="builder",
            metadata={
                "style": "s",
                "capabilities": ["cap"],
                "output_formats": ["png"],
                "dependencies": [],
            },
            trusted=True,
        )
        renderer_desc = SimpleNamespace(
            identifier="renderer",
            metadata={
                "capabilities": [],
                "output_formats": ["png"],
                "dependencies": [],
                "supports_interactive": False,
            },
            trusted=False,
        )
        plot_desc = SimpleNamespace(
            identifier="style",
            metadata={"builder_id": "builder", "renderer_id": "renderer", "fallbacks": []},
            trusted=True,
        )
        report = SimpleNamespace(
            skipped_denied=[
                SimpleNamespace(
                    identifier="denied", metadata={}, provider="prov", source="src", details={}
                )
            ],
            skipped_untrusted=[],
            checksum_failures=[],
        )

        with (
            patch("calibrated_explanations.plugins.cli.load_entrypoint_plugins") as mock_load,
            patch(
                "calibrated_explanations.plugins.cli.list_explanation_descriptors",
                return_value=[descriptor],
            ),
            patch(
                "calibrated_explanations.plugins.cli.list_interval_descriptors",
                return_value=[interval_desc],
            ),
            patch(
                "calibrated_explanations.plugins.cli.list_plot_builder_descriptors",
                return_value=[builder_desc],
            ),
            patch(
                "calibrated_explanations.plugins.cli.list_plot_renderer_descriptors",
                return_value=[renderer_desc],
            ),
            patch(
                "calibrated_explanations.plugins.cli.list_plot_style_descriptors",
                return_value=[plot_desc],
            ),
            patch(
                "calibrated_explanations.plugins.cli.get_last_discovery_report", return_value=report
            ),
            patch("calibrated_explanations.plugins.cli.is_identifier_denied", return_value=False),
        ):
            exit_code = cmd_list(args)

        assert exit_code == 0
        assert mock_load.called

    def test_cmd_validate_interval_success(self, capsys):
        class DummyIntervalPlugin(IntervalCalibratorPlugin):
            def create(self, *args, **kwargs):
                return "ok"

        descriptor = SimpleNamespace(
            plugin=DummyIntervalPlugin(),
            metadata={"fast_compatible": True, "dependencies": ["dep"]},
        )
        with (
            patch("calibrated_explanations.plugins.cli.load_entrypoint_plugins"),
            patch(
                "calibrated_explanations.plugins.cli.find_interval_descriptor",
                return_value=descriptor,
            ),
        ):
            exit_code = cmd_validate_interval(Mock(plugin="interval"))
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "validated successfully" in captured.out

    def test_cmd_set_default_updates_styles(self, capsys):
        args = Mock(style="style.id")
        style_desc = SimpleNamespace(identifier="style.id", metadata={"legacy_compatible": True})

        descriptors = [
            style_desc,
            SimpleNamespace(identifier="other", metadata={}),
        ]
        calls = []

        def fake_register(identifier, metadata):
            calls.append((identifier, metadata))

        with (
            patch(
                "calibrated_explanations.plugins.registry.find_plot_style_descriptor",
                return_value=style_desc,
            ),
            patch(
                "calibrated_explanations.plugins.registry.list_plot_style_descriptors",
                return_value=descriptors,
            ),
            patch(
                "calibrated_explanations.plugins.cli.register_plot_style", side_effect=fake_register
            ),
        ):
            exit_code = cmd_set_default(args)

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Set 'style.id' as default plot style" in captured.out
        assert any(metadata.get("is_default") for _, metadata in calls)

    def test_cmd_show_missing_and_found(self, capsys):
        args = Mock(kind="explanations", identifier="missing")
        with patch(
            "calibrated_explanations.plugins.cli.find_explanation_descriptor", return_value=None
        ):
            exit_code = cmd_show(args)
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "Explanation plugin 'missing' is not registered" in captured.out

        args_found = Mock(kind="plots", identifier="style.id")
        descriptor = SimpleNamespace(
            identifier="style.id",
            metadata={"name": "Style"},
            trusted=True,
            source="repo",
        )

        with patch(
            "calibrated_explanations.plugins.cli.find_plot_style_descriptor",
            return_value=descriptor,
        ):
            exit_code = cmd_show(args_found)
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Identifier : style.id" in captured.out
        assert "Trusted    : yes" in captured.out
        assert "Source     : repo" in captured.out
        assert "Metadata   :" in captured.out

    @patch("calibrated_explanations.plugins.cli.load_entrypoint_plugins")
    @patch("calibrated_explanations.plugins.cli.get_discovery_report")
    def test_cmd_report(self, mock_get_report, mock_load, capsys):
        """Test report command triggers discovery and emits report."""
        report = SimpleNamespace(
            skipped_denied=[],
            skipped_untrusted=[],
            checksum_failures=[],
            accepted=[],
        )
        mock_get_report.return_value = report

        exit_code = cmd_report(Mock())

        assert exit_code == 0
        mock_load.assert_called_once_with(include_untrusted=True)
        mock_get_report.assert_called_once()
        # emit_discovery_report is called, but since report is empty, no specific assertions
