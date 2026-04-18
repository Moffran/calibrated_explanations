"""Root CLI entry point for calibrated_explanations."""

from __future__ import annotations

import argparse
import pprint
from collections.abc import Sequence

from .core.config_manager import ConfigManager
from .plugins.cli import main as plugins_main


def _cmd_config_show(_args: argparse.Namespace) -> int:
    """Display the current effective configuration snapshot."""
    snapshot = ConfigManager.from_sources().export_effective()
    print(f"profile_id={snapshot.profile_id}")
    print(f"schema_version={snapshot.schema_version}")
    print("effective_values:")
    print(pprint.pformat(dict(snapshot.values), sort_dicts=True))
    return 0


def _cmd_config_export(_args: argparse.Namespace) -> int:
    """Export the current effective configuration snapshot as a dict payload."""
    snapshot = ConfigManager.from_sources().export_effective()
    payload = {
        "profile_id": snapshot.profile_id,
        "schema_version": snapshot.schema_version,
        "values": dict(snapshot.values),
        "sources": dict(snapshot.sources),
    }
    print(pprint.pformat(payload, sort_dicts=True))
    return 0


def _cmd_plugins(args: argparse.Namespace) -> int:
    """Delegate plugin-related commands to ``ce.plugins`` CLI."""
    return int(plugins_main(args.plugin_args))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the root CLI entry point and return the exit code."""
    parser = argparse.ArgumentParser(
        prog="ce",
        description="Calibrated Explanations CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    config_parser = subparsers.add_parser(
        "config",
        help="Inspect effective runtime configuration",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    config_show_parser = config_subparsers.add_parser(
        "show",
        help="Show effective configuration snapshot used by this CLI invocation",
    )
    config_show_parser.set_defaults(func=_cmd_config_show)

    config_export_parser = config_subparsers.add_parser(
        "export",
        help="Export effective configuration snapshot used by this CLI invocation",
    )
    config_export_parser.set_defaults(func=_cmd_config_export)

    plugins_parser = subparsers.add_parser(
        "plugins",
        help="Manage plugins (delegates to ce.plugins)",
    )
    plugins_parser.add_argument("plugin_args", nargs=argparse.REMAINDER)
    plugins_parser.set_defaults(func=_cmd_plugins)

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if not getattr(exc, "code", None) or exc.code != 0:
            parser.print_help()
        return exc.code

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    if args.command == "config" and not getattr(args, "config_command", None):
        config_parser.print_help()
        return 0

    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import sys

    sys.exit(main())
