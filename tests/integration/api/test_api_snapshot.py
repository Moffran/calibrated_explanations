from importlib import import_module
from pathlib import Path

SNAPSHOT = Path(__file__).parent / "data" / "api_snapshot.txt"


def exported_symbols(mod):
    return sorted(getattr(mod, "__all__", []))


def test_public_api_snapshot():
    root = import_module("calibrated_explanations")
    core_pkg = import_module("calibrated_explanations.core.__init__")
    snapshot_current = [
        "# root __all__",
        *exported_symbols(root),
        "# core __all__",
        *exported_symbols(core_pkg),
    ]
    if not SNAPSHOT.exists():
        SNAPSHOT.write_text("\n".join(snapshot_current))
        raise AssertionError("API snapshot file created; commit and re-run tests.")
    snapshot_saved = SNAPSHOT.read_text().splitlines()
    assert (
        snapshot_saved == snapshot_current
    ), "Public API symbols changed unexpectedly. Update snapshot intentionally if desired."
