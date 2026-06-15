#!/usr/bin/env python3
"""
SessionStart context injection for calibrated_explanations (OSS).
Injects: current CE version, release stage, contributor guidance pointer.
Non-blocking — exits 0 regardless.
"""
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
CHANGELOG = _REPO_ROOT / "CHANGELOG.md"
PYPROJECT = _REPO_ROOT / "pyproject.toml"
CONTRIBUTING = _REPO_ROOT / "CONTRIBUTING.md"


def _read_version() -> str:
    try:
        text = PYPROJECT.read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


def _read_release_stage() -> str:
    """Return the top heading from CHANGELOG (current dev state)."""
    try:
        lines = CHANGELOG.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.startswith("## "):
                return line.lstrip("#").strip()
    except Exception:
        pass
    return "unknown"


def main() -> int:
    version = _read_version()
    stage = _read_release_stage()
    has_contributing = CONTRIBUTING.exists()

    parts = [
        f"📦 calibrated_explanations v{version}",
        f"📋 Latest changelog entry: {stage}",
    ]
    if has_contributing:
        parts.append("📖 See CONTRIBUTING.md for contributor workflow and coding standards.")

    output = {"systemMessage": "\n".join(parts)}
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
