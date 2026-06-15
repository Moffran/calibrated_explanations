#!/usr/bin/env python3
"""
PostToolUse auto-formatter hook.
After a file-edit tool call on a .py file, runs ruff format on it.
Non-blocking — exits 0 regardless. Reports result via systemMessage.
"""
import json
import subprocess
import sys
from pathlib import Path

FILE_EDIT_TOOL_NAMES = {
    "edit_file",
    "write_file",
    "create_file",
    "str_replace_editor",
    "str_replace_based_edit_tool",
    "EditFile",
    "WriteFile",
}


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        data = json.loads(raw)
    except Exception:
        return 0

    tool_name = data.get("toolName", data.get("tool", ""))
    if tool_name not in FILE_EDIT_TOOL_NAMES:
        return 0

    tool_input = data.get("toolInput", data.get("input", {}))
    file_path = ""
    if isinstance(tool_input, dict):
        file_path = tool_input.get("path", tool_input.get("file_path", tool_input.get("target_file", "")))
    elif isinstance(tool_input, str):
        file_path = tool_input

    if not file_path or not file_path.endswith(".py"):
        return 0

    p = Path(file_path)
    if not p.exists():
        return 0

    try:
        result = subprocess.run(
            ["ruff", "format", str(p)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            message = f"[autoformat] ruff format applied to {p.name}"
        else:
            message = f"[autoformat] ruff format failed for {p.name}: {result.stderr.strip()}"
    except FileNotFoundError:
        return 0
    except subprocess.TimeoutExpired:
        return 0
    except Exception as e:
        message = f"[autoformat] unexpected error: {e}"

    output = {"systemMessage": message}
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
