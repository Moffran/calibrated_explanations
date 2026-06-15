#!/usr/bin/env python3
"""
PreToolUse safety guard hook.
Reads tool call info from stdin, blocks dangerous shell commands.
Exit 0 = allow, exit 2 = block.
JSON stdout with permissionDecision = "deny" also blocks the call.
"""
import json
import sys
import re

# Patterns that should be blocked in shell/terminal commands.
BLOCKED_PATTERNS = [
    (r"git\s+push\s+--force", "git push --force is not allowed. Use --force-with-lease if you are certain, and do it manually."),
    (r"git\s+push\s+-f\b", "git push -f is not allowed."),
    (r"git\s+reset\s+--hard", "git reset --hard is not allowed. Stage and stash changes instead."),
    (r"git\s+clean\s+(-f|-fd|-fx|-n)", "git clean with -f is not allowed in agent sessions."),
    (r"rm\s+-rf?\s*/", "rm -rf on root paths is not allowed."),
    (r"rm\s+-rf\s+\.", "rm -rf on current directory is not allowed."),
    (r"\bsudo\s+rm\b", "sudo rm is not allowed."),
    (r"\bsudo\s+chmod\b", "sudo chmod is not allowed."),
    (r"pip\s+install\b(?!.*--dry-run).*--index-url", "pip install from custom index requires manual review."),
    (r"twine\s+upload", "Package publishing (twine upload) requires manual execution."),
    (r"npm\s+publish", "npm publish requires manual execution."),
    (r"gh\s+release\s+create", "GitHub release creation requires manual confirmation."),
]

TERMINAL_TOOL_NAMES = {
    "run_terminal_command",
    "execute_command",
    "bash",
    "shell",
    "terminal",
    "Bash",
    "Shell",
}


def deny(reason: str) -> None:
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"[safety-guard] {reason}",
        }
    }
    print(json.dumps(output))
    sys.exit(0)


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        data = json.loads(raw)
    except Exception:
        return 0

    tool_name = data.get("toolName", data.get("tool", ""))
    if tool_name not in TERMINAL_TOOL_NAMES:
        return 0

    tool_input = data.get("toolInput", data.get("input", {}))
    command = ""
    if isinstance(tool_input, dict):
        command = tool_input.get("command", tool_input.get("cmd", ""))
    elif isinstance(tool_input, str):
        command = tool_input

    if not command:
        return 0

    for pattern, reason in BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            deny(reason)

    return 0


if __name__ == "__main__":
    sys.exit(main())
