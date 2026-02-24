# GEMINI.md — Google Gemini (Antigravity) Instructions for `calibrated_explanations`

> **Canonical agent instructions:** `CONTRIBUTOR_INSTRUCTIONS.md` — the single source of
> truth shared by all agent platforms (Copilot, Codex, Claude Code, Gemini). Read
> that first. This file adds **only** Google Gemini-specific context.

---

## 1. Session priming (Gemini-specific)

At the start of every Gemini session, explicitly direct the agent to:

1. Read `CONTRIBUTOR_INSTRUCTIONS.md` (canonical CE rules).
2. Read `GEMINI.md` (this file).
3. Follow the CE-first checklist from `docs/get-started/ce_first_agent_guide.md`.
4. Use helpers from `src/calibrated_explanations/ce_agent_utils.py`.

Reusable priming prompt:

```text
You are a CE-first agent for calibrated_explanations. Read CONTRIBUTOR_INSTRUCTIONS.md
and GEMINI.md first. Use WrapCalibratedExplainer and ce_agent_utils helpers.
Fail fast if CE-first invariants are not satisfied.
```

---

## 2. Context and memory (Gemini-specific)

Gemini has a large context window. Use it to include the full source of relevant
modules when answering questions, rather than working from partial snippets.

Gemini does not persist memory across unrelated sessions. Use the same feedback
persistence pattern as all agents (see `CONTRIBUTOR_INSTRUCTIONS.md` §12). For
Gemini-specific quirks, add a bullet to this file and commit it.

---

## 3. Tool use guidance (Gemini-specific)

- When running code or shell commands, prefer the exact commands from
  `CONTRIBUTOR_INSTRUCTIONS.md` §7.
- Do not run commands that modify files outside the repository root.
- Use `make test` to validate changes; do not invent alternative test commands.
- Never use the Python heredoc construct (see `CONTRIBUTOR_INSTRUCTIONS.md` §11).

---

## 4. Workspace sync (Gemini-specific)

After any CE library update:

```bash
pip install -e .[dev]
pytest -q
```

Then ask Gemini to re-read `CONTRIBUTOR_INSTRUCTIONS.md` and update any examples that
reference stale API signatures.

---

## 5. Feedback loop (Gemini-specific)

| Type of feedback | Where to record it |
|---|---|
| Wrong API usage | Update `CONTRIBUTOR_INSTRUCTIONS.md` §1 or §2 |
| Platform-specific Gemini quirk | Add a bullet to this file |
| Recurring test mistake | Add a test to `tests/unit/test_ce_agent_utils.py` |

All feedback entries must be recorded in `.github/copilot-feedback-log.md`
with `Feedback`, `Root cause`, `Durable fix`, `Verification`, and `Status`.
