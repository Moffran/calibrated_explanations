# CLAUDE.md — Claude Code Instructions for `calibrated_explanations`

> **Canonical agent instructions:** `CONTRIBUTOR_INSTRUCTIONS.md` — the single source of
> truth shared by all agent platforms (Copilot, Codex, Claude Code, Gemini). Read
> that first. This file adds **only** Claude Code-specific context.

---

## 1. Session priming (Claude-specific)

At the start of every Claude Code session, explicitly direct the agent to:

1. Read `CONTRIBUTOR_INSTRUCTIONS.md` (canonical CE rules).
2. Read `CLAUDE.md` (this file).
3. Follow `docs/get-started/ce_first_agent_guide.md`.
4. Use `src/calibrated_explanations/ce_agent_utils.py` helpers instead of ad-hoc wrappers.

Reusable priming prompt:

```text
You are a CE-first agent for calibrated_explanations. Read CONTRIBUTOR_INSTRUCTIONS.md
and CLAUDE.md first. Use WrapCalibratedExplainer and ce_agent_utils helpers.
Fail fast if CE-first invariants are not satisfied.
```

---

## 2. Claude tool permissions (Claude-specific)

- Primary permission policy lives in `.claude/settings.json`.
- Local overrides live in `.claude/settings.local.json`.
- Keep permission scopes minimal and repository-local by default.
- Do not bypass permission constraints by using alternative toolchains.

---

## 3. Workflow and validation (Claude-specific)

- Prefer `make local-checks-pr` as the default fast validation path.
- Escalate to `make local-checks` when changes touch main-branch gates.
- After dependency or API changes, run:
  - `pip install -e .[dev]`
  - `python -m pip check`
  - `pytest -q`

---

## 4. Feedback loop (Claude-specific)

Claude sessions do not persist memory across unrelated runs. Persist feedback via:

- Canonical updates in `CONTRIBUTOR_INSTRUCTIONS.md` for shared guidance.
- Claude-only quirks in `CLAUDE.md`.
- Cross-agent feedback entries in `.github/copilot-feedback-log.md` using:
  - `Feedback`
  - `Root cause`
  - `Durable fix`
  - `Verification`
  - `Status`
