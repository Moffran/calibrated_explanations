# /refresh-ce-context

Update the agent instruction files so that they reflect the **current state** of
the `calibrated_explanations` library. Run this prompt whenever:
- The public API changes (new methods, removed parameters, renamed symbols).
- A new ADR is accepted or an existing one is closed/superseded.
- User feedback identified a gap in any agent's CE knowledge.

## What this prompt does

Copilot will inspect the codebase and produce **diff-ready edits** to the files
below. It will NOT generate new files unless asked.

### Canonical file (update first)

| File | What to update |
|---|---|
| `AGENT_INSTRUCTIONS.md` | §1 CE-first policy, §2 architecture, §3 coding standards, §5 key files |

### Platform-specific files (propagate changes after updating canonical)

| File | What to update |
|---|---|
| `.github/copilot-instructions.md` | Sections §1–§8 that mirror canonical; §9–§11 Copilot-specific only |
| `AGENTS.md` | §5 CE-first utilities if helper signatures changed |
| `CLAUDE.md` | §2–§4 only if Claude Code tool rules changed |
| `GEMINI.md` | §3–§5 only if Gemini tool rules changed |
| `.ai/tool_description.yaml` | `public_api`, `basic_workflow`, `common_parameters` |
| `PROMPTS.md` | Agent checklist, canonical usage patterns |

## Steps Copilot must follow

1. **Read current API** – scan `src/calibrated_explanations/core/__init__.py` and
   `src/calibrated_explanations/__init__.py` for exported names and their signatures.
2. **Diff against `AGENT_INSTRUCTIONS.md`** – identify stale method names, removed
   parameters, or missing new symbols.
3. **Read active ADRs** – check `docs/improvement/adrs/` for any ADR whose status
   changed since the last context update.
4. **Update `AGENT_INSTRUCTIONS.md` first** – minimal diff only; do not rewrite
   unchanged sections.
5. **Propagate to platform files** – apply the same minimal diff to the relevant
   sections of each platform file.
6. **Incorporate user feedback** – if the user supplied feedback text (see `feedback=`
   below), create a new bullet in the relevant section of `AGENT_INSTRUCTIONS.md`
   and append a dated entry to `.github/copilot-feedback-log.md` using:
   - `**Feedback:**`
   - `**Root cause:**`
   - `**Durable fix:**`
   - `**Verification:**`
   - `**Status:**`

## Inputs (optional)

- `feedback="<free-text describing what an agent got wrong or missed>"`
- `adr=<ADR-NNN>` (limit the refresh to one ADR)
- `module=<dotted.module.path>` (limit the refresh to one module)

## Checklist on completion

- [ ] All exported symbols in `__init__.py` are reflected in `AGENT_INSTRUCTIONS.md`.
- [ ] No stale method names remain in any agent instruction file.
- [ ] New ADR status changes are noted in `AGENT_INSTRUCTIONS.md`.
- [ ] All platform-specific files reference the updated canonical correctly.
- [ ] User feedback entry added to `.github/copilot-feedback-log.md` (if `feedback=` was provided).
