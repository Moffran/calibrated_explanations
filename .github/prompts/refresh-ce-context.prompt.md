# /refresh-ce-context

Update the Copilot instruction files so that they reflect the **current state** of
the `calibrated_explanations` library. Run this prompt whenever:
- The public API changes (new methods, removed parameters, renamed symbols).
- A new ADR is accepted or an existing one is closed/superseded.
- User feedback identified a gap in Copilot's CE knowledge.

## What this prompt does

Copilot will inspect the codebase and produce **diff-ready edits** to one or more
of the instruction files below. It will NOT generate new files unless asked.

| File | What to update |
|---|---|
| `.github/copilot-instructions.md` | §1 architecture, §5 key files, any new policy section |
| `.github/instructions/source-code.instructions.md` | API signatures, module layout changes |
| `.ai/tool_description.yaml` | `public_api`, `basic_workflow`, `common_parameters` |
| `PROMPTS.md` | Agent checklist, canonical usage patterns |

## Steps Copilot must follow

1. **Read current API** – scan `src/calibrated_explanations/core/__init__.py` and
   `src/calibrated_explanations/__init__.py` for exported names and their signatures.
2. **Diff against instruction files** – identify stale method names, removed parameters,
   or missing new symbols.
3. **Read active ADRs** – check `docs/improvement/adrs/` for any ADR whose status changed
   since the last context update.
4. **Propose edits** – show a minimal diff for each affected file; do not rewrite
   unchanged sections.
5. **Incorporate user feedback** – if the user supplied feedback text (see `feedback=`
   below), create a new bullet point under the relevant section of
   `.github/copilot-instructions.md` and keep a dated entry in
   `.github/copilot-feedback-log.md`.

## Inputs (optional)

- `feedback="<free-text describing what Copilot got wrong or missed>"`
- `adr=<ADR-NNN>` (limit the refresh to one ADR)
- `module=<dotted.module.path>` (limit the refresh to one module)

## Checklist on completion

- [ ] All exported symbols in `__init__.py` are reflected in `.ai/tool_description.yaml`.
- [ ] No stale method names remain in `copilot-instructions.md`.
- [ ] New ADR status changes are noted in the relevant instruction file.
- [ ] User feedback entry added to `.github/copilot-feedback-log.md` (if `feedback=` was provided).
