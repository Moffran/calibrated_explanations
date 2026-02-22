# AGENTS.md — Codex (OpenAI) Instructions for `calibrated_explanations`

> **Canonical agent instructions:** `AGENT_INSTRUCTIONS.md` — the single source of
> truth shared by all agent platforms (Copilot, Codex, Claude Code, Gemini). Read
> that first. This file adds **only** Codex-specific context.

---

## 1. Session priming (Codex-specific)

At the start of every Codex session, explicitly direct the agent to:

1. Read `AGENT_INSTRUCTIONS.md` (canonical CE rules).
2. Read `AGENTS.md` (this file).
3. Follow the CE-first checklist from `docs/get-started/ce_first_agent_guide.md`.
4. Use helpers from `src/calibrated_explanations/ce_agent_utils.py` instead of
   ad-hoc wrappers.

Reusable priming prompt:

```text
You are a CE-first agent for calibrated_explanations. Read AGENT_INSTRUCTIONS.md
and AGENTS.md first. Use WrapCalibratedExplainer and ce_agent_utils helpers.
Fail fast if CE-first invariants are not satisfied.
```

---

## 2. Task template (Codex-specific)

Use this structure for all implementation tasks to get consistent, reviewable output:

```text
Task:
- Goal: <what to build or fix>
- Constraints: CE-first only; no custom wrapper classes; keep public API stable unless requested.
- Validation: run `make test` and summarize failures.
- Deliverables: patch + concise rationale + migration notes if API changed.
```

---

## 3. Workspace sync (Codex-specific)

Run this sync routine after any CE version update or after pulling changes:

```bash
source .venv/bin/activate      # or activate your environment
pip install -e .[dev]          # keep editable install current
python -m pip check            # verify no conflicts
pytest -q                      # confirm tests pass
```

Then ask Codex to:
1. Re-read `AGENT_INSTRUCTIONS.md` and `docs/get-started/ce_first_agent_guide.md`.
2. Diff `src/calibrated_explanations/` for changed signatures.
3. Update any examples or tests that reference old signatures.

---

## 4. Feedback loop (Codex-specific)

Codex has no persistent cross-session memory. Convert feedback into durable files:

| Type of feedback | Where to record it |
|---|---|
| Wrong API usage | Update `AGENT_INSTRUCTIONS.md` §1 or §2 |
| Wrong coding style | Update `AGENT_INSTRUCTIONS.md` §3 |
| Recurring test mistake | Add a test to `tests/unit/test_ce_agent_utils.py` |
| Platform-specific Codex quirk | Add a bullet to this file |
| General improvement | Update `PROMPTS.md` |

Commit the changes in the same PR so the correction is permanent.

---

## 5. CE-first utilities (Codex-specific)

Prefer these helpers over ad-hoc code:

- `ensure_ce_first_wrapper(model)` — wraps and validates
- `fit_and_calibrate(explainer, x_proper, y_proper, x_cal, y_cal)` — fit+calibrate
- `explain_and_narrate(explainer, X, mode)` — explain + return narrative
- `wrap_and_explain(model, x_proper, y_proper, x_cal, y_cal, X_query)` — full pipeline
- `probe_optional_features(explainer)` — check which optional features are available

All helpers are in `src/calibrated_explanations/ce_agent_utils.py`.
