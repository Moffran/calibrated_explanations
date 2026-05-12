# AGENTS.md — Codex (OpenAI) Instructions for `calibrated_explanations`

> **Canonical agent instructions:** `CONTRIBUTOR_INSTRUCTIONS.md` — the single source of
> truth shared by all agent platforms (Copilot, Codex, Claude Code, Gemini). Read
> that first. This file adds **only** Codex-specific context.

---

## 1. Session priming (Codex-specific)

At the start of every Codex session, explicitly direct the agent to:

1. Read `CONTRIBUTOR_INSTRUCTIONS.md` (canonical CE rules).
2. Read `AGENTS.md` (this file).
3. Follow the CE-first checklist from `docs/get-started/ce_first_agent_guide.md`.
4. Use `WrapCalibratedExplainer` from the public CE API directly. Do not use
   `ce_agent_utils` as an implementation shortcut.

Reusable priming prompt:

```text
You are a CE-first agent for calibrated_explanations. Read CONTRIBUTOR_INSTRUCTIONS.md
and AGENTS.md first. Use WrapCalibratedExplainer and the public CE API directly.
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

## 3. Compact task modes (Codex-specific)

Use these modes to keep responses and validation proportional:

- `mode=quick-fix`: small edit, targeted test/script validation.
- `mode=feature`: code + tests + docs for one behavior change.
- `mode=hardening`: instruction/CI/test policy work across files.

For each mode, explicitly state:
- changed files
- validation command(s)
- residual risk

---

## 4. Workspace sync (Codex-specific)

Run this sync routine after any CE version update or after pulling changes:

```bash
source .venv/bin/activate      # or activate your environment
pip install -e .[dev]          # keep editable install current
python -m pip check            # verify no conflicts
pytest -q                      # confirm tests pass
```

Then ask Codex to:
1. Re-read `CONTRIBUTOR_INSTRUCTIONS.md` and `docs/get-started/ce_first_agent_guide.md`.
2. Diff `src/calibrated_explanations/` for changed signatures.
3. Update any examples or tests that reference old signatures.

---

## 5. Validation path (Codex-specific)

Default validation order:
1. `make local-checks-pr` (fast required checks)
2. `make local-checks` only when changes touch main-branch gates (coverage/perf/over-testing)

If a command is unavailable in the current shell, run the equivalent Python entrypoint and report it.

---

## 6. Feedback loop (Codex-specific)

Codex has no persistent cross-session memory. Convert feedback into durable files:

| Type of feedback | Where to record it |
|---|---|
| Wrong API usage | Update `CONTRIBUTOR_INSTRUCTIONS.md` §1 or §2 |
| Wrong coding style | Update `CONTRIBUTOR_INSTRUCTIONS.md` §3 |
| Recurring test mistake | Add a test to the relevant unit test file under `tests/unit/` |
| Platform-specific Codex quirk | Add a bullet to this file |
| General improvement | Update `PROMPTS.md` |

Feedback entries must be appended to `.github/copilot-feedback-log.md` using:
- `Feedback`
- `Root cause`
- `Durable fix`
- `Verification`
- `Status`

Commit the changes in the same PR so the correction is permanent.

---

## 7. CE-first lifecycle (Codex-specific)

Agents must use the public CE API directly. Do **not** use
`calibrated_explanations.ce_agent_utils` as an implementation shortcut.

```python
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=feature_names)

if not explainer.fitted or not explainer.calibrated:
    raise RuntimeError("CE-first lifecycle violation: fit and calibrate before use.")

# Factual explanations
explanations = explainer.explain_factual(X_query)
print(explanations[0].to_narrative(format="short"))

# Alternative / counterfactual explanations
alternatives = explainer.explore_alternatives(X_query)

# Calibrated predictions with uncertainty intervals
probabilities, interval = explainer.predict_proba(X_query, uq_interval=True)
```

> **Note:** `ce_agent_utils.py` is retained for backward compatibility and as a
> legacy example. It is not the recommended agent interface. New agent code must
> not import from it.
