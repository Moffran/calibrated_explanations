# AI Agent Setup Guide for `calibrated_explanations`

> **Goal:** Configure any AI agent platform so it understands CE deeply, learns
> from your feedback, and stays in sync with every API change.
>
> The **canonical CE rules** that apply to all agents live in `CONTRIBUTOR_INSTRUCTIONS.md`.
> Platform-specific setup files build on top of that canonical:
>
> | Platform | File |
> |---|---|
> | GitHub Copilot | `.github/copilot-instructions.md` + `.github/prompts/` |
> | Codex (OpenAI) | `AGENTS.md` |
> | Claude Code | `CLAUDE.md` + `.claude/settings.json` |
> | Google Gemini | `GEMINI.md` |

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python environment | Use one venv per CE branch; install editable: `pip install -e .[dev]` |
| Agent platform | See platform-specific file for tool installation steps |
| VS Code (Copilot only) | v1.90+ with Copilot and Copilot Chat extensions |

Quick dev environment setup (all platforms):

```bash
python -m venv .venv
source .venv/bin/activate       # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -e .[dev]
```

---

## 2. How CE agent context is structured

Every agent platform reads a two-layer set of files:

### Layer 1 — Canonical (all agents)

`CONTRIBUTOR_INSTRUCTIONS.md` at the repo root contains:

- CE-first policy (the mandatory pre-explain checklist)
- Architecture and module boundary rules
- Coding standards (type hints, Numpy docstrings, lazy imports)
- Testing standards and fallback visibility policy
- ADR/Standards reference map (all 26 active ADRs + 5 STDs with when-to-consult guidance)
- Test-quality improvement method (8-agent team from `docs/improvement/test-quality-method/`)
- Key files & directories, development workflow, TDD patterns

### Layer 2 — Platform-specific

Each platform file adds only what is unique to that agent:

| File | Platform | What it adds |
|---|---|---|
| `.github/copilot-instructions.md` | GitHub Copilot | Auto-injected instruction files, prompt slash commands, `@workspace` chat tips |
| `AGENTS.md` | Codex | Session priming prompt, task template, workspace sync routine, CE-first utility list |
| `CLAUDE.md` | Claude Code | Permissions model (`.claude/settings.json`), bash tool rules, tool use guidance |
| `GEMINI.md` | Google Gemini | Session priming prompt, context management, workspace sync |

---

## 3. Session priming (all platforms)

At the start of any agent session, prime the agent with:

```text
You are a CE-first agent for calibrated_explanations. Read CONTRIBUTOR_INSTRUCTIONS.md
and <platform file> first. Use WrapCalibratedExplainer and ce_agent_utils helpers.
Fail fast if CE-first invariants are not satisfied.
```

Replace `<platform file>` with `AGENTS.md`, `CLAUDE.md`, or `GEMINI.md` as appropriate.
For GitHub Copilot, the instructions are injected automatically — no priming needed.

---

## 4. GitHub Copilot — VS Code workspace setup

Open the repository root in VS Code. Copilot features are provided by the
Copilot/Copilot Chat extensions and GitHub's instruction-file loading. This
repository does not require custom VS Code Copilot keys in `.vscode/settings.json`.

- **Copilot completions** for Python, Markdown, and YAML.
- **Next Edit Suggestions** – Copilot proposes the next logical edit as you type.
- **Instruction files** – all `.github/instructions/*.instructions.md` files are
  automatically loaded into Copilot's context based on the file you are editing.
- **Workspace agent** – Copilot can search your local files when answering questions.

Instruction files injected automatically by VS Code:

| File | Scope | Purpose |
|---|---|---|
| `.github/copilot-instructions.md` | all files | Architecture, coding standards, TDD policy, fallback rules |
| `.github/instructions/source-code.instructions.md` | `src/**/*.py` | Module layout, import rules, docstring style |
| `.github/instructions/tests.instructions.md` | `tests/**`, `*test*` | Testing framework, naming, coverage gate |
| `.github/instructions/execution plan.instructions.md` | all files | Release plan, ADR conformance, changelog policy |

Prompt slash commands available in Copilot Chat:

| Command | Use when |
|---|---|
| `/generate-tests-strict` | Writing new tests for any CE module |
| `/implement-plugin` | Scaffolding a new calibrator, plot, or explanation plugin |
| `/fix-issue` | Diagnosing and fixing a bug or failing test |
| `/refresh-ce-context` | Updating instruction files after an API change or to incorporate feedback |

---

## 5. Keeping agents up to date

### After an API change

1. For GitHub Copilot, run in Chat:
   ```
   /refresh-ce-context module=calibrated_explanations.core.calibrated_explainer
   ```
   For other platforms, ask the agent to read `CONTRIBUTOR_INSTRUCTIONS.md`, compare it
   against the current `src/calibrated_explanations/` source, and propose minimal diffs.
2. Review the proposed diffs; accept or adjust.
3. Commit the updated instruction files alongside the code change.

### After an ADR is accepted or closed

Run `/refresh-ce-context adr=ADR-NNN` (Copilot) or ask the agent to update the
ADR status row in `CONTRIBUTOR_INSTRUCTIONS.md` §9.

### Workspace sync routine (Codex / Claude / Gemini)

```bash
source .venv/bin/activate
pip install -e .[dev]
python -m pip check
pytest -q
```

Then ask the agent to re-read `CONTRIBUTOR_INSTRUCTIONS.md` and diff
`src/calibrated_explanations/` for changed signatures.

---

## 6. Feedback loop (all platforms)

No agent platform retains memory across unrelated sessions. The only way to make
feedback durable is to encode it in versioned repository files.

### Quick feedback

For GitHub Copilot, run:
```
/refresh-ce-context feedback="Agent suggested importing matplotlib at module level – it must always be lazy"
```
This appends a dated entry to `.github/copilot-feedback-log.md` and adds a
clarifying bullet to the relevant instruction file.

For other platforms, update `CONTRIBUTOR_INSTRUCTIONS.md` directly with a new bullet in
the relevant section, then add a dated entry to `.github/copilot-feedback-log.md`
manually.

Use this mandatory entry schema:
- `**Feedback:**` what the agent got wrong
- `**Root cause:**` why the miss happened
- `**Durable fix:**` exact instruction/test/script updates
- `**Verification:**` command(s) proving the fix
- `**Status:**` `open | ✅ incorporated`

### Structured feedback review

After each sprint or release:
1. Open `.github/copilot-feedback-log.md` and review accumulated entries.
2. Verify each correction is reflected in `CONTRIBUTOR_INSTRUCTIONS.md` or the platform
   instruction file.
3. Mark resolved entries ✅ and commit the changes so every team member benefits.

---

## 7. Common CE workflows (all platforms)

### Explain a prediction end-to-end

```text
Read CONTRIBUTOR_INSTRUCTIONS.md, then explain how WrapCalibratedExplainer.explain_factual
works from the public call down to the plugin registry dispatch.
```

### Scaffold a new plugin

```text
/implement-plugin plugin_type=calibrator plugin_name=isotonic_regression target_adr=ADR-013
```
(Copilot slash command) or give the equivalent instruction to any other agent.

### Run a test-quality improvement cycle

```text
Read docs/improvement/test-quality-method/README.md, then act as the test-creator
agent and produce a prioritized coverage-gap analysis.
```

### Check release readiness

```text
Read docs/improvement/RELEASE_PLAN_v1.md and list all items still open for the
current milestone.
```

### Debug a failing test

```text
/fix-issue failing_test=tests/unit/core/test_explainer.py::test_calibration_state
```

---

## 8. Tips for best results (all platforms)

- **Keep instruction files short and precise.** A densely written instruction is
  more useful than a long one.
- **Commit instruction-file updates in the same PR as the code change** so history
  stays in sync.
- **Always validate** with `make test` (or `pytest -q`) after any code change.
- **Pin ADR references** in code comments when making architectural decisions so
  future agent sessions understand *why* a pattern is used.
- **Use the test-quality-method agents** (`test-creator`, `pruner`, etc.) for large
  test changes rather than writing coverage-padding tests by hand.
