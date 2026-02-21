# GitHub Copilot Setup Guide for `calibrated_explanations`

> **Goal:** Configure an environment where Copilot understands CE deeply, learns from
> your feedback, and stays in sync with every API change.

---

## 1. Prerequisites

| Tool | Minimum version | Notes |
|---|---|---|
| VS Code | 1.90 | Required for instruction-file support |
| GitHub Copilot extension | latest | Chat + completions |
| GitHub Copilot Chat extension | latest | Prompt files and `/` commands |
| Python extension (ms-python) | latest | Test runner integration |

Install the Copilot extensions from the VS Code Marketplace or via the CLI:
```bash
code --install-extension GitHub.copilot
code --install-extension GitHub.copilot-chat
```

---

## 2. Open the workspace

Open the repository root in VS Code. The workspace settings in `.vscode/settings.json`
activate the following automatically:

- **Copilot completions** for Python, Markdown, and YAML.
- **Next Edit Suggestions** – Copilot proposes the next logical edit as you type.
- **Instruction files** – all `.github/instructions/*.instructions.md` files are
  automatically loaded into Copilot's context based on the file you are editing.
- **Workspace agent** – Copilot can search your local files when answering questions.

---

## 3. Repository-level instruction files

These files feed Copilot context automatically – you do not need to paste them into
the chat yourself.

| File | Scope | Purpose |
|---|---|---|
| `.github/copilot-instructions.md` | all files | Architecture, coding standards, TDD policy, fallback visibility rules |
| `.github/instructions/source-code.instructions.md` | `src/**/*.py` | Module layout, import rules, docstring style, error handling |
| `.github/instructions/tests.instructions.md` | `tests/**`, `*test*` | Testing framework, naming, structure, coverage gate |
| `.github/instructions/execution plan.instructions.md` | all files | Release plan, ADR conformance, changelog policy |

---

## 4. Reusable prompt files (slash commands)

Type `/` in Copilot Chat to see all available prompts. CE-specific prompts:

| Prompt | Use when |
|---|---|
| `/generate-tests-strict` | Writing new tests for any CE module |
| `/implement-plugin` | Scaffolding a new calibrator, plot, or explanation plugin |
| `/fix-issue` | Diagnosing and fixing a bug or failing test |
| `/refresh-ce-context` | Updating instruction files after an API change or to incorporate feedback |

---

## 5. Keeping Copilot up to date

### After an API change

1. Open Copilot Chat and run:
   ```
   /refresh-ce-context module=calibrated_explanations.core.explainer
   ```
2. Review the proposed diff; accept or adjust.
3. Commit the updated instruction files alongside the code change.

### After an ADR is accepted or closed

1. Run:
   ```
   /refresh-ce-context adr=ADR-NNN
   ```
2. The prompt will suggest targeted edits to `.github/copilot-instructions.md`
   and `.github/instructions/source-code.instructions.md`.

---

## 6. Providing feedback to Copilot (learning loop)

Copilot does not have a built-in persistent memory, but you can close the feedback
loop by updating the instruction files directly.

### Quick feedback during a chat session

When Copilot gives a wrong answer, use the thumbs-down button **and** follow up with:

```
/refresh-ce-context feedback="Copilot suggested importing matplotlib at module level – it must always be lazy"
```

This creates a dated entry in `.github/copilot-feedback-log.md` **and** adds a
clarifying bullet to the relevant instruction file, so the same mistake will not
recur in future sessions.

### Structured feedback review

Schedule a periodic review (e.g. after each sprint or release):

1. Open `.github/copilot-feedback-log.md` and review accumulated entries.
2. For each pattern of errors, update the relevant instruction file and remove the
   log entry once incorporated.
3. Commit the changes so every team member benefits.

---

## 7. Recommended Copilot Chat workflows

### Explain a prediction end-to-end

```
@workspace explain how WrapCalibratedExplainer.explain_factual works, starting from
the public call to the plugin registry dispatch
```

### Scaffold a new plugin

```
/implement-plugin plugin_type=calibrator plugin_name=isotonic_regression target_adr=ADR-013
```

### Check release readiness

```
@workspace which items in docs/improvement/RELEASE_PLAN_v1.md are still open for
the current milestone?
```

### Debug a failing test

```
/fix-issue failing_test=tests/unit/core/test_explainer.py::test_calibration_state
```

---

## 8. Tips for best results

- **Always use `@workspace`** for questions about CE internals – it gives Copilot access
  to the indexed source files.
- **Keep instruction files short and precise.** Copilot's context window is limited;
  a densely written instruction is more useful than a long one.
- **Commit instruction-file updates in the same PR as the code change** so the two
  stay in sync in git history.
- **Use the `/generate-tests-strict` prompt** when writing tests – it enforces the
  CE test rubric automatically.
- **Pin ADR references** in code comments when making architectural decisions so
  future Copilot sessions understand *why* a pattern is used.
