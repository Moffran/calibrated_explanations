# Agent Instructions for `calibrated_explanations`

> **Canonical source of truth** for repository-wide contributor and agent
> instructions. Platform files (`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, and
> `.github/copilot-instructions.md`) add only platform-specific guidance.

## Information ownership

- This file owns globally mandatory operating rules and navigation.
- Accepted/proposed ADRs in [`development/adrs/README.md`](development/adrs/README.md)
  own architectural decisions.
- Engineering standards in
  [`development/standards/README.md`](development/standards/README.md) own
  engineering and quality requirements.
- Skills in [`skills/README.md`](skills/README.md) own
  task-specific workflows.
- Documentation owns verified API guidance and runnable examples.
- Generated indexes are discovery surfaces, never independent authority.
- Scripts and verification gates enforce deterministic checks.

If sources conflict, the governing ADR or engineering standard wins over a
release plan, a skill, and this operational summary.

---

## 1. Required CE-first lifecycle

Use the public `calibrated_explanations` API directly:

1. Import `WrapCalibratedExplainer` from `calibrated_explanations`.
2. Instantiate `WrapCalibratedExplainer` directly. A subclass is acceptable only
   when it is an intentional, verified extension that preserves the wrapper
   contract; never invent an ad-hoc wrapper.
3. Call `fit(x_proper, y_proper)` and require `fitted is True`.
4. Call `calibrate(x_cal, y_cal, ...)` and require `calibrated is True`.
5. Use `explain_factual`, `explore_alternatives`, `predict`, or `predict_proba`.
6. Return calibrated outputs and uncertainty by default. Do not silently return
   uncalibrated output.
7. Verify the fit/calibrate lifecycle explicitly in the calling code.

Authoritative runnable examples live in
[`docs/get-started/ce_first_agent_guide.md`](docs/get-started/ce_first_agent_guide.md).

Use guarded explanations through `guarded_options=GuardedOptions()`. The removed
`explain_guarded_factual`, `explore_guarded_alternatives`, and
`guarded=True` surfaces must not be used.

---

## 2. Architecture and coding invariants

Consult the generated ADR index, then read every relevant accepted source file.
Common boundaries are:

- `core/` never imports `plugins/`; extensible behavior belongs in `plugins/`.
- Runtime configuration reads go through `ConfigManager` (ADR-034).
- Heavy optional libraries are imported lazily.
- Use the unified exception hierarchy in `core.exceptions` (ADR-002).
- Use `typing.Protocol` and the existing plugin registry rather than deep
  inheritance or a new dependency-injection framework.
- Preserve stable public APIs and scientific semantics unless the approved plan
  explicitly requires a change.
- Add `from __future__ import annotations` to Python source, use comprehensive
  type hints, and write NumPy-style docstrings (STD-002).
- Use `utils.deprecate` and the ADR-011 lifecycle for public deprecations.

Do not convert ADRs or engineering standards into skills. Do not add a second
policy hierarchy, master roadmap, requirement catalogue, or canonical registry.

---

## 3. Testing and capability verification

[`tests/README.md`](tests/README.md) is the single test authority; ADR-030 and
STD-003 govern quality and coverage.

- Use pytest and the existing fixtures.
- Extend the nearest existing test file first.
- Name tests `test_should_<behavior>_when_<condition>`.
- Structure tests as Arrange–Act–Assert and assert observable behavior.
- Keep tests deterministic; do not access private implementation details merely
  for coverage.
- Keep normal fallback chains disabled through the autouse fixtures. A fallback
  behavior test must opt in with `enable_fallbacks`.
- The package coverage gate is at least 90%; preserve documented exceptions.

For externally visible capability claims, preserve ADR-040's chain:

```text
ADR / Standard
  -> constrains capability claim
  -> decomposes into requirement
  -> exercised by TIF or documented exemption
  -> executed by test / gate / review
  -> recorded as evidence
```

Use the canonical capability locations in `development/capabilities/`. A passing
test is execution, not evidence, and does not prove assumptions outside the
documented boundary. Run `make capability-chain-check` for structural changes to
claims, requirements, TIFs, tests, or evidence.

---

## 4. Fallback visibility

Every retained user-visible fallback must:

1. Emit `warnings.warn(..., UserWarning)`.
2. Emit an `INFO` log summarizing the failure and chosen path.
3. Be documented in `CHANGELOG.md` when newly introduced.

Narrow internal or best-effort exceptions are allowed only when
`scripts/quality/check_warning_policy.py` records a site-specific disposition and
reason with the strongest available logger signal. An exemption is not a blanket
policy waiver and must never be inferred from wording in a skill or overlay.

---

## 5. Canonical locations and generated discovery

Start with [`development/README.md`](development/README.md), then use:

| Material | Canonical location |
|---|---|
| Proposed work | GitHub issues |
| Approved release scope | GitHub milestones |
| Active release coordination | `development/current-work/vX.Y.Z_plan.md` |
| Closed planning/evidence | `development/finished-work/` |
| Architectural decisions | [`development/adrs/`](development/adrs/) |
| Engineering standards | [`development/standards/`](development/standards/) |
| Capability verification | `development/capabilities/` |
| Test quality method | [`development/standards/test-quality-method/`](development/standards/test-quality-method/) |
| Test guidance | [`tests/README.md`](tests/README.md) |
| Canonical skills | [`skills/`](skills/) |
| Generated skill discovery packages | `.claude/skills/` (Claude Code), `.agents/skills/` (Codex, GitHub Copilot, Gemini CLI) |
| Public contributor docs | [`docs/contributor/`](docs/contributor/) |

No new root-level directory may be created without explicit maintainer approval.
The removed legacy documentation roots docs/improvement and docs/standards must
not be recreated.

The generated indexes are:

- `development/adrs/README.md`
- `development/standards/README.md`
- `skills/README.md`
- `docs/contributor/agent_skills.md`

After an ADR, standard, or canonical skill change:

```bash
make instruction-index-sync
make instruction-index-check
```

`--write` updates derived files; verification uses the non-mutating `--check`
mode. CI validates generated content but does not repair it.

Every CE skill, with its references, scripts and assets, has exactly one source
under `skills/<name>/`. The platform discovery directories `.claude/skills/`
and `.agents/skills/` are complete generated copies (`[tool.skill-packages]`
in `pyproject.toml`); never edit them. `make instruction-index-sync` refreshes
indexes and packages together; `make skills-sync` / `make skills-check`
handle the packages alone. Skills owned outside this repository are installed
at user level, not committed here. Run `make check-agent-instructions` when
touching instruction surfaces.

---

## 6. Planning authority and release workflow

The planning hierarchy is:

1. `development/README.md`
2. GitHub issues (proposed work) and milestones (approved scope)
3. The sole active version plan
4. ADRs and engineering standards

Before implementation:

1. Read the active `development/current-work/vX.Y.Z_plan.md`.
2. Confirm an issue/milestone or explicit maintainer instruction supports the
   requested work.
3. Identify only the ADRs and standards relevant to the change and read their
   decision/normative sections.
4. Do not require every agent to read every ADR or skill at session start.

Do not redesign the planning hierarchy or create another roadmap. Do not mark a
plan row `Done` or close an issue until its verification evidence exists. Update
`CHANGELOG.md` under `[Unreleased]` when a plan item is completed satisfactorily.

---

## 7. Verification profiles

Set up and synchronize the development environment using the repository
dependency constraints when tools are unavailable.

```bash
make quick                                      # inner loop
make local-checks-task TASK=<n>                 # mapped release task closure
make local-checks-pr                            # blocking PR preflight
make local-checks-full                          # maintainer-requested final gate
make local-checks-release                       # release boundary only
make ci-local-runblocks                         # workflow run-block smoke
```

Use task-proportional validation. `quick` and `pr` contain the read-only
instruction-index check. Add a new standalone CI workflow only when the existing
inventory demonstrably cannot support the gate (ADR-035).

---

## 8. Python commands

Do not use Python stdin heredocs. Use `python -c` for short probes and versioned
script entry points for repository checks:

```bash
python -c "import calibrated_explanations as ce; print(ce.__version__)"
python scripts/quality/sync_instruction_indexes.py --check
```

---

## 9. Feedback persistence

Convert recurring or cross-platform agent errors into durable repository changes,
not session-only notes:

1. Prefer a focused rule, test, or skill change at the authoritative source.
2. Record feedback in `.github/copilot-feedback-log.md` with `**Feedback:**`,
   `**Root cause:**`, `**Durable fix:**`, `**Verification:**`, and `**Status:**`.
3. Commit the durable fix and evidence in the same PR.
