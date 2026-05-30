---
name: ce-code-review
description: >
  Review code changes for CE coding standards, ADR conformance, and behavior or API regression risk.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Ce Code Review — Core Instructions

# CE Code Review

You are reviewing code for conformance with calibrated_explanations standards.
Work through each review dimension below and produce a finding per violation.

Load `references/review_dimensions.md` for full dimension details with code examples.

---

## Review dimensions (summary)

1. **Module boundary (ADR-001)** — CRITICAL: `core/` must never import `plugins/` internals.
2. **Lazy imports** — CRITICAL: `matplotlib`, `pandas`, `joblib` must be function-scoped.
3. **Future annotations** — REQUIRED: every `.py` file starts with `from __future__ import annotations`.
4. **Docstrings (Numpy style)** — REQUIRED: `Parameters` -> `Returns` -> `Raises` -> `Notes` -> `Examples`.
5. **Exception handling (ADR-002)** — REQUIRED: use CE exception hierarchy, not bare `ValueError`.
6. **Fallback visibility** — CRITICAL: every fallback needs `_LOGGER.info()` + `warnings.warn(UserWarning)`.
7. **Type hints** — REQUIRED: avoid `Any` without documented reason; prefix private with `_`.
8. **Deprecation (ADR-011)** — REQUIRED: use `deprecate()` helper; 2 minor releases before removal.
9. **CE-First compliance** — CRITICAL: check ALL of the following sub-criteria:
   - **Public contract fidelity** — public methods must return calibrated types and assert `fitted + calibrated` before use. Any method that skips these assertions is a CE-First violation.
   - **Shadow-API drift** — any helper (e.g. in `ce_agent_utils`) that diverges from the public `WrapCalibratedExplainer` contract without explicit delegation back to it is a CE-First violation.
   - **Silent kwarg dropping** — if a helper accepts kwargs and silently ignores or adapts them instead of passing them to the public API, that is a CE-First violation.
   - **Heuristic advice injection** — helpers must not generate advice, recommendation text, or heuristic interpretations inside canonical CE-First entry points. Such logic belongs only in narrative/plugin layers.
   - **Uncalibrated escape hatch normalization** — `calibrated=False` and similar escape hatches must never appear as a default, a fallback without warning, or as a documented "normal" usage path. They are explicit opt-outs, not canonical output modes.
   - **Façade-reinforcing tests/docs** — tests or docstrings that teach `wrap_and_explain(...)` as the *primary* example, or that mock out `fit`/`calibrate` to avoid testing the real lifecycle, are CE-First compliance issues — not merely style issues.

---

## Quick-check command

```bash
make local-checks-pr               # fast required checks
make local-checks                  # full checks (only needed for main-branch gates)
pre-commit run --all-files         # linting, ruff, mypy subset
```

---

## Review Report Template

```
CE Code Review: <module/PR name>
=================================
ADR-001 module boundary:    PASS / FAIL
  violations: <list file:line>

Lazy imports:               PASS / FAIL
  eager heavy imports:      <list>

Future annotations:         PASS / FAIL
  missing in:               <list>

Docstrings (numpy style):   PASS / FAIL
  missing sections in:      <list fn:section>

Exception handling:         PASS / FAIL
  bare exceptions at:       <list>

Fallback visibility:        PASS / FAIL
  missing warn()/log() at:  <list>

Type hints:                 PASS / FAIL
  untyped parameters:       <list>

Deprecation (ADR-011):      PASS / FAIL / N_A

CE-First compliance:        PASS / FAIL  [BLOCKING]
  public contract fidelity:   <list fn:issue>
  shadow-API drift:           <list helper:issue>
  silent kwarg dropping:      <list fn:issue>
  heuristic advice injection: <list fn:issue>
  uncalibrated escape hatch:  <list fn:issue>
  façade-reinforcing docs/tests: <list>

Overall: CONFORMANT / NON-CONFORMANT (<N> issues, <M> blocking)
```

## Evaluation Checklist

- [ ] All 9 dimensions checked.
- [ ] ADR-001 boundary violations are blocking (must fix before merge).
- [ ] Fallback visibility violations are blocking.
- [ ] Lazy-import violations are blocking.
- [ ] CE-First compliance violations are blocking (treat as CRITICAL, not style).
- [ ] CE-First sub-criteria all checked: contract fidelity, shadow-API drift, silent kwarg dropping, heuristic injection, uncalibrated escape hatch, façade-reinforcing tests/docs.
- [ ] Report produced with file:line references for each issue.


## Self-Check Before Responding

- [ ] All 9 dimensions checked.
- [ ] ADR-001 boundary violations are blocking (must fix before merge).
- [ ] Fallback visibility violations are blocking.
- [ ] Lazy-import violations are blocking.
- [ ] CE-First compliance violations are blocking (treat as CRITICAL, not style).
- [ ] CE-First sub-criteria all checked: contract fidelity, shadow-API drift, silent kwarg dropping, heuristic injection, uncalibrated escape hatch, façade-reinforcing tests/docs.
- [ ] Report produced with file:line references for each issue.