# Standard-004: Documentation Audience Standard

> **Status note (2026-03-18):** Last edited 2026-03-18 · Retain indefinitely as an engineering standard.

Formerly ADR-027. Reclassified as an engineering standard to keep ADRs scoped to
architectural or contract decisions.

Status: Active

## 1. Purpose

This standard keeps documentation audience-first while making guarantee/assumption
language precise, mode-specific, and reviewable.

## 2. Normative language

The key words **MUST**, **SHOULD**, and **MAY** are to be interpreted as described in RFC 2119.

## 3. Scope and entry-point taxonomy

An **entry point** is any page that users are expected to open first for a task,
workflow, or conceptual understanding. Entry points are grouped into three tiers:

| Tier | Page types | Primary audience goal |
| --- | --- | --- |
| Tier 1 | README, landing pages, decision-tree pages, 60-second pages | Fast orientation and next step |
| Tier 2 | Quickstarts, task pages, playbooks | Execute a concrete workflow correctly |
| Tier 3 | Foundations, reference, contributor-facing semantics pages | Understand formal semantics, assumptions, and limits |

## 4. Audience-first structure requirements

1. Documentation navigation **MUST** start with getting-started and audience hubs
   (practitioner, researcher, contributor/maintainer) before deep architecture.
2. Tier 1 pages **MUST** stay brief and action-oriented; they are not the place for
   full semantics prose.
3. Tier 2 pages **MUST** include runnable guidance for the task they cover.
4. Tier 3 pages **MUST** hold the full semantics and non-guarantees detail.
5. Every entry-point page **MUST** include an explicit tier label at the bottom
   of the page in the form `Entry-point tier: Tier 1|2|3`.

## 5. Guarantees/assumptions policy (tiered and mode-specific)

### 5.1 Global rules

1. Guarantee language **MUST** be consistent with ADR-021.
2. Pages **MUST NOT** collapse all model modes into one undifferentiated semantics
   statement.
3. When semantics are stated, pages **MUST** separate these labels explicitly:
   - **Calibration prerequisites**
   - **Mode-specific guarantees**
   - **Assumptions**
   - **Explicit non-guarantees**
   - **Explanation-envelope / feature-level interval limits**
4. User-facing pages (Tier 1 and Tier 2) **MUST** link first to an internal RTD
   semantics page under `docs/foundations/` (or equivalent RTD route), not directly
   to raw GitHub ADR URLs.
5. Tier 3 pages **MAY** additionally link directly to ADR-021 on GitHub.

### 5.2 Tier-specific requirements

#### Tier 1 (README / landing / decision-tree / 60-second)

Tier 1 pages **MUST** include a short semantics pointer (1-3 lines) that:
- identifies that guarantees are mode-specific,
- names the internal RTD semantics page, and
- avoids detailed guarantee text.

Tier 1 pages **MUST NOT** include a full "Guarantees & Assumptions" block unless the
page itself is designated Tier 3.

#### Tier 2 (quickstart / task / playbook)

Tier 2 pages **MUST** contain a **mode-specific semantics note** for the workflow
shown. The note **MUST** specify the applicable mode:
- **Classification** (Venn-Abers probability intervals)
- **Percentile/interval regression** (CPS percentile intervals, no threshold)
- **Probabilistic/thresholded regression** (CPS + Venn-Abers event probabilities)

For the chosen mode, the note **MUST** include concise bullets for:
- calibration prerequisites,
- guarantees,
- assumptions,
- explicit non-guarantees,
- explanation-envelope / feature-level interval limits,
- link to internal RTD semantics page.

Tier 2 pages **SHOULD** keep this note to about 6-12 lines total.

#### Tier 3 (foundations / reference / contributor semantics)

Tier 3 **semantics-source pages** **MUST** provide full mode-specific semantics
coverage for all three modes and **MUST** include ADR-021-aligned non-guarantees, including:
- dependence on exchangeability/distribution match,
- no guarantee under drift or regime shift,
- limits on propagated feature-level interval claims.

Other Tier 3 pages **MAY** stay scope-limited if they link to the canonical Tier 3
semantics-source page.

Tier 3 semantics-source pages **MUST** be the normative target that Tier 1 and Tier 2 pages link to.

## 6. Anti-bloat controls

1. Tier 1 pages **MUST NOT** duplicate Tier 3 semantics prose.
2. Tier 2 pages **MUST NOT** copy large semantics blocks from Tier 3; they should
   summarize only what is needed for the demonstrated mode.
3. Repeated semantics text across multiple Tier 2 pages **SHOULD** be replaced with a
   shared include or a common RTD subsection.

## 7. Reviewer acceptance criteria (pass/fail)

A PR changing docs passes this standard only if all applicable checks pass:

1. **Tier identified:** Each changed entry point has a bottom-of-page label
   indicating Tier 1, 2, or 3.
2. **Correct semantics depth:**
   - Tier 1 has pointer-only semantics text.
   - Tier 2 has mode-specific note with required labeled elements.
   - Tier 3 semantics-source pages have full three-mode treatment; other Tier 3
     pages provide scoped content plus explicit pointer to the semantics-source page.
3. **No semantics flattening:** No page presents one combined guarantee statement for
   classification + interval regression + probabilistic regression.
4. **Non-guarantees present where required:** Tier 2 and Tier 3 include explicit
   non-guarantees aligned with ADR-021.
5. **Link routing correct:** Tier 1/2 link first to internal RTD semantics page;
   direct raw GitHub ADR links are only in Tier 3 or deeper technical pages.
6. **Audience-first preserved:** Navigation and page order keep getting-started and
   audience hubs ahead of deep architecture references.
7. **Bloat check:** Tier 1 remains concise; Tier 2 semantics note remains concise and
   scoped to the workflow mode.

## 8. Migration requirements by page type

- **README / landing / decision-tree pages:** replace generic guarantee boxes with a
  short mode-aware semantics pointer and internal RTD link.
- **Quickstarts / task pages / playbooks:** replace generic all-mode guarantee text
  with one mode-specific semantics note using required labels.
- **Foundations / reference / contributor semantics pages:** consolidate full
  semantics, assumptions, and non-guarantees as the canonical source; include
  ADR-021 links as supporting references.

## 9. References

- ADR-021: Calibrated Interval Semantics
- ADR-012: Documentation & Gallery Build Policy
- ADR-026: Explanation Plugin Semantics
- STD-002: Code Documentation Standard
- Capability Manifest: {doc}`../tasks/capabilities`
- Terminology Guide: {doc}`../foundations/concepts/terminology`
