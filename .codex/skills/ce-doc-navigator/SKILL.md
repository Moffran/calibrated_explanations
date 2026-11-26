---
name: ce-doc-navigator
description: >
  Route any calibrated-explanations question, task, or problem to the correct skill(s), canonical files, and documentation sections in the OSS CE library. Use when unsure which skill to invoke, when looking for where something is documented, or when a task spans multiple skills. Triggers on: "which skill should I use", "where is X documented", "find the skill for", "navigate to", "what handles", "I need to do X in CE", "where do I find", "which ce skill", "what skill covers".
---

## Inputs

- **`query`** (text, required): A question, task description, concept name, or problem statement. Can be vague — the navigator's job is to resolve it.
  - Example: `I need to check if my calibration is valid under covariate shift`

## Output Format

Format: `markdown`

Required sections:
- interpretation
- primary_skill
- supporting_skills
- canonical_files
- suggested_invocation

# CE Doc Navigator — Core Instructions

You are a navigation layer over the calibrated-explanations OSS skill library.
Your job is not to answer the question directly — it is to route it to the right
skill(s), files, and documentation so the user can act immediately.

This repository (`calibrated_explanations`) is the open-source library only.
Do not route to, invent, or reference skills, packages, or deployment/
governance layers outside this repository's scope — the installable library
(calibrated prediction, explanations, plugins, serialization, docs), not
enterprise/operational deployment, monitoring, or governance tooling (see
`CONTRIBUTOR_INSTRUCTIONS.md`).

---

## Skill Registry

### Core CE Skills (`ce-` prefix)

#### Calibration & Prediction
| Skill | Handles |
|---|---|
| `ce-calibrated-predict` | Calibrated probability outputs, Venn-Abers, CPS |
| `ce-mondrian-conditional` | Conditional/group-wise calibration, Mondrian bins |
| `ce-regression-intervals` | Prediction intervals for regression tasks |
| `ce-classification` | Classification-specific calibration and explanation |
| `ce-fallback-impl` | Fallback/default calibration implementation |
| `ce-fallback-test` | Testing fallback calibration behavior |
| `ce-modality-extension` | Extending CE to new data modalities |
| `ce-integration-compare` | Comparing CE with other calibration/explanation methods |

#### Explanations & Alternatives
| Skill | Handles |
|---|---|
| `ce-alternatives-explore` | Exploring counterfactual/alternative explanations |
| `ce-explain-interact` | Interactive explanation workflows |
| `ce-factual-explain` | Factual (non-counterfactual) explanations |
| `ce-reject-policy` | Rejection/abstention policies with coverage guarantees |

#### Code Quality & Engineering
| Skill | Handles |
|---|---|
| `ce-code-quality-auditor` | Auditing CE code for quality issues |
| `ce-code-review` | Reviewing CE-related code changes |
| `ce-deadcode-hunter` | Finding dead/unused code in CE projects |
| `ce-deprecation` | Handling deprecated CE APIs and migration |
| `ce-logging-observability` | Logging and observability in CE pipelines |
| `ce-performance-tuning` | Performance optimisation of CE workflows |
| `ce-serialization-audit` | Auditing serialisation/deserialisation of CE objects |
| `ce-serializer-impl` | Implementing serialisers for CE objects |

#### Testing
| Skill | Handles |
|---|---|
| `ce-test-audit` | Auditing existing CE test coverage |
| `ce-test-author` | Writing new CE tests |
| `ce-test-creator` | Scaffolding CE test suites |
| `ce-test-pruning-expert` | Removing redundant/low-value tests |
| `ce-test-quality-method` | Assessing test quality and methodology |

#### Documentation
| Skill | Handles |
|---|---|
| `ce-docstring-author` | Writing docstrings for CE code |
| `ce-rtd-auditor` | Auditing ReadTheDocs documentation |
| `ce-rtd-writer` | Writing ReadTheDocs documentation |

#### Architecture & Design
| Skill | Handles |
|---|---|
| `ce-adr-author` | Writing Architecture Decision Records |
| `ce-adr-consult` | Consulting on architectural decisions |
| `ce-adr-gap-analyzer` | Finding gaps in ADR coverage |
| `ce-standards-gap-analyzer` | Finding gaps against engineering standards |
| `ce-plugin-audit` | Auditing CE plugin implementations |
| `ce-plugin-scaffold` | Scaffolding new CE plugins |

#### Release & Lifecycle
| Skill | Handles |
|---|---|
| `ce-release-check` | Pre-release validation checklist |
| `ce-release-finalize` | Finalising a CE release |
| `ce-release-planner` | Planning CE release scope and sequence |
| `ce-release-task` | Executing specific release tasks |

#### Meta / Skill Management
| Skill | Handles |
|---|---|
| `ce-skill-audit` | Auditing the CE skill library itself |
| `ce-skill-creator` | Creating new CE skills |
| `ce-notebook-audit` | Auditing Jupyter notebooks in CE projects |
| `ce-plot-review` | Reviewing CE visualisations/plots |
| `ce-plotspec-author` | Writing plot specifications for CE outputs |
| `ce-devils-advocate` | Stress-testing CE decisions and proposals |
| `ce-data-preparation` | Preparing datasets for CE workflows |
| `ce-onboard` | Onboarding to the CE codebase |
| `ce-payload-governance` | Governing CE API payload contracts |

---

### Universal Skills (from generic-skill-library)

| Skill | Handles |
|---|---|
| `conformal-methods-reviewer` | Theoretical review of conformal methods (coverage, exchangeability) |
| `paper-distiller` | Distilling research papers relevant to CE |
| `experiment-result-interpreter` | Interpreting CE experiment results and benchmarks |
| `rigorous-technical-writer` | Improving CE documentation and paper prose |
| `red-team-my-idea` | Stress-testing CE design proposals |
| `ai-systems-architect` | Designing systems that integrate CE |
| `ai-adoption-briefing` | Briefing stakeholders on CE capabilities |
| `decision-memo-drafter` | Structuring CE-related decisions |

---

## CE-First Contract Reminder

Before routing: the canonical CE entry points are `WrapCalibratedExplainer`,
`fit()`, `calibrate()`, and the `explain_*` / `predict*` methods.
`ce_agent_utils` is a **secondary helper layer** — never route users to it as
the primary mental model or as a substitute for the public API.

If a user asks about `ce_agent_utils` specifically:
→ Route to `ce-pipeline-builder` and explain that new agent code uses the direct
  public wrapper lifecycle; helpers remain only for legacy maintenance.

---

## Routing Logic

### By problem type

**"My calibrated probabilities look wrong"**
→ Primary: `ce-calibrated-predict`
→ Supporting: `ce-mondrian-conditional` (if group-specific)

**"I need a rejection/abstention policy with coverage guarantees"**
→ Primary: `ce-reject-policy`

**"I want to use ce_agent_utils / wrap_and_explain"**
→ Primary: `ce-pipeline-builder`
→ Note: do not use `ce_agent_utils` for new workflows; construct `WrapCalibratedExplainer` directly

**"I want to build a pipeline / start using CE"**
→ Primary: `ce-pipeline-builder`
→ Note: start with the explicit WrapCalibratedExplainer skeleton, not the helper shorthand

**"I want to generate/review alternative explanations"**
→ Primary: `ce-alternatives-explore`
→ Supporting: `ce-reject-policy` (if coverage guarantees matter)

**"I want to review a CE paper or theoretical claim"**
→ Primary: `conformal-methods-reviewer`
→ Supporting: `paper-distiller`

**"I need to write/fix tests"**
→ Primary: `ce-test-author` or `ce-test-audit`
→ Supporting: `ce-test-quality-method`

**"I need to add a plugin"**
→ Primary: `ce-plugin-scaffold`
→ Supporting: `ce-plugin-audit`

---

## Output Structure

### INTERPRETATION
Restate what the user is trying to do in one sentence.

### PRIMARY SKILL
Name and one-line reason why this is the right skill.

### SUPPORTING SKILLS
List any secondary skills, each with a one-line reason.
If none needed, say "None — primary skill is sufficient."

### CANONICAL FILES
The most authoritative files/sections for this query.
Be specific: file path + what to look for there.
If you don't know the exact path, say so — do not invent paths.

### SUGGESTED INVOCATION
A ready-to-use prompt the user can paste directly into the target skill.
Format: "Invoke `skill-name` with: [exact prompt text]"

---

## Maintenance Note

This registry must stay current and scoped to this repository's own skills.
When new skills are added under `.claude/skills/` (the canonical registry per
`CONTRIBUTOR_INSTRUCTIONS.md` §6A), add them to the appropriate section above.
Do not add skills, packages, or routing targets that do not exist in this
repository.

## Constraints

- Always name a specific primary skill — never return "it depends" without a recommendation.
- If a query spans multiple skills, rank them by relevance, do not list them equally.
- If no skill covers the query, say so explicitly rather than forcing a poor match.
- Suggested invocation must be a concrete prompt, not a description of one.
- Do not route to or mention skills, products, or layers outside this
  repository's OSS CE scope (the installable library, not enterprise/
  operational deployment or governance tooling).

## Self-Check Before Responding

- [ ] Is the primary skill specific (not just a category)?
- [ ] Is the suggested invocation a ready-to-use prompt?
- [ ] Are canonical file references specific (not just "the README")?
- [ ] Does every referenced skill actually exist in this repository?
