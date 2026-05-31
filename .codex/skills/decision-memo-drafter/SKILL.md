---
name: decision-memo-drafter
description: >
  Generate a concise, structured decision memo covering the decision to be made, options, tradeoffs, recommendation, risks, and open questions. Use for organizational decisions, research direction choices, procurement, or any situation requiring a structured written recommendation. Triggers on: "write a decision memo", "draft a recommendation", "help me decide", "structure this decision", "options analysis", "I need to recommend", "decision brief", "pros and cons structured".
---

## Inputs

- **`situation`** (text, required): The decision context: what needs to be decided, by whom, and by when.
- **`options`** (text, optional): Known options. If not provided, the skill will generate them.
- **`constraints`** (text, optional): Hard constraints that eliminate options or bound the decision space.

## Output Format

Format: `markdown`

Required sections:
- decision_statement
- context
- options
- tradeoff_analysis
- recommendation
- risks
- open_questions
- next_actions

# Decision Memo Drafter - Core Instructions

You are drafting a memo that helps a real decision get made.
The memo must be compact, explicit, and useful under time pressure.

Clarify the decision before evaluating options. If the decision statement is
fuzzy, the rest of the memo will drift.

Treat options asymmetrically only when the evidence justifies it. Always include
the status quo so the recommendation is compared against doing nothing.

The recommendation must be supported by the tradeoff analysis above it.
Do not hide behind "it depends." If uncertainty is material, state what would
change the recommendation and put that in open questions.

Risks must be attached to the recommended path, not listed as a generic warning
section detached from the actual choice.


## Constraints

- Always include "do nothing / status quo" as an option.
- The recommendation must be specific — not "it depends on priorities".
- Risks must be for the recommended option, not generic.
- Next actions must have explicit owners (even if placeholder for decision maker).

## Self-Check Before Responding

- [ ] Is the decision statement precise and time-bounded?
- [ ] Is "do nothing" included as an option?
- [ ] Is the recommendation unambiguous?
- [ ] Are risks specific to the recommended path?
