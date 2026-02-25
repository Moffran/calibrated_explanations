# Skill Description Style Guide (Strict)

Use this guide for every `SKILL.md` frontmatter `description`.

## Required format

- One sentence only.
- 80-140 characters preferred (hard cap: 160).
- Start with an action verb (`Audit`, `Build`, `Write`, `Configure`, etc.).
- State the scope and trigger intent with precision.
- End with a period.

## Avoid

- Long synonym dumps.
- Quoted phrase lists.
- ADR lists in description text.
- Multi-paragraph descriptions.
- Ambiguous verbs like "handle stuff", "assist with anything".

## Pattern

`<Action + object + scope>, <trigger or usage intent>.`

Examples:
- `Audit notebooks for public-API correctness, policy compliance, and private-member violations.`
- `Build CE-first end-to-end pipelines using WrapCalibratedExplainer with fit-calibrate-explain sequencing.`

## Where long trigger lists go

Store detailed phrase catalogs in `references/trigger_phrases.md` for the skill
instead of embedding them in frontmatter.
