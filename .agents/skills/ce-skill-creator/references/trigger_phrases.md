# Trigger Phrase Guidance

When a skill description accumulates many trigger phrases, move the examples
out of YAML frontmatter and into a small reference file like this one.

Use trigger phrases to clarify activation cues, not to widen the skill's scope.

Good patterns:

- "Use when the user asks to update an ADR"
- "Use when the task is a CE-first release-plan implementation"
- "Use when a repository skill listing changed and registries must be synced"

Avoid:

- long catch-all lists that overlap many unrelated skills
- vague phrases like "any code task" or "documentation work"
- duplicating the same trigger wording across multiple skills
