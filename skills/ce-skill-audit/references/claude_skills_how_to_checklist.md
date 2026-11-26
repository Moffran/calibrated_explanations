# Claude Skill Quality Checklist (Paraphrased)

Reference source: `https://docs.claude.com/en/docs/agents-and-tools/claude-code/tutorials/create-a-skill`
accessed 2026-02-25.

## Authoring rules to enforce

1. Frontmatter
- Include `name` and `description`.
- Keep naming stable and directory-aligned.

2. Description quality
- Be specific about when the skill should trigger.
- Avoid vague, broad phrasing that causes over-triggering.

3. Skill file size and structure
- Keep `SKILL.md` concise and maintainable.
- Move long templates into `assets/`.
- Move deep references into `references/`.
- Place executable helpers in `scripts/` when deterministic scaffolding helps.

4. Testing and iteration
- Re-test trigger behavior after edits.
- Tighten or narrow description wording if activation is noisy.

5. Optional controls
- Use `disable-model-invocation: true` only for manual/read-only skills.
- Constrain tool access only when needed.
