# Copilot Feedback Log

Dated entries are added here by the `/refresh-ce-context` prompt whenever
`feedback=` is supplied. Once a pattern is reflected in an instruction file,
mark the entry ✅ and it can be removed in the next cleanup pass.

This file is the shared compatibility feedback log for all agent platforms
(Copilot, Codex, Claude Code, Gemini).

Format:
```
## YYYY-MM-DD – <short description>
**Feedback:** <what Copilot got wrong or missed>
**Root cause:** <why the miss happened>
**Durable fix:** <which instruction/test/script files were updated>
**Verification:** <command(s) that prove the fix>
**Status:** open | ✅ incorporated
```


<!-- entries will be appended below this line by /refresh-ce-context -->
## 2026-02-22 – Copilot optimization loop initialization
**Feedback:** Copilot/agents do not consistently learn from feedback or update canonical instructions.
**Root cause:** Feedback log not actively used; instruction files not updated after feedback.
**Durable fix:** Added feedback log entry template; will update copilot-instructions.md and AGENT_INSTRUCTIONS.md after each feedback.
**Verification:** Check that feedback log and instructions are updated after each PR/release.
**Status:** open
