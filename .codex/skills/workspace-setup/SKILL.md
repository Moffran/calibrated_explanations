# Skill: workspace-setup

Use this skill to set up the Kristinebergs multi-repo workspace for the first time,
or to onboard a new developer to the full agent infrastructure.

> ⚠️ **Bootstrap-only skill** — this skill guides human-performed setup steps.
> It does not auto-configure anything by itself.

---

## Overview

The workspace consists of 4 repos under a common parent directory:

```
kristinebergs/
├── calibrated_explanations/          (OSS library)
├── calibrated-explanations-enterprise/   (enterprise product)
├── calibrated-explanations-plugins/   (plugin monorepo)
├── company-knowledge-private/        (company data + policies)
└── kristinebergs.code-workspace      (multi-root VS Code workspace)
```

All repos must be cloned as siblings inside `kristinebergs/`. The workspace file
uses relative paths.

---

## Steps

### Step 1: Clone the repositories

```bash
mkdir kristinebergs && cd kristinebergs

git clone <your-org>/calibrated_explanations
git clone <your-org>/calibrated-explanations-enterprise
git clone <your-org>/calibrated-explanations-plugins
git clone <your-org>/company-knowledge-private
```

Download or copy `kristinebergs.code-workspace` to `kristinebergs/kristinebergs.code-workspace`.

### Step 2: Open the workspace in VS Code

```
File → Open Workspace from File → kristinebergs.code-workspace
```

All 4 repos appear as root folders in the Explorer.

### Step 3: Configure GitHub MCP (operational)

The workspace file has a GitHub MCP server entry. It requires a personal access token.

**Minimum scopes for read-only operation:**
- `repo:read` (issues, PRs, releases)

**Additional scope for `notion-snapshot-refresh` skill:**
- `workflow:write` (to trigger Actions)

Set the token in your VS Code settings or environment:
```json
// in VS Code user settings, or .env:
GITHUB_TOKEN=<your-token>
```

Alternatively, use the VS Code MCP server configuration UI (Settings → MCP).

### Step 4: Configure Notion MCP (bootstrap-only, optional)

See `company-knowledge-private/docs/mcp-setup.md` for the full guide.

The Notion MCP is **disabled by default** in the workspace file. To enable it:
1. Copy `company-knowledge-private/.mcp.example.json` → `company-knowledge-private/.mcp.json`
2. Add your Notion integration token to `.mcp.json`
3. Uncomment the notion server entry in `kristinebergs.code-workspace`

**Only enable Notion MCP when performing admin or bootstrap tasks.**
Disable it again when done.

### Step 5: Link Claude skills (if using Claude Code)

From `generic-skill-library/`:
```powershell
.\skills.ps1 link ../calibrated_explanations
.\skills.ps1 link ../calibrated-explanations-enterprise
```

This creates `.claude/skills/` symlinks or copies for each repo.

### Step 6: Deploy VS Code Copilot skills

From `generic-skill-library/`:
```powershell
.\skills.ps1 deploy-copilot ../calibrated_explanations
.\skills.ps1 deploy-copilot ../calibrated-explanations-enterprise
```

This populates `.github/skills/` in each target repo for VS Code Copilot discovery.

### Step 7: Verify setup

1. Open VS Code with the workspace file
2. In Copilot Chat, type `/` and confirm skills appear (e.g., `/ce-implement-test`)
3. Run `.\skills.ps1 list` to see all skills with Claude + Copilot status columns
4. Check `company-knowledge-private/exports/notion/export-index.yaml` for snapshot freshness

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Copilot skills not appearing | Verify `.github/skills/` dirs exist; run `deploy-copilot` |
| Claude skills not found | Run `skills.ps1 link` for target repo |
| Snapshot stale | Use `notion-snapshot-refresh` skill or trigger `notion-sync.yml` manually |
| Notion MCP not connecting | See `company-knowledge-private/docs/mcp-setup.md` |
| Python hook scripts fail | Check Python path; hooks use `python3` (Linux/macOS) or `python` (Windows) |

---

> **Access**: this skill requires human-performed steps. It does not write any files automatically.
> For Notion MCP setup, see `docs/mcp-setup.md`. For agent access policy, see `access/agent-access-policy.md`.
