# Skill: notion-snapshot-refresh

Use this skill when the company-knowledge snapshot is stale (> 7 days old)
and you need fresh Notion data before running an admin session.

> ⚠️ **Bootstrap-only skill** — this skill requires the GitHub MCP to be configured
> with `workflow:write` permission scope. It is not for operational use.

---

## Prerequisites

- GitHub MCP configured in `kristinebergs.code-workspace`
- GitHub token with `workflow:write` scope for the `company-knowledge-private` repo
- The `notion-sync.yml` workflow exists in `company-knowledge-private`

---

## Steps

### 1. Check current snapshot age

Read `company-knowledge-private/exports/notion/export-index.yaml`.
Find the `exported_at` or `last_sync` field and compute age in days.

If age < 1 day: snapshot is fresh — **stop here, no refresh needed**.
If age 1–7 days: note it but proceed only if the user explicitly requests.
If age > 7 days: **proceed with refresh**.

### 2. Trigger the refresh workflow

Use the GitHub MCP to trigger a `workflow_dispatch` event:

```
Repository: company-knowledge-private
Workflow: notion-sync.yml
Event: workflow_dispatch
Inputs: {} (no inputs required)
```

Wait for acknowledgment that the dispatch was accepted (HTTP 204).
The workflow typically completes in 2–5 minutes.

### 3. Confirm completion

After 3–5 minutes, check the workflow run status via GitHub MCP:
- List recent workflow runs for `notion-sync.yml`
- Confirm the latest run has `conclusion: success`

If the workflow fails:
1. Report the failure reason from the workflow log
2. Do NOT retry automatically — wait for human intervention
3. Suggest checking: Notion API token expiry, Notion MCP configuration, workflow file validity

### 4. Verify the snapshot

After a successful run, re-read `export-index.yaml` and confirm:
- `exported_at` / `last_sync` is within the last hour
- At least one `.json` file in `exports/notion/snapshots/` has a recent modification time

### 5. Summarize

Report:
- Previous snapshot age
- New snapshot timestamp
- Which snapshot files were updated (list filenames)
- Ready to proceed with admin session: ✅

---

## Fallback (No GitHub MCP)

If the GitHub MCP is not available, instruct the user to:
1. Open `company-knowledge-private` on GitHub
2. Navigate to Actions → `notion-sync.yml`
3. Click "Run workflow" → "Run workflow"
4. Wait for completion
5. Pull the updated repo: `git pull origin main`

See `docs/mcp-setup.md` for GitHub MCP configuration.

---

> **Access policy**: requires `workflow:write` GitHub token.
> This skill modifies live Notion-derived data. Verify Notion MCP connectivity before proceeding.
