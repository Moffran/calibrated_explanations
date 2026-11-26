---
name: ce-claim-author
description: >
  Create or revise CE capability claims with stable identifiers, scoped status,
  and requirement links.
---

# CE Claim Author

## Use this skill when

- Authoring a new capability claim for a user-visible CE behaviour
- Revising an existing `CE-CAP-*.yaml` file (status, requirements list, adr_links)
- Splitting a compound claim into sub-claims
- Linking a claim to an ADR or STD that constrains it

## Inputs

- Intent description of the capability being claimed (user-visible statement)
- Governing ADR(s) and/or STD(s) (e.g. ADR-008, ADR-015)
- Public API methods covered (WrapCalibratedExplainer surface)
- Requirement IDs that decompose this claim (CE-REQ-... identifiers)

## Workflow

1. **Determine the next claim ID.** Scan `development/capabilities/claims/` for
   the highest existing `CE-CAP-<AREA>-<NNN>.yaml` for the target area. Increment.

2. **Write the claim text.** One sentence. User-visible statement. No hedging.
   Uses observable terms ("produces", "returns", "rejects") not mechanism terms
   ("calls", "stores", "computes").

3. **Fill all required fields** from the schema in
   `development/capabilities/claims/README.md`:
   - `claim_id`, `claim_type`, `owner`, `status`
   - `adr_links` — list of governing ADR IDs
   - `claim_text` — the normative, falsifiable statement
   - `public_api` — full dotted paths to public methods
   - `task_types` — applicable CE task modes
   - `requirements` — CE-REQ-... IDs derived from this claim
   - `verification.proves` — api_contract | behavioral_contract | numerical_behavior
   - `assumptions` — user/data assumptions required for the claim to hold
   - `evidence_required` — fields that must appear in evidence records

4. **Validate upward linkage.** Confirm every governing ADR in `adr_links` is an
   active (non-superseded) ADR in `development/adrs/`.

5. **Validate downward linkage.** Confirm every `CE-REQ-` ID in `requirements`
   exists in `development/capabilities/requirements/` or is being created in the
   same session.

6. **Update `claims/README.md` claim index** if the file contains a table of claims.

## Verification

```bash
python scripts/quality/validate_capability_chain.py --claims-only
```

## Output contract

Return:

1. The complete `CE-CAP-<AREA>-<NNN>.yaml` file contents.
2. A one-line summary: claim ID, claim text, ADR links, and requirement IDs.
3. Any companion README index update applied.

## Constraints

- `claim_text` must be a single, falsifiable assertion — no compound "and" sentences.
- `public_api` entries must be fully qualified dotted paths resolvable in the current codebase.
- `status` must be one of: `current`, `draft`, `superseded`.
- Every `requirements` entry must trace back to this claim in its `claim_refs` field.
- Do not invent requirement IDs — only list IDs that exist or are being co-authored.
- The claim layer is the root of the chain: claims must NOT reference TIF IDs or test IDs directly.
