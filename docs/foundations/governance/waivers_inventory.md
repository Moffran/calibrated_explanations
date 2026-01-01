# Coverage waiver inventory

Track outstanding Standard-019 coverage waivers and remediation ownership. Update
this table whenever a waiver is issued, extended, or retired so the runtime tech
lead can audit the state during release preparation.

| Module | Justification | Owner | Expiry | Status | Follow-up |
| --- | --- | --- | --- | --- | --- |
| _None_ | Waivers are retired for v0.9.0; add rows here only if new waivers are required. | Runtime tech lead | – | Retired | – |

## Update process

1. Document any new waiver request in the associated pull request using the
   release template and link to the issue tracking remediation.
2. Add a row to the table above with the module path, justification summary, owner
   (defaulting to the runtime tech lead), planned expiry, current status, and a
   hyperlink to the follow-up issue.
3. When the waiver is retired, update the status to **Retired**, include the
   completion pull request link, and remove the row after the next release audit.
4. If no waivers remain, keep the placeholder row so readers know the inventory
   is intentionally empty.

Record the runtime tech lead's sign-off for the latest audit in the
:doc:`release_checklist`.
