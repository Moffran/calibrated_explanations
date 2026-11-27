---
applyTo: '**'
---
When asked to "proceed according to plan" or to determine the next development step, you **must**:

1. **Consult the active version plan**:
   - Read and interpret the sole active
     `development/current-work/vX.Y.Z_plan.md` file — its `## Included work`
     table, `## Excluded` list, and `## Release-specific gates`.
   - Identify the current release target, gates, and outstanding work items
     (rows whose Status is not `Done`).
   - Select the next actionable step that aligns with the plan and its
     linked GitHub issues/milestone.

2. **Maintain a future-oriented action plan and update the changelog**:
   - When an item in the action plan has been completed satisfactorily, **add it to the `CHANGELOG.md`** under the appropriate section.

3. **Enforce ADR conformance**:
   - For any implementation, design, or architectural decision, **review all relevant ADRs** in `development/adrs/`.
   - Ensure that your code, design, or recommendation strictly adheres to the protocols, contracts, and constraints described in the ADRs.
   - If a plan conflicts with an accepted ADR or active standard, **the ADR or standard takes precedence**. A plan cannot supersede it; amend or supersede the ADR or standard through its own process first.

4. **Document rationale**:
   - When proposing or generating code for a next step, briefly reference the relevant section(s) of the plan and ADR(s) that justify your choice.

5. **Test instructions**:
   - Make sure to follow the instructions in `.github/instructions/tests.instructions.md` when generating or modifying tests.

**Never** proceed with a step that is not supported by both the current plan and the ADRs. If ambiguity exists, request clarification or escalate for ADR or plan update before proceeding.
