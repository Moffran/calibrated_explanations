---
applyTo: '**'
---
When asked to "proceed according to plan" or to determine the next development step, you **must**:

1. **Consult the development plan and action plan**:
   - Read and interpret the current `improvement_docs/RELEASE_PLAN_V1.md` file.
   - Identify the current and upcoming release targets, gates, and outstanding work items.
   - Select the next actionable step that aligns with the stated priorities and release sequence.

2. **Maintain a future-oriented action plan and update the changelog**:
   - When an item in the action plan has been completed satisfactorily, **add it to the `CHANGELOG.md`** under the appropriate section.
   - **Mark completed items in the `improvement_docs/RELEASE_PLAN_V1.md`** as finished.

3. **Enforce ADR conformance**:
   - For any implementation, design, or architectural decision, **review all relevant ADRs** in `improvement_docs/adrs/`.
   - Ensure that your code, design, or recommendation strictly adheres to the protocols, contracts, and constraints described in the ADRs.
   - If a conflict arises between a plan and an ADR, **the ADR takes precedence** unless the plan explicitly supersedes it.

4. **Document rationale**:
   - When proposing or generating code for a next step, briefly reference the relevant section(s) of the plan and ADR(s) that justify your choice.

**Never** proceed with a step that is not supported by both the current plan and the ADRs. If ambiguity exists, request clarification or escalate for ADR or plan update before proceeding.
