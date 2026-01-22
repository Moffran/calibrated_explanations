# Roadmap

`ROADMAP.md` summarises the release milestones defined in
`docs/improvement/RELEASE_PLAN_v1.md`. Each entry below lists the flagship work
streams that gate the milestone; follow the linked release plan for the full
implementation checklist (and the companion `docs/improvement/vx.y.z_plan.md` for
vx.y.z specifics).

## Milestone summaries

- **v0.10.1 (schema & visualization contracts)** – Finalise the v1 explanation
  schema contract, refresh fixtures/docs, stabilise PlotSpec metadata/validation,
  ship the finished visualization plugin architecture (with legacy fallback),
  refresh the legacy plotting maintenance reference, document dynamically
  generated plugin classes, prototype streaming exports with telemetry, continue
  anti-pattern remediation, and complete the open-source readiness workstream
  (community health files, API map, dependency scanning, and contribution
  licensing).
- **v0.10.2 (plugin trust & packaging compliance)** – Enforce ADR-006 trust
  controls, close ADR-013/ADR-015 protocol gaps (FAST integration and CLI
  diagnostics), split optional dependencies per ADR-010, surface interval/FAST
  metadata in telemetry, and bake CI rules to ban new private-member tests.
- **v0.11.0 (domain model & preprocessing finalisation)** – Make the ADR-008
  domain model the canonical runtime story, finalise ADR-009 preprocessing and
  Standard-001 cleanup, extend governance dashboards, and shift
  `condition_source` defaults to prediction with documented upgrade guidance.
- **v1.0.0-rc (release candidate readiness)** – Freeze schema/PlotSpec contracts,
  reconfirm wrap/exception contracts, close Standard-001/018 guardrails,
  validate caching/parallel toggles (with telemetry), cement Standard-003
  coverage dashboards and release branch policies, promote ADRs, launch
  versioned docs previews, and publish an RC upgrade checklist plus ADR gap
  audit.
- **v1.0.0 (stability declaration)** – Announce the stable contracts, tag the
  release, backport docs and upgrade guidance, validate telemetry/plugin/caching
  behaviour in staging, keep Standard-001/018 guardrails enforced, and publish
  long-term documentation dashboards with scheduled maintenance sweeps.

## Staying aligned

- Read `docs/improvement/RELEASE_PLAN_v1.md` for the detailed milestone breakdown
  and release gate narratives.
- Follow `docs/improvement/vx.y.z_plan.md` whenever you work on vx.y.z tasks
  and the related pillars (schema, visualization, docs, streaming, and readiness).
- Use `docs/foundations/governance/release_checklist.md` to verify the lightweight
  gates before cutting a release and to find the next checklist owner.
