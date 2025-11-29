# STRATEGY_REV Log – ADR-003 Caching Strategy

**Purpose:** Audit trail of cache invalidation triggers and version bumps.

Per ADR-003, the cache uses versioned keys to enable invalidation when algorithm parameters, code logic, or strategy implementation changes affect the meaning of cached values. This log records when and why version bumps occur.

---

## Version History

| Version | Date | Reason | Tickets | Changes |
|---|---:|---|---|---|
| v1 | 2025-11-29 | Initial deployment (v0.10.0) | ADR-003 closure | First stable cache backend with cachetools, LRU/TTL eviction, telemetry |

---

## Future Bump Triggers

**Scenarios requiring version reset:**

1. **Calibrator Algorithm Change**
   - Example: Venn-Abers update, new interval calibrator
   - Action: Bump version, update log, notify via release notes

2. **Feature Schema Change**
   - Example: Categorical encoding method changes
   - Action: Bump version, document backward-compatibility plan

3. **Library Dependency Update**
   - Example: Breaking change in scikit-learn or crepes
   - Action: Bump version only if semantics affected (not every patch)

4. **Core Model Prediction Path Change**
   - Example: Prediction logic refactored; results drift > tolerance
   - Action: Bump version, add regression test to catch future drift

5. **Hardware/Platform Migration**
   - Example: Float precision handling differs across platforms
   - Action: Consider platform-specific version tags if necessary

---

## Release Checklist Integration

**v1.0.0 and onwards:** Before each release:

```
[ ] Review STRATEGY_REV_LOG – any algorithm changes since last version?
[ ] If changes: bump version, commit log update
[ ] If no changes: log notes "No changes" + date
[ ] Update CHANGELOG with cache version info (optional)
[ ] Run cache-enabled tests: `CE_CACHE=1 pytest tests/...`
[ ] Verify telemetry dashboards capture signals
```

---

## Notes for Implementers

- **Backward Compatibility:** Cache is disabled by default; users opt-in. Old configs continue working until `reset_version()` is called.
- **Performance:** Version bumps don't require physical cache wipe; stale entries evicted gradually by LRU/TTL policies.
- **Visibility:** Each bump is logged here for audit; developers and DevOps can cross-reference with commit history.
- **Automation:** Future: tie version bumps to CI checks (e.g., test output hash comparison).

---

**Last Updated:** 2025-11-29  
**Owner:** ADR-003 Maintainer  
**Review Frequency:** Per-release (before v1.0.0-rc, then before each minor release)
