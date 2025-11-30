# Quick Reference: ADR-001 Gap Analysis

**Analysis Date**: 2025-11-30  
**Status**: Complete & Ready for Review  
**Documents**: 7 strategic files (100 KB total)

---

## The Problem (In One Sentence)

**153 cross-sibling import violations detected by linter; need to decide: refactor code or document as intentional?**

---

## The Findings

| Metric | Value |
|:---|:---:|
| Total violations | 153 |
| Violation patterns | 6 identified |
| Strategic options | 4 (A/B/C/D) |
| Top problem file | core/calibrated_explainer.py (25 violations) |
| Top violation type | core.exceptions imports (57) |

---

## The Options (Quick Comparison)

| Option | What | Effort | When | Status |
|:---|:---|:---:|:---|:---|
| **A: Allow-List** | Document violations as intentional | 2h | NOW (v0.10.0) | ✅ RECOMMENDED |
| **B: Contracts** | Create contracts layer, refactor imports | 10h | v0.10.1 | ✅ RECOMMENDED |
| **C: Lazy Imports** | Use TYPE_CHECKING blocks | 12h | v0.10.1 | 🟡 ALTERNATIVE |
| **D: Coordinator** | Create coordinator facade (major refactor) | 50h | v0.11.0+ | 🟢 OPTIONAL |

---

## The Recommendation

**Hybrid Approach** (Phase 1 + Phase 2):

1. **Phase 1 (v0.10.0)**: Option A (2h)
   - Update linter allowlist
   - Document violations as intentional
   - Wire into CI
   - Unblock v0.10.0 release

2. **Phase 2 (v0.10.1)**: Option B (10h)
   - Create `core/contracts.py`
   - Migrate imports
   - Cleaner architecture

3. **Phase 3 (v0.11.0+)**: Option D (optional, 50h)
   - Only if multi-distribution split is a goal
   - Coordinator pattern

---

## The Critical Decision Points

### ❓ Question 1: Are exceptions a "shared domain contract"?

**Context**: 57 violations are from all packages importing `core.exceptions` (ADR-002 unified exception taxonomy)

**Options**:
- ✅ YES (allow explicitly in linter)
- 🟡 MAYBE (move to contracts facade v0.10.1)
- ❌ NO (duplicate exceptions everywhere—not recommended)

**Recommendation**: **YES → move to contracts in Phase 2**

**Decision**: Yes. Move to contracts

---

### ❓ Question 2: Should orchestrator be a hub importing all siblings?

**Context**: 25 violations from `core.calibrated_explainer` importing calibration, plugins, cache, parallel, explanations

**Options**:
- ✅ YES (current hub pattern, document as intentional)
- 🟡 MAYBE (migrate to coordinator pattern in v0.11.0)
- ❌ NO (too much coupling—but then how to coordinate?)

**Recommendation**: **YES for now → revisit in v0.11.0 if packaging goals require split**

**Decision**: `core.calibrated_explainer` is a thin delegator and the main entrance point. 

---

### ❓ Question 3: Should siblings import core domain interfaces?

**Context**: 12 violations from siblings importing `core.explain.feature_task`, `core.prediction`, etc.

**Options**:
- ✅ YES (interfaces are shared contracts, allow)
- 🟡 MAYBE (move to contracts facade v0.10.1)
- ❌ NO (duplicate interfaces everywhere—not recommended)

**Recommendation**: **YES → move to contracts in Phase 2**

---

## Document Navigation

| I am a... | Read this | Time |
|:---|:---|:---:|
| **Maintainer/Decision-maker** | ADR-001-DECISION-BRIEF.md | 15 min |
| **Developer (Phase 1)** | ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md | 2.5h |
| **Architect** | ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md | 45 min |
| **New contributor** | ADR-001-VIOLATIONS-VISUAL-SUMMARY.md | 20 min |
| **Everyone (start here)** | README-ADR-001-ANALYSIS.md | 10 min |

---

## Implementation Timeline

```
NOW (v0.10.0)    →    v0.10.1    →    v0.11.0+
     Phase 1         Phase 2         Phase 3 (opt)
    (2 hours)      (10 hours)       (50 hours)
      ✅            ✅ RECOMMENDED    🟢 OPTIONAL
   Allow-List    Contracts Layer   Coordinator
   Linter ready  Cleaner bounds    Max agility
```

---

## Success Criteria

### Phase 1 ✅
- [ ] Linter passes (`python scripts/check_import_graph.py` → 0 violations)
- [ ] CI enforces linter
- [ ] Documentation created
- [ ] v0.10.0 ships

### Phase 2 ✅
- [ ] `core/contracts.py` created
- [ ] Imports migrated (~15 files)
- [ ] Tests pass
- [ ] v0.10.1 ships

### Phase 3 ✅ (if approved)
- [ ] Coordinator implemented
- [ ] All cross-package calls routed through it

---

## What Maintainers Need to Do

1. **Read** (15 min): ADR-001-DECISION-BRIEF.md
2. **Vote** on 3 questions
3. **Approve** Phase 1 (yes/no/defer)
4. **Assign** developer if yes (2-hour task)

---

## What Developer Needs to Do (Phase 1)

1. **Read** (30 min): ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md
2. **Follow** Steps 1-7 (2 hours)
3. **Test** linter locally + in CI
4. **Submit** PR

---

## Key Insights

1. **Violations are intentional**, not bugs
2. **Exception taxonomy needs to be shared** (ADR-002)
3. **Orchestrator hub pattern is by design**
4. **Fast win available**: Phase 1 (2h) unblocks v0.10.0
5. **Architecture improvement available**: Phase 2 (10h) improves boundaries

---

## Next Actions

### RIGHT NOW
- [ ] Maintainers open: ADR-001-DECISION-BRIEF.md
- [ ] Discuss 3 decision points with team
- [ ] Vote on Phases

### NEXT SPRINT (If Phase 1 Approved)
- [ ] Assign developer
- [ ] Allocate 2 hours
- [ ] Target: Unblock v0.10.0

### AFTER v0.10.0 (If Phase 2 Approved)
- [ ] Schedule 10-12 hour task
- [ ] Implement contracts layer
- [ ] Merge before v0.10.1

---

## Files to Reference

**Analysis**:
- improvement_docs/README-ADR-001-ANALYSIS.md (master index)
- improvement_docs/ADR-001-DECISION-BRIEF.md (executive summary)
- improvement_docs/ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md (detailed)

**Implementation**:
- improvement_docs/ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md (step-by-step)
- scripts/check_import_graph.py (linter to update)

**Reference**:
- improvement_docs/ADR-001-VIOLATIONS-VISUAL-SUMMARY.md (heatmaps)
- violations.txt (raw violations list)

---

## Bottom Line

✅ **Analysis complete**  
✅ **7 strategic documents ready**  
✅ **Clear recommendations provided**  
✅ **2-hour fast track available (Phase 1)**  
✅ **Ready for maintainer decision**

**Recommendation**: Approve Phase 1 + 2 for v0.10.0–v0.10.1.

---

**Start here**: Read ADR-001-DECISION-BRIEF.md (15 min) to decide
