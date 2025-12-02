# ADR-001 Gap Analysis: Complete Documentation Package

**Generated**: 2025-11-30  
**Status**: Analysis Complete, Ready for Implementation  
**Audience**: Maintainers, Steering Committee, Developers

---

## 📋 Documentation Overview

This analysis package provides a complete assessment of ADR-001 Stage 5 import boundary enforcement gaps and strategic options for closure. Below is a map of all documents and how to use them.

---

## 📚 Document Guide

### 1. **ADR-001-DECISION-BRIEF.md** ⭐ START HERE
   - **Length**: 15 min read
   - **For**: Maintainers & steering committee
   - **What**: Executive summary, decision matrix, voting recommendations
   - **Key sections**:
     - The Situation (153 violations discovered)
     - Critical Decision Points (3 architectural questions)
     - Recommended Approach (Hybrid Phase 1 + Phase 2)
     - Risk Analysis & Next Steps
   - **Takeaway**: Decide whether to approve Phase 1, 2, and/or 3

---

### 2. **ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md** 📊 COMPREHENSIVE
   - **Length**: 45 min deep dive
   - **For**: Architects, senior developers
   - **What**: Detailed analysis of 4 strategic options (A, B, C, D)
   - **Key sections**:
     - Part 1: Violation Pattern Analysis (6 patterns identified)
     - Part 2: Strategic Refactoring Options (pros/cons/effort for each)
     - Part 3: Recommended Hybrid Approach (Phase 1 + 2 + optional 3)
     - Part 4: Implementation Roadmap
     - Part 5: Decision Matrix
   - **Takeaway**: Understand trade-offs between quick wins vs. long-term architecture

---

### 3. **ADR-001-VIOLATIONS-VISUAL-SUMMARY.md** 🎨 VISUAL
   - **Length**: 20 min skim
   - **For**: All developers (onboarding-friendly)
   - **What**: Heatmaps, architecture diagrams, pattern summaries
   - **Key sections**:
     - Violation Heatmap (by package pair)
     - Architecture Map (current vs. desired state)
     - Top 10 Problem Files
     - 6 Violation Patterns with root causes
     - Quick-start guide
   - **Takeaway**: Quick visual understanding of the landscape

---

### 4. **ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md** 🔧 HANDS-ON
   - **Length**: 30 min implementation
   - **For**: Developers implementing Phase 1
   - **What**: Step-by-step walkthrough to update linter & wire into CI
   - **Key sections**:
     - Step 1: Update allowlist in `scripts/check_import_graph.py`
     - Step 2: Add documentation comments
     - Step 3: Test linter locally
     - Step 4: Create rationale documentation
     - Step 5: Wire linter into CI
     - Step 6: Update CHANGELOG
     - Step 7: Verify & test
   - **Takeaway**: Copy-paste implementation; 2-hour task

---

### 5. **ADR-001-EXCEPTIONS-AND-CONTRACTS.md** (will be created in Phase 1)
   - **Length**: 20 min read
   - **For**: All developers
   - **What**: Rationale for each allowed cross-sibling import + migration timeline
   - **Key sections**:
     - Shared Domain Contracts (why exceptions, orchestrators, interfaces are allowed)
     - Non-Allowed Imports (examples of violations to avoid)
     - Enforcement & Testing
     - Transition Schedule (v0.10.0 → v0.10.1 → v0.10.2 → v0.11.0+)
   - **Takeaway**: "Why can X import Y?" answered clearly for every rule

---

## 🎯 Decision Flow

```
START: Read ADR-001-DECISION-BRIEF.md
│
├─ Question 1: Ship v0.10.0 with allow-listed violations?
│  ├─ YES → Approve Phase 1
│  │        ↓
│  │        Go to: ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md
│  │        Timeline: 2 hours (implement now)
│  │
│  └─ NO → Stop. Halt v0.10.0. Need additional refactoring.
│
├─ Question 2: Clean up boundaries in v0.10.1 (Option B)?
│  ├─ YES → Schedule Phase 2
│  │        ↓
│  │        Timeline: 10–12 hours (schedule post-v0.10.0)
│  │        Deliverable: core/contracts.py + migrated imports
│  │
│  └─ NO → Stop. Remain with Phase 1 indefinitely (acceptable).
│
└─ Question 3: Pursue Option D (Coordinator) in v0.11.0?
   ├─ YES → Reserve v0.11.0 capacity
   │        Timeline: 40–50 hours (only if multi-distribution packaging needed)
   │
   └─ NO → Stop. Phase 2 is final target state.
```

---

## ⏱️ Timeline Summary

| Phase | When | Effort | Outcome |
|:---|:---|:---:|:---|
| **Phase 1** (Option A: Allow-list) | v0.10.0 NOW | 2h | Linter enforced, CI-ready |
| **Phase 2** (Option B: Contracts) | v0.10.1 | 10–12h | Cleaner boundaries |
| **Phase 3** (Option D: Coordinator) | v0.11.0+ | 40–50h | Multi-distrib ready (optional) |

---

## 📊 Key Metrics

| Metric | Value |
|:---|:---:|
| Total violations found | 153 |
| Violations by source (top 3) | core (76), calibration (26), explanations (15) |
| Violations by target (top 3) | core.exceptions (57), core.calibrated_expl (19), others (77) |
| Violation patterns | 6 identified |
| Top problem file | core/calibrated_explainer.py (25 violations) |
| Phase 1 effort | 2 hours |
| Phase 2 effort | 10–12 hours |
| Phase 1 → v0.10.0 unblock | ✅ YES |
| Phase 2 → architecture improvement | ✅ YES |
| Phase 3 → future agility | 🟡 OPTIONAL |

---

## 🚀 Quick Start for Different Roles

### 👔 **Maintainer/Steering Committee**

1. Read: **ADR-001-DECISION-BRIEF.md** (15 min)
2. Vote on: Approve Phase 1? Approve Phase 2? Defer Phase 3?
3. If YES to Phase 1: Assign developer + schedule 2 hours this iteration
4. If YES to Phase 2: Schedule post-v0.10.0

### 👨‍💻 **Developer (Implementing Phase 1)**

1. Read: **ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md** (10 min)
2. Follow: Step 1–7 checklist (2 hours)
3. Reference: **ADR-001-EXCEPTIONS-AND-CONTRACTS.md** for allowlist rationale
4. Test: Run linter locally, verify CI integration

### 🏗️ **Architect/Technical Lead**

1. Read: **ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md** (45 min)
2. Understand: 4 strategic options + trade-offs
3. Provide: Architecture recommendation (Phase 1 + 2, or Phase 1 only, or custom approach)
4. Reference: **ADR-001-VIOLATIONS-VISUAL-SUMMARY.md** for root causes

### 📚 **New Contributor**

1. Skim: **ADR-001-VIOLATIONS-VISUAL-SUMMARY.md** (20 min)
2. Read: **ADR-001-EXCEPTIONS-AND-CONTRACTS.md** (once created in Phase 1)
3. Understand: Which cross-sibling imports are allowed + why
4. Reference: When writing code that touches multiple packages

---

## 🎓 Key Insights

### Insight 1: Violations Are Intentional, Not Bugs

The 153 violations represent **documented architectural patterns**, not code defects. Examples:

- Exception taxonomy (ADR-002) is centralized in `core.exceptions` → all packages must import it.
- Orchestrator hub (`CalibratedExplainer`) coordinates all subsystems → must import calibration, plugins, cache, etc.
- Domain interfaces (feature tasks, strategies) are core abstractions → siblings need type hints.

**Action**: Document these patterns explicitly in the linter config (Phase 1).

---

### Insight 2: Exception Taxonomy Is a Shared Contract

ADR-002 unified exception types across all code. This is intentional—every package uses the same exceptions for validation, configuration, state, and runtime errors.

**Current**: 57 imports of `core.exceptions` from siblings.  
**Phase 1**: Allow explicitly.  
**Phase 2**: Move to `core/contracts.py` re-export (cleaner boundary).

---

### Insight 3: Orchestrator Hub Pattern Is Dominant

The wrapper explainer (`CalibratedExplainer`) is the coordination center. It imports from calibration, cache, plugins, parallel—not because of poor design, but because it *orchestrates* these subsystems.

**Current**: 25 violations (mostly in explainer importing siblings).  
**Future (v0.11.0+)**: Coordinator pattern could mediate these calls if multi-distribution is a goal. But it's optional.

---

### Insight 4: Fast Wins vs. Long-Term Architecture

| Approach | Speed | Architecture Quality |
|:---|:---:|:---:|
| Phase 1 (allow-list) | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Phase 1 + 2 (contracts) | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| All phases (coordinator) | ⭐ | ⭐⭐⭐⭐⭐ |

**Recommendation**: Phase 1 + 2 is the sweet spot—fast enough for v0.10.0/v0.10.1, clean enough for maintainability.

---

## 📝 Next Steps

### Immediate (Today/Tomorrow)

- [ ] Maintainers review **ADR-001-DECISION-BRIEF.md**
- [ ] Vote on Phase 1, 2, 3 approvals
- [ ] Assign developer if Phase 1 approved

### If Phase 1 Approved (Next 2 hours)

- [ ] Developer implements Phase 1 following **ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md**
- [ ] Linter tested locally and in CI
- [ ] PR submitted with linter changes + documentation
- [ ] CHANGELOG updated

### If Phase 2 Approved (Post-v0.10.0 release)

- [ ] Schedule 10–12 hour task
- [ ] Create `core/contracts.py`
- [ ] Migrate imports across 15 files
- [ ] Update linter config
- [ ] Merge before v0.10.1 release

### If Phase 3 Deferred (Reassess v0.11.0)

- [ ] Document decision (packaging goals, distribution strategy)
- [ ] Revisit in v0.11.0 planning if multi-distribution split is on roadmap

---

## 📖 Complete Reading List (by depth)

**Shallow (5–15 min):**
1. ADR-001-DECISION-BRIEF.md (overview + decisions)

**Medium (20–30 min):**
2. ADR-001-VIOLATIONS-VISUAL-SUMMARY.md (visual + quick reference)
3. ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md (if implementing)

**Deep (45–60 min):**
4. ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md (all options + rationale)
5. ADR-001-EXCEPTIONS-AND-CONTRACTS.md (once created; detailed contract rules)

**Reference (as needed):**
- `violations.txt` (raw linter output)
- `scripts/check_import_graph.py` (linter implementation)

---

## ✅ Success Criteria

### Phase 1 Success
- [ ] Linter passes with 0 violations (`python scripts/check_import_graph.py`)
- [ ] CI enforces linter (blocks PRs with violations)
- [ ] Documentation created + reviewed
- [ ] CHANGELOG updated
- [ ] v0.10.0 ships with linting infrastructure

### Phase 2 Success (if approved)
- [ ] `core/contracts.py` created with exception re-exports + protocols
- [ ] Imports migrated in 15 files
- [ ] Linter updated for contracts imports
- [ ] Tests pass + coverage maintained
- [ ] v0.10.1 ships with cleaner boundaries

### Phase 3 Success (if pursued v0.11.0+)
- [ ] Coordinator pattern designed + documented
- [ ] All cross-package calls routed through coordinator
- [ ] Packages could theoretically split into separate distributions
- [ ] v0.11.0+ ships with enhanced modularity

---

## 🔗 Cross-References

- **ADR-001**: Core Decomposition Boundaries (defines top-level packages)
- **ADR-002**: Exception Taxonomy (defines unified exceptions; see why it's a shared contract)
- **ADR-003**: Caching Strategy (includes cross-package integration)
- **ADR-004**: Parallel Execution (includes orchestrator coordination)
- **ADR-006**: Plugin Trust Model (coming v0.10.2; will formalize plugin interface)
- **RELEASE_PLAN_V1.md**: Tracks Stage 5 status and v0.10.0–v0.11.0 roadmap

---

## 📞 Questions & Feedback

**For clarification on analysis**:
- Review ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md (Part 5: Decision Matrix)

**For implementation questions**:
- Consult ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md (Step 7: Common Issues & Fixes)

**For architectural direction**:
- Engage with maintainers using ADR-001-DECISION-BRIEF.md (Critical Decision Points)

---

## 📅 Document Versioning

| Version | Date | Status | Notes |
|:---|:---|:---|:---|
| 1.0 | 2025-11-30 | ✅ COMPLETE | Initial analysis, 4 options, recommendations |
| — | — | — | Updates after maintainer decision |

---

**Status**: Analysis Complete. Awaiting Maintainer Decision.

**Next Actions**: Review DECISION-BRIEF.md, vote on phases, begin Phase 1 implementation.
