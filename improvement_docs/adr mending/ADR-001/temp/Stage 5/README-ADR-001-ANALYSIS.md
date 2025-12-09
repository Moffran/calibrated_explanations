# ADR-001 Import Boundary Gap Analysis: Master Index

**Analysis Date**: 2025-11-30
**Status**: Complete & Ready for Review
**Violations Found**: 153 cross-sibling imports
**Strategic Options**: 4 (A: Allow-List, B: Contracts, C: Lazy Imports, D: Coordinator)
**Recommended Approach**: Hybrid (Phase 1 + Phase 2)

---

## 📑 Complete Document Set

All analysis documents have been generated and are ready for review. Below is the master index showing what each document contains and how to use them.

### **5 Strategic Documents Created**

```
improvement_docs/
├── ADR-001-COMPLETE-ANALYSIS-PACKAGE.md          [12 KB] ⭐ READ FIRST
├── ADR-001-DECISION-BRIEF.md                    [14 KB] ⭐ FOR MAINTAINERS
├── ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md [25 KB] 📊 COMPREHENSIVE
├── ADR-001-VIOLATIONS-VISUAL-SUMMARY.md         [15 KB] 🎨 VISUAL GUIDE
└── ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md      [22 KB] 🔧 HANDS-ON
```

---

## 📖 Document-by-Document Breakdown

### 1. **ADR-001-COMPLETE-ANALYSIS-PACKAGE.md** (12 KB)

**Purpose**: Master index and reading guide for all documents

**Best for**: First-time readers, orienting yourself
**Read time**: 10 min
**Key content**:
   - Document guide (what to read when)
   - Decision flow diagram
   - Timeline summary
   - Quick start by role (maintainer, developer, architect)
   - Success criteria for each phase

**Recommended next**: Choose your role, follow "Quick Start" section

---

### 2. **ADR-001-DECISION-BRIEF.md** (14 KB)

**Purpose**: Executive summary for decision-makers

**Best for**: Maintainers, steering committee, quick decisions
**Read time**: 15 min
**Key content**:
   - The Situation (what violations mean)
   - Critical Decision Points (3 architectural questions)
   - Risk Analysis
   - Recommended Approach (Hybrid Phase 1 + 2)
   - Decision Matrix & Voting Recommendations

**Structure**:
- What's the problem? → Why it matters → What are our options? → Recommended vote
- Clear yes/no/defer questions that maintainers need to answer

**Recommended next**: If YES to Phase 1, → ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md

---

### 3. **ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md** (25 KB)

**Purpose**: Deep technical analysis of all strategic options

**Best for**: Architects, technical leads, detailed understanding
**Read time**: 45 min
**Key content**:
   - Part 1: Violation Patterns (6 patterns analyzed)
   - Part 2: Strategic Options (A/B/C/D with pros/cons/effort)
   - Part 3: Recommended Hybrid Approach
   - Part 4: Implementation Roadmap
   - Part 5: Decision Matrix
   - Appendix: Detailed Violation Breakdown

**Structure**:
- Root cause analysis → Strategic options → Hybrid recommendation → Phased roadmap

**Contains**:
- Code examples for each option
- Effort estimates (2h, 10h, 12h, 50h)
- Trade-off analysis
- Implementation details

**Recommended next**: If approving Phase 2, share this with team for architecture discussion

---

### 4. **ADR-001-VIOLATIONS-VISUAL-SUMMARY.md** (15 KB)

**Purpose**: Visual and quick-reference guide to violations

**Best for**: All developers (onboarding-friendly, visual learners)
**Read time**: 20 min
**Key content**:
   - Violation Heatmap (by package pair)
   - Current State vs. Desired State Architecture Diagrams
   - Top 10 Problem Files
   - 6 Violation Patterns with root causes
   - Decision Framework
   - Implementation Timeline
   - Glossary

**Structure**:
- Heatmaps → Architecture diagrams → Problem ranking → Quick start

**Visual elements**:
```
┌─────────────────────┐
│  core (orchestra)   │
├─────────────────────┤
│ ↙ ↙ ↙ ↙ ↙ ↙ ↙ ↙     │
└─────────────────────┘
   │   │   │   │   │
   ▼   ▼   ▼   ▼   ▼
[cal][exp][cch][plg][viz]
```

**Recommended next**: Use as reference when coding; share with new contributors

---

### 5. **ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md** (22 KB)

**Purpose**: Step-by-step walkthrough for Phase 1 (Option A)

**Best for**: Developers implementing Phase 1
**Read time**: 30 min + 2 hours implementation
**Key content**:
   - Overview (what we're doing)
   - Step 1: Update allowlist (30 min)
   - Step 2: Add documentation (20 min)
   - Step 3: Test linter (20 min)
   - Step 4: Create rationale docs (40 min)
   - Step 5: Wire linter into CI (30 min)
   - Step 6: Update documentation (20 min)
   - Step 7: Verify & test (20 min)
   - Checklist & troubleshooting

**Structure**:
- Overview → 7 implementation steps → Verification → Checklist

**Copy-paste ready**:
- Exact code snippets for linter updates
- YAML templates for CI workflow
- Markdown template for documentation
- Shell commands for testing

**Recommended next**: Follow steps 1–7 sequentially for Phase 1 implementation

---

## 🎯 Which Document to Read When

### **I'm a Maintainer/Steering Committee Member**

**Goal**: Decide whether to approve Phase 1, 2, and/or 3

**Read this sequence**:
1. **ADR-001-COMPLETE-ANALYSIS-PACKAGE.md** (2 min: Decision Flow section)
2. **ADR-001-DECISION-BRIEF.md** (15 min: full read)
3. **Optionally**: ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md (if deep dive needed)

**Action**: Vote on 3 decisions → Assign developer if Phase 1 approved

**Time**: 20–30 min for decision, optional deep dive 45 min more

---

### **I'm a Developer (Implementing Phase 1)**

**Goal**: Update linter + wire into CI

**Read this sequence**:
1. **ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md** (full read, 30 min)
2. **Reference**: ADR-001-VIOLATIONS-VISUAL-SUMMARY.md (if you want context)
3. **Follow**: Step 1–7 checklist (2 hours implementation)

**Deliverables**: Updated linter, CI workflow, documentation, tests passing

**Time**: 30 min reading + 2 hours coding + 30 min testing = ~3 hours total

---

### **I'm a Tech Lead/Architect**

**Goal**: Understand architecture implications of each option

**Read this sequence**:
1. **ADR-001-COMPLETE-ANALYSIS-PACKAGE.md** (10 min: overview)
2. **ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md** (45 min: full deep dive)
3. **Reference**: ADR-001-VIOLATIONS-VISUAL-SUMMARY.md (architecture diagrams)

**Action**: Provide recommendation to maintainers (Phase 1+2 recommended)

**Time**: 60 min deep read

---

### **I'm a New Contributor**

**Goal**: Understand why some cross-sibling imports are allowed

**Read this sequence**:
1. **ADR-001-VIOLATIONS-VISUAL-SUMMARY.md** (20 min: visual overview)
2. **ADR-001-EXCEPTIONS-AND-CONTRACTS.md** (will be created in Phase 1, 20 min)

**Action**: Reference when writing code that touches multiple packages

**Time**: 40 min (one-time onboarding)

---

## 🚀 Quick Start: 3 Paths Forward

### Path 1: Fast Decision (No Deep Dive)
```
You have: 20 minutes
→ Read: ADR-001-DECISION-BRIEF.md
→ Vote on 3 questions
→ Done
```

### Path 2: Informed Decision (Medium Dive)
```
You have: 60 minutes
→ Read: ADR-001-COMPLETE-ANALYSIS-PACKAGE.md (10 min)
→ Read: ADR-001-DECISION-BRIEF.md (15 min)
→ Read: ADR-001-VIOLATIONS-VISUAL-SUMMARY.md (20 min)
→ Skim: ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md (15 min)
→ Vote
```

### Path 3: Deep Understanding (Full Dive)
```
You have: 2-3 hours
→ Read: ADR-001-COMPLETE-ANALYSIS-PACKAGE.md (10 min)
→ Read: ADR-001-DECISION-BRIEF.md (15 min)
→ Read: ADR-001-VIOLATIONS-VISUAL-SUMMARY.md (20 min)
→ Read: ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md (45 min)
→ Read: ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md (30 min)
→ Deep questions answered; ready to lead implementation
```

---

## 📊 Analysis Summary (By the Numbers)

| Metric | Value | Reference |
|:---|---:|:---|
| **Violations Found** | 153 | VIOLATIONS-VISUAL-SUMMARY.md |
| **Top Violator File** | core/calibrated_explainer.py (25 violations) | CROSS-SIBLING-REFACTORING-OPTIONS.md |
| **Top Violation Pattern** | core.exceptions imports (57) | CROSS-SIBLING-REFACTORING-OPTIONS.md |
| **Strategic Options** | 4 (A, B, C, D) | CROSS-SIBLING-REFACTORING-OPTIONS.md |
| **Recommended Option** | Hybrid Phase 1 + Phase 2 | DECISION-BRIEF.md |
| **Phase 1 Effort** | 2 hours | PHASE-1-IMPLEMENTATION-GUIDE.md |
| **Phase 2 Effort** | 10–12 hours | CROSS-SIBLING-REFACTORING-OPTIONS.md |
| **Phase 3 Effort** | 40–50 hours (optional) | CROSS-SIBLING-REFACTORING-OPTIONS.md |

---

## 🎓 Key Takeaways

1. **Violations Are Intentional**: Not bugs; they represent documented architectural patterns (orchestrator hub, shared exception taxonomy, domain interfaces).

2. **Fast Win Available**: Phase 1 (2 hours) allow-lists violations and enables CI enforcement for v0.10.0.

3. **Architecture Improvement Opportunity**: Phase 2 (10–12 hours) creates cleaner boundaries via `core/contracts.py` for v0.10.1.

4. **Long-Term Option**: Phase 3 (40–50 hours, optional) coordinator pattern enables future multi-distribution split.

5. **Recommended Path**: Phase 1 + Phase 2 balances speed and architecture quality.

---

## ✅ Next Steps

### For Maintainers

- [ ] Review **ADR-001-DECISION-BRIEF.md** (15 min)
- [ ] Discuss 3 critical decision points with team
- [ ] Vote: Approve Phase 1? Phase 2? Defer Phase 3?
- [ ] If YES Phase 1: Assign developer + schedule 2 hours

### For Developers (If Phase 1 Approved)

- [ ] Read **ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md** (30 min)
- [ ] Follow steps 1–7 (2 hours)
- [ ] Test linter locally + in CI
- [ ] Submit PR with linter changes + docs

### For All Team Members

- [ ] Bookmark **ADR-001-VIOLATIONS-VISUAL-SUMMARY.md** for future reference
- [ ] Await Phase 1 completion + CI integration
- [ ] Follow linter rules when making PRs (block violations in CI)

---

## 🔗 Supporting Resources

**Related ADRs**:
- ADR-001: Core Decomposition Boundaries (defines packages)
- ADR-002: Exception Taxonomy (defines exceptions; why they're shared)
- ADR-003: Caching Strategy (cross-package integration)
- ADR-004: Parallel Execution (orchestrator coordination)
- ADR-006: Plugin Trust Model (coming v0.10.2; will formalize plugin interface)

**Related Documents**:
- RELEASE_PLAN_V1.md: Stage 5 status + v0.10.0–v0.11.0 roadmap
- improvement_docs/adr\ mending/ADR-001/: Previous stage completion reports
- scripts/check_import_graph.py: Linter implementation
- tests/unit/test_import_graph_enforcement.py: Enforcement tests

**Related Files** (for reference during Phase 1):
- violations.txt: Raw linter output (shows all 153 violations)
- scripts/check_import_graph.py: Linter to update
- .github/workflows/lint.yml: CI workflow to create/update

---

## 📝 Document Versioning

| Version | Date | Status | Changes |
|:---|:---|:---|:---|
| 1.0 | 2025-11-30 | ✅ COMPLETE | Initial analysis: 4 options, hybrid recommendation |
| 1.1 | TBD | ⏳ PENDING | Updates after maintainer vote (add decision rationale) |
| 2.0 | TBD | ⏳ PENDING | Phase 1 completion report (linter enforced, CI integrated) |
| 2.1 | TBD | ⏳ PENDING | Phase 2 completion report (contracts layer created) |
| 3.0 | TBD | ⏳ PENDING | Phase 3 completion report (coordinator implemented, if approved) |

---

## 📞 Questions?

**On this analysis**:
- Deep questions: See ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md (Part 5: Decision Matrix)
- Quick questions: See ADR-001-DECISION-BRIEF.md (Critical Decision Points)

**On Phase 1 implementation**:
- Step-by-step help: See ADR-001-PHASE-1-IMPLEMENTATION-GUIDE.md (Step 7: Common Issues & Fixes)
- Troubleshooting: See same document, "Common Issues & Fixes" section

**For architectural direction**:
- Contact: Maintainers (see CONTRIBUTING.md)
- Discuss: Using ADR-001-DECISION-BRIEF.md as talking points

---

## 🎉 Final Status

**Analysis**: ✅ COMPLETE
**Documentation**: ✅ COMPLETE (5 documents, ~90 KB)
**Ready for**: ✅ Maintainer Review & Decision

**Expected outcomes**:
- Phase 1 (v0.10.0): Linter enforced, CI-ready, documented
- Phase 2 (v0.10.1): Cleaner boundaries via contracts layer
- Phase 3 (v0.11.0+): Long-term modularity (if needed)

**Recommended action**: Begin with ADR-001-DECISION-BRIEF.md for maintainers to vote.

---

**Master Index Created**: 2025-11-30
**Analysis Status**: Complete & Ready for Implementation
**Next Phase**: Maintainer Decision → Phase 1 Implementation → v0.10.0 Release
