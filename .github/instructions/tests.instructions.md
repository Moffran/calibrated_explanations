---
applyTo:
  - "tests/**"
  - "**/*test*.{py,ts,tsx,js,java,cs}"
  - "**/*.spec.{ts,tsx,js}"
priority: 100
---


**Framework:** pytest (+pytest-mock if already present). Do not introduce new libraries.

**Where to put tests**
- Unit → `tests/unit/<package>/test_<module>.py` (append to existing file if present)
- Integration → `tests/integration/<feature>/test_<feature>.py`

**File creation gate**
- Create new file only if: no suitable file exists **and** post-change size would exceed ~400 lines/50 cases **or** scope differs (unit vs integration).

**Style**
- Use `pytest` style functions; avoid `unittest.TestCase` unless already used in the target file.
- Name tests: `test_<func>__should_<behavior>_when_<condition>`.
- Prefer `@pytest.mark.parametrize` over loops.
- Use `freezegun`/time-freeze patterns if present; otherwise stub the clock.
- No I/O, env, or network in unit tests—use monkeypatch/mocks.
- Tests should be testing the behavior, not implementation or by calling private helper methods.

### ✅ Behavior vs. Implementation: Decision Tree

Before writing a test, answer these questions in order:

**Q1: What am I testing?**
- If: Internal mechanism (e.g., dict keys, private method calls)
  - ❌ RED FLAG: You're testing implementation, not behavior
- If: User-facing outcome (e.g., "data is saved," "model is calibrated")
  - ✅ PROCEED: You're testing behavior

**Q2: Would this test break if I refactored the implementation (keeping behavior unchanged)?**
- If: YES → You're testing implementation
  - ❌ REFACTOR: Change test to assert on behavior, not mechanism
- If: NO → You're testing behavior
  - ✅ KEEP: This test

**Q3: Can I state this test as a business rule or domain invariant?**
- If: NO → You're testing internal detail
  - ❌ CLARIFY: State the business rule first
- If: YES → You're testing behavior
  - ✅ PROCEED: Commit test

**Example: Good Test (Behavior)**
```python
def test_plot_spec_intervals_are_properly_bounded():
    """Verify that low <= predict <= high after deserialization."""
    original = PlotSpec(header=IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8))
    serialized = plotspec_to_dict(original)
    deserialized = plotspec_from_dict(serialized)
    
    # Semantic assertion: invariant holds
    assert deserialized.header.low <= deserialized.header.pred
    assert deserialized.header.pred <= deserialized.header.high
```

**Example: Bad Test (Implementation Detail)**
```python
def test_plotspec_to_dict_uses_specific_keys():
    """Verify dict has 'plotspec_version', 'title', etc."""
    spec = PlotSpec(...)
    d = plotspec_to_dict(spec)
    
    # Implementation assertion: specific keys
    assert "plotspec_version" in d  # ← Breaks if we rename
    # Refactor to assert on behavior, not dict structure
```

---

### ✅ Private Helpers: Guidelines

**Do not test private functions (`_helper`) directly** unless they implement a stable, reusable domain concept.

**Decision Tree for Private Helpers:**
- Is this a stable, reusable domain concept?
  - YES → Make it public (`_helper` → `helper`); test as public API
  - NO → Keep private; test through the public function that uses it
  
- Only one module uses this helper?
  - YES → Keep private; test through its caller (integration test if complex)
  - NO → Make public if multiple users depend on it

**Example: Refactor Private Helper to Public**
```python
# BEFORE: Testing _private_function directly (❌ BAD)
def test_private_helper():
    assert builders._looks_like_probability_values(0.5)

# AFTER: Make public, test through public API (✅ GOOD)
# src/calibrated_explanations/viz/builders.py
def is_valid_probability_values(*values: Any) -> bool:
    """Check if all values are valid probabilities in [0, 1]."""
    # Implementation (formerly _looks_like_probability_values)
    ...

# tests/unit/viz/test_builders_validation.py
def test_is_valid_probability_values__should_accept_valid_probabilities():
    from calibrated_explanations.viz.builders import is_valid_probability_values
    assert is_valid_probability_values(0.0, 0.5, 1.0)
```

**Example: Keep Private, Test Through Public API**
```python
# BEFORE: Testing _private_function directly (❌ BAD)
def test_color_brew():
    palette = builders._legacy_color_brew(4)
    assert len(palette) == 4

# AFTER: Test through public rendering function (✅ GOOD)
def test_regression_plot__should_use_distinct_colors():
    """Verify that rendered plots use visually distinct colors."""
    spec = build_regression_bars_spec(
        title="test",
        predict={"predict": 0.5, "low": 0.2, "high": 0.8},
        feature_weights={"predict": np.array([0.1, -0.2])}
    )
    fig_path = tmp_path / "plot.png"
    render(spec, save_path=str(fig_path), show=False)
    
    # Verify colors are distinct (through rendered output)
    assert fig_path.exists()
    # Optional: image analysis to verify distinct colors
```

---

### ✅ Mocking: Best Practices

**Mocks are for isolation (avoiding slow I/O, network, RNG), not for shadowing real contracts.**

**Mocking Checklist:**
- [ ] **Isolation:** Mocks avoid expensive I/O, network, or RNG (not to hide implementation)
- [ ] **Outcome Assertions:** Tests assert on *result*, not mock calls (no `mock.assert_called_with`)
- [ ] **Integration Pairing:** For every mock-heavy unit test, there's a real integration test
- [ ] **Readability:** Mock implementations are minimal and documented
- [ ] **Reusability:** Mocks are in fixtures or `conftest.py`, not copied

**Example: Mock-Heavy Unit Test (❌ Anti-Pattern)**
```python
def test_plugin_explain():
    # Mocks everything; only checks type
    plugin = DummyPlugin()
    assert isinstance(plugin, ExplanationPlugin)  # ← Type check only, no behavior
    # Never actually uses the plugin
```

**Example: Refactored - Pair Unit + Integration Tests (✅ GOOD)**
```python
# Unit test: Lightweight (still mocks, but checks behavior)
def test_explanation_plugin__should_implement_required_interface():
    plugin = DummyPlugin()
    ctx = ExplanationContext(...)
    plugin.initialize(ctx)  # Should not raise
    
    batch = plugin.explain_batch(...)
    assert isinstance(batch, ExplanationBatch)
    assert len(batch.instances) > 0  # ← Outcome assertion

# Integration test: Real system
def test_plugin_discovery_and_explain_workflow():
    """Verify complete plugin lifecycle with real implementations."""
    from real_plugins import RealPlugin
    plugin = RealPlugin()  # ← NOT a mock
    registry.register(plugin)
    
    found = registry.find_for(SomeModel())
    assert plugin in found
    
    # ... continue with real workflow ...
```

---

### ✅ Snapshots and Serialization

**Snapshots verify structure is stable; pair with semantic assertions to verify correctness.**

**Pattern:**
1. Create object
2. Serialize → Dict
3. Deserialize → Object
4. Assert equality (snapshot) ← Verifies structure
5. **ADD:** Assert invariants (semantics) ← Verifies correctness

**Example: Snapshot Only (❌ Anti-Pattern)**
```python
def test_plotspec_roundtrip():
    spec = PlotSpec(header=IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8), ...)
    d = plotspec_to_dict(spec)
    s2 = plotspec_from_dict(d)
    assert s2 == spec  # ← That's it! Doesn't verify correctness
```

**Example: Snapshot + Semantics (✅ GOOD)**
```python
def test_plotspec_roundtrip__should_preserve_serialization_and_semantics():
    """Verify roundtrip preserves both structure (snapshot) and semantics (invariants)."""
    original = PlotSpec(
        header=IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8),
        body=BarHPanelSpec(bars=[BarItem(label="f1", value=0.1)])
    )
    
    # Roundtrip
    d = plotspec_to_dict(original)
    restored = plotspec_from_dict(d)
    
    # Snapshot: structure preservation
    assert restored == original
    
    # Semantics: domain invariants
    assert restored.header.low <= restored.header.pred <= restored.header.high
    assert len(restored.body.bars) > 0
```

---

### ✅ Test Naming Convention (Extended)

**Pattern:**
```
test_<module/class>__should_<behavior>_when_<condition>
```

**Examples:**
- ✅ `test_calibrated_explainer__should_require_fit_before_explain()`
- ✅ `test_interval_calibrator__should_update_thresholds_when_calibrate_called()`
- ✅ `test_plotspec_roundtrip__should_preserve_interval_invariants()`
- ✅ `test_plugin_registry__should_discover_plugins_by_model_type()`

**Ambiguous Names → Clarify:**
- ⚠️ `test_explanation_plot()` → `test_explanation_plot__should_render_without_error()`
- ⚠️ `test_calibration()` → `test_calibration__should_reduce_uncertainty_on_held_out_data()`

**Bad Names (Red Flags):**
- ❌ `test_helpers()` — Too vague
- ❌ `test_implementation_detail()` — Signals anti-pattern; reconsider
- ❌ `test_private_function_X()` — Direct private testing; refactor

---

### ✅ Anti-Patterns to Avoid

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| **Direct private function tests** | Break when internals refactor; not user-facing | Test through public API or delete |
| **Dict key assertions** | Brittle to implementation changes | Assert on domain invariants instead |
| **Snapshot only (no semantics)** | Passes even if data is wrong | Add post-deserialization semantic checks |
| **Mock-heavy, spy assertions** | Unit passes; system breaks | Add integration tests; use outcome assertions |
| **Dataclass `FrozenInstanceError`** | Tied to Python implementation | Use generic `Exception`; test contract not mechanism |
| **Platform-specific path tests** | Break on Windows/POSIX | Use `pathlib.Path` abstractions |

---

### ✅ Integration Testing

Use this checklist to decide if a test belongs in integration suite:

- [ ] Does this test exercise multiple components together?
- [ ] Does it verify a user-facing workflow or contract?
- [ ] Would this catch bugs that unit tests miss?
- [ ] Does it require I/O, real models, or external dependencies?
- [ ] Is it too slow for unit test suite? (mark with `@pytest.mark.slow`)

**If YES to ≥3 → Add as integration test**

**Structure:**
```python
# tests/integration/<feature>/test_<feature>.py

@pytest.mark.integration
def test_<workflow>_with_<scenario>():
    """Comprehensive test of a user-facing workflow.
    
    Scenario: <Describe the user's intent>
    """
    # 1. Prepare data
    data = prepare_test_data()
    
    # 2. Setup components
    model = train_model(data)
    
    # 3. Execute workflow
    result = execute_workflow(model, data)
    
    # 4. Verify end-to-end correctness
    assert workflow_is_correct(result)
```

---

**Fixtures**
- Import shared fixtures from `conftest.py` or existing fixture modules; only create a new fixture file when SUT-specific and not reusable.

**Examples**
- If editing `pkg/module.py`, target `tests/unit/pkg/test_module.py`.
- If adding a new integration around an HTTP client, target `tests/integration/http/test_client.py`.

**You are generating or editing tests. Follow this policy strictly:**

1. Prefer amending existing files over creating new ones.
2. Only create a new file if **all** creation criteria in the repo policy are satisfied.
3. Use the correct directory and naming mapping for the detected language.
4. Reuse fixtures/helpers; do not duplicate them.
5. Keep diffs minimal: focus the change on the SUT under edit.
6. If a rule would be violated, output a short justification block for the PR and then proceed by **modifying an existing file instead**.


**Snapshots**
- Keep snapshots minimal and stable; prefer explicit assertions for logic.
