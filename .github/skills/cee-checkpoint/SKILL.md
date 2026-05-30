---
name: cee-checkpoint
description: >
  Implement CEE checkpoint and rollback patterns using PyArrow/Parquet and MLflow, ensuring deterministic state persistence and safe recovery from drift events.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Cee Checkpoint — Core Instructions

# CEE Checkpoint & Rollback

## Use this skill when
- Implementing or modifying checkpoint save/load logic
- Adding MLflow integration for checkpoint artifacts
- Implementing rollback on drift detection
- Debugging checkpoint corruption or round-trip failures
- Writing tests for checkpoint persistence

## Inputs
- `packages/adaptive/src/.../semi_online/window.py` — SlidingWindowBuffer checkpoint methods
- `packages/common/src/.../persistence/` — checkpoint interfaces
- `development/strategic-pillars/03_checkpoints_and_rollback.md` — design rationale

## Architecture

```
CalibratedAdaptiveExplainer.save_state(path)
    └── SemiOnlineManager → SlidingWindowBuffer.write_checkpoint(directory)
           └── PyArrow → <uuid>.parquet

MLflowCheckpointManager  (higher-level artifact store)
    └── mlflow.log_artifact(parquet_path, artifact_path)
```

## Checkpoint Types

| Type | Implementation | Storage | Use Case |
|---|---|---|---|
| Window snapshot | `SlidingWindowBuffer.write_checkpoint()` | Local Parquet file | Recovery from drift |
| State save | `CalibratedAdaptiveExplainer.save_state()` | Path on disk | Session persistence |
| MLflow artifact | `MLflowCheckpointManager.log_checkpoint()` | MLflow + S3/GCS | Production audit trail |

## Key Invariants

1. **Checkpoints are never committed to git** — `.parquet`, `.pkl`, `.db` are in `.gitignore`
2. **Round-trip safety** — `write_checkpoint()` then `load_checkpoint()` must restore identical state
3. **UUID-based naming** — checkpoint files use `uuid.uuid4().hex[:8]` to avoid collisions
4. **No partial writes** — write to temp file then rename atomically
5. **Schema versioning** — Parquet metadata must include schema version for forward compatibility

## Workflow

### Implementing a new checkpoint strategy

1. Read existing `SlidingWindowBuffer` checkpoint code
2. Implement `write_checkpoint(directory: Path) -> CheckpointArtifact`:
   ```python
   def write_checkpoint(self, directory: Path) -> CheckpointArtifact:
       table = pa.Table.from_pylist(self._records)
       path = directory / f"semi_online_window_{uuid.uuid4().hex[:8]}.parquet"
       pq.write_table(table, path)
       return CheckpointArtifact(path=path, num_rows=len(self._records))
   ```
3. Implement `load_checkpoint(path: Path) -> int`:
   ```python
   def load_checkpoint(self, path: Path) -> int:
       table = pq.read_table(path)
       self._records = table.to_pylist()
       return len(self._records)
   ```
4. Add round-trip test (see Test Patterns below)

### Implementing drift-triggered rollback

```python
# In SemiOnlineManager.process_batch():
if drift_result.triggered and policy.remediation == "FALLBACK_LAST_CHECKPOINT":
    if self._last_checkpoint_path:
        self._buffer.load_checkpoint(self._last_checkpoint_path)
        # Re-calibrate from restored window
        self._recalibrate_from_window()
    else:
        logger.warning("Drift detected but no checkpoint available for rollback")
```

### MLflow integration

```python
import mlflow

class MLflowCheckpointManager:
    def log_checkpoint(self, artifact: CheckpointArtifact, run_id: str) -> str:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(artifact.path), artifact_path="checkpoints")
        return mlflow.get_artifact_uri("checkpoints")

    def load_checkpoint(self, run_id: str, artifact_name: str) -> Path:
        artifact_uri = f"runs:/{run_id}/checkpoints/{artifact_name}"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)
        return Path(local_path)
```

## Test Patterns

```python
def test_checkpoint_round_trip_preserves_all_records():
    """Write checkpoint, load it back, verify identical state."""
    buffer = SlidingWindowBuffer(size=100)
    records = [{"features": [i, i+1], "target": i % 2} for i in range(50)]
    buffer.extend(records)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = buffer.write_checkpoint(Path(tmpdir))

        new_buffer = SlidingWindowBuffer(size=100)
        loaded_count = new_buffer.load_checkpoint(artifact.path)

    assert loaded_count == 50
    assert new_buffer.to_list() == buffer.to_list()  # exact equality

def test_checkpoint_file_not_committed_to_git():
    """Verify .parquet files are in .gitignore."""
    gitignore = Path(".gitignore").read_text()
    assert "*.parquet" in gitignore
```

## Verification
```bash
pytest packages/adaptive/tests/ -k checkpoint -q
pytest tests/integration/ -k checkpoint -m integration -q
# Verify no runtime artifacts are staged:
git status | grep -E "\.(parquet|pkl|db)$" && echo "FAIL: runtime artifacts staged" || echo "PASS"
```

## Output contract
For implementation tasks, return:
1. Checkpoint implementation with write + load methods
2. Round-trip safety test
3. MLflow integration if production persistence required
4. Confirmation that checkpoint files are in `.gitignore`

## Constraints
- Checkpoint files must NEVER be committed to git
- Round-trip must be exact: `load(write(state)) == state`
- Use PyArrow/Parquet as the serialization format (not pickle, not JSON)
- MLflow integration goes in `common.persistence` (not in `adaptive` or `governance`)