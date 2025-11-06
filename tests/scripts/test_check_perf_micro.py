import json
from pathlib import Path

from scripts import check_perf_micro


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_missing_explain_metrics_fails(tmp_path):
    baseline = {
        "import_time_seconds": 1.0,
        "classification": {
            "explain_factual_time_s": 0.1,
            "explore_alternatives_time_s": 0.2,
        },
        "regression": {
            "explain_factual_time_s": 0.1,
            "explore_alternatives_time_s": 0.2,
        },
    }
    current = {
        "import_time_seconds": 1.0,
        # Missing classification metrics on purpose
        "regression": {
            "explain_factual_time_s": 0.1,
            "explore_alternatives_time_s": 0.2,
        },
    }
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    thresholds_path = Path("benchmarks/perf_thresholds.json")
    _write(baseline_path, baseline)
    _write(current_path, current)

    exit_code = check_perf_micro.main(
        [str(baseline_path), str(current_path), str(thresholds_path)]
    )
    assert exit_code == 2


def test_explain_metrics_pass_when_within_threshold(tmp_path, capsys):
    baseline = {
        "import_time_seconds": 2.0,
        "classification": {
            "explain_factual_time_s": 0.2,
            "explore_alternatives_time_s": 0.3,
        },
        "regression": {
            "explain_factual_time_s": 0.2,
            "explore_alternatives_time_s": 0.3,
        },
    }
    current = {
        "import_time_seconds": 1.8,
        "classification": {
            "explain_factual_time_s": 0.18,
            "explore_alternatives_time_s": 0.25,
        },
        "regression": {
            "explain_factual_time_s": 0.19,
            "explore_alternatives_time_s": 0.29,
        },
    }
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    thresholds_path = Path("benchmarks/perf_thresholds.json")
    _write(baseline_path, baseline)
    _write(current_path, current)

    exit_code = check_perf_micro.main(
        [str(baseline_path), str(current_path), str(thresholds_path)]
    )
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "classification.explain_factual_time_s OK" in captured
    assert "regression.explore_alternatives_time_s OK" in captured


def test_committed_micro_baseline_includes_explain_metrics():
    micro_files = sorted(Path("benchmarks").glob("micro_*.json"))
    assert micro_files, "expected at least one micro baseline file"
    latest = micro_files[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    for section in ("classification", "regression"):
        assert section in payload, f"{section} missing from {latest}"
        section_data = payload[section]
        assert "explain_factual_time_s" in section_data, f"{section} factual timing missing"
        assert (
            "explore_alternatives_time_s" in section_data
        ), f"{section} alternatives timing missing"
