import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from generate_weekly_submission_preview import validate


def test_submission_preview_validation_accepts_matching_targets() -> None:
    cycle = {
        "cycle_id": "cyc_test",
        "targets": [{"station_id": "03000", "target_at": "2026-01-01T00:15:00Z"}],
    }
    payload = {
        "schema_version": "1.0",
        "cycle_id": "cyc_test",
        "client_run_id": "weekly-naive-preview",
        "data_cutoff": "2026-01-01T00:00:00Z",
        "model": {"version": "weekly-naive-v1"},
        "predictions": [{"station_id": "03000", "target_at": "2026-01-01T00:15:00Z", "value": 42.0}],
    }
    assert validate(payload, cycle) == []


def test_submission_preview_validation_rejects_unexpected_target() -> None:
    cycle = {"cycle_id": "cyc_test", "targets": [{"station_id": "03000", "target_at": "2026-01-01T00:15:00Z"}]}
    payload = {
        "schema_version": "1.0", "cycle_id": "cyc_test", "client_run_id": "preview-001",
        "data_cutoff": "2026-01-01T00:00:00Z", "model": {"version": "v1"},
        "predictions": [{"station_id": "03000", "target_at": "2026-01-01T00:30:00Z", "value": 42.0}],
    }
    assert "prediction targets do not exactly match the active cycle" in validate(payload, cycle)
