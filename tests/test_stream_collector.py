import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from sync_stream_observations import new_rows_since, released_marker, version_for


def test_collector_uses_release_time_for_late_arriving_data() -> None:
    rows = [
        {"station_id": "03000", "observed_at": "2026-09-10T00:00:00Z", "released_at": "2026-09-10T01:00:00Z"},
        {"station_id": "03000", "observed_at": "2026-09-09T23:45:00Z", "released_at": "2026-09-10T02:00:00Z"},
    ]
    fresh = new_rows_since(rows, "2026-09-10T01:30:00Z")
    assert fresh == [rows[1]]
    assert released_marker(rows[1]) == "2026-09-10T02:00:00Z"


def test_data_version_is_deterministic_and_changes_with_data() -> None:
    rows = [{"station_id": "03000", "observed_at": "2026-09-10T00:00:00Z", "released_at": "2026-09-10T01:00:00Z", "demand": 10}]
    assert version_for(rows) == version_for(list(reversed(rows)))
    changed = [{**rows[0], "demand": 11}]
    assert version_for(rows) != version_for(changed)
