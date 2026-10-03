import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import pytest
from sync_stream_observations import new_rows_since, normalize_rows, released_marker, version_for


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


def test_same_release_is_not_new_when_the_cursor_comes_back_in_another_format() -> None:
    rows = [{"station_id": "03000", "observed_at": "2026-09-18T03:30:00Z", "released_at": "2026-09-30T20:12:55.248593Z"}]
    # Supabase returns the stored cursor as +00:00; as strings "Z" > "+" made this row "new" forever
    assert new_rows_since(rows, "2026-09-30T20:12:55.248593+00:00") == []
    assert new_rows_since(rows, "2026-09-30T20:12:55+00:00") == rows


def test_schema_v2_measurement_rows_map_onto_the_observations_columns() -> None:
    v1 = {"station_id": "02300", "observed_at": "2026-09-20T12:00:00Z", "released_at": "2026-10-03T23:09:00Z", "demand": 541}
    v2 = {"station_id": "02300", "observed_at": "2026-09-20T07:15:00-05:00", "released_at": "2026-10-03T23:19:15.124962Z",
          "schema_version": 2, "measurement": {"value": "546.00", "unit": "passengers", "quality": "observed"}}
    rows, skipped = normalize_rows([v1, v2])
    assert skipped == 0
    assert rows[0] == v1
    assert rows[1] == {"station_id": "02300", "observed_at": "2026-09-20T12:15:00Z",
                       "released_at": "2026-10-03T23:19:15.124962Z", "demand": 546}


def test_rows_without_a_value_are_skipped_and_unknown_units_fail_loudly() -> None:
    base = {"station_id": "03000", "observed_at": "2026-09-20T12:15:00Z", "released_at": "2026-10-03T23:19:15Z", "schema_version": 2}
    missing = {**base, "measurement": {"value": None, "unit": "passengers", "quality": "missing"}}
    thousands = {**base, "measurement": {"value": "0.546", "unit": "thousand_passengers", "quality": "observed"}}
    rows, skipped = normalize_rows([missing, thousands])
    assert skipped == 1 and rows[0]["demand"] == 546
    with pytest.raises(ValueError, match="Unknown measurement unit"):
        normalize_rows([{**base, "measurement": {"value": "1", "unit": "furlongs"}}])
