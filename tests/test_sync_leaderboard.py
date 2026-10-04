import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from sync_leaderboard import cycle_results  # noqa: E402


def test_cycle_result_recovers_the_newest_cycle_wape() -> None:
    # 9 cycles at total demand 900 with WAPE 0.20, then a 100-demand cycle scored at WAPE 0.10
    previous = [{"display_name": "a", "raw_wape": 0.20, "coverage": 1.0}]
    current = [{"display_name": "a", "raw_wape": (0.20 * 900 + 0.10 * 100) / 1000, "coverage": 1.0}]
    wape, accuracy = cycle_results(previous, current, 10, 900.0, 1000.0)["a"]
    assert abs(wape - 0.10) < 1e-9 and abs(accuracy - 90.0) < 1e-9


def test_a_missed_cycle_has_no_result() -> None:
    previous = [{"display_name": "b", "raw_wape": 0.30, "coverage": 8 / 9}]
    current = [{"display_name": "b", "raw_wape": 0.30, "coverage": 8 / 10}]
    assert cycle_results(previous, current, 10, 900.0, 1000.0) == {"b": None}


def test_no_result_without_demand_totals_or_a_previous_snapshot() -> None:
    current = [{"display_name": "a", "raw_wape": 0.2, "coverage": 1.0}]
    assert cycle_results([], current, 10, 900.0, 1000.0) == {}
    assert cycle_results(current, current, 10, None, 1000.0) == {}


def test_missing_actuals_are_interpolated_so_the_cycle_can_still_be_derived() -> None:
    import pandas as pd

    from sync_leaderboard import fill_missing_demand

    t = pd.date_range("2026-09-20T14:00Z", periods=4, freq="15min")
    obs = pd.DataFrame({"station_id": ["A", "A", "A", "B", "B", "B", "B"], "observed_at": [t[0], t[1], t[3], *t],
                        "demand": [100.0, 200.0, 400.0, 1.0, 2.0, 3.0, 4.0]})
    out = fill_missing_demand(obs).set_index(["station_id", "observed_at"])["demand"]
    assert len(out) == 8 and out[("A", t[2])] == 300.0
