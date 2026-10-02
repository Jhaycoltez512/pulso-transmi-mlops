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
