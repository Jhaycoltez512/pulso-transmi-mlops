import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from forecast_adjustments import (  # noqa: E402
    blend_with_reference, cap_growth, combine, expert_weights, reference_inputs, reference_lag_days,
)


def test_reference_day_keeps_weekdays_and_weekends_apart() -> None:
    # 2026-09-14 is a Monday in Bogota; noon local = 17:00 UTC
    days = pd.Series(pd.to_datetime([f"2026-09-{d}T17:00:00Z" for d in (14, 15, 18, 19, 20)], utc=True))
    # Mon -> Fri (3), Tue -> Mon (1), Fri -> Thu (1), Sat -> last Sat (7), Sun -> last Sun (7)
    assert reference_lag_days(days).tolist() == [3, 1, 1, 7, 7]


def test_blend_is_heavier_at_longer_horizons_and_skips_missing_reference() -> None:
    model = np.array([100.0, 100.0, 100.0])
    reference = np.array([200.0, 200.0, np.nan])
    out = blend_with_reference(model, reference, np.array([15, 60, 60]))
    assert np.allclose(out, [100 + 0.0625 * 100, 125.0, 100.0])


def test_growth_cap_only_applies_to_inflated_stations() -> None:
    prediction = np.array([6718.0, 6718.0, 500.0])
    level_now = np.array([3759.0, 900.0, 400.0])
    reference_level = np.array([700.0, 800.0, 100.0])  # x5.4 inflated, x1.1 normal, x4 inflated
    reference = np.array([741.0, 741.0, 600.0])
    out = cap_growth(prediction, level_now, reference_level, reference)
    # inflated: capped at max(level, comparable day); normal: untouched; already below cap: untouched
    assert out.tolist() == [3759.0, 6718.0, 500.0]


def test_reference_inputs_look_up_the_comparable_day() -> None:
    times = pd.date_range("2026-09-14T00:00:00Z", periods=4 * 96, freq="15min", tz="UTC")
    data = pd.DataFrame({"station_id": "07111", "observed_at": times, "demand": np.arange(len(times), dtype=float)})
    origin = pd.Timestamp("2026-09-16T17:00:00Z")  # Wednesday -> Tuesday
    frame = pd.DataFrame({"station_id": ["07111"], "observed_at": [origin], "target_at": [origin + timedelta(hours=1)]})
    out = reference_inputs(data, frame).iloc[0]
    index = {t: i for i, t in enumerate(times)}
    assert out["reference"] == index[origin + timedelta(hours=1) - timedelta(days=1)]
    assert out["persistence"] == index[origin]
    assert out["level_now"] == np.mean([index[origin] - k for k in range(4)])


def test_expert_weights_follow_recent_errors_and_ignore_the_future() -> None:
    t0 = pd.Timestamp("2026-09-18T00:00:00Z")
    history = pd.DataFrame({
        "station_id": "07111", "horizon_minutes": 15,
        "target_at": [t0 + timedelta(hours=h) for h in range(6)],
        "actual": 100.0,
        # model is 20 off, persistence 5 off, comparable 50 off -- except one future row
        "model": [120.0] * 5 + [100.0], "persistence": [105.0] * 5 + [900.0], "comparable": [150.0] * 6,
    })
    queries = pd.DataFrame({"station_id": ["07111"], "horizon_minutes": [15], "observed_at": [t0 + timedelta(hours=4)]})
    weights = expert_weights(history, queries, ["station_id", "horizon_minutes"], timedelta(hours=6), power=1.0).iloc[0]
    assert weights["persistence"] > weights["model"] > weights["comparable"]
    assert np.isclose(weights.sum(), 1.0)
    # inverse-MAE: 1/5 : 1/20 : 1/50
    assert np.isclose(weights["persistence"] / weights["model"], 4.0)


def test_expert_weights_fall_back_to_the_model_without_history() -> None:
    history = pd.DataFrame(columns=["station_id", "target_at", "actual", "model", "persistence", "comparable"])
    history["target_at"] = pd.to_datetime(history["target_at"], utc=True)
    queries = pd.DataFrame({"station_id": ["07111"], "observed_at": [pd.Timestamp("2026-09-18T00:00:00Z")]})
    weights = expert_weights(history.astype({"station_id": str}), queries, ["station_id"], timedelta(hours=6), power=2.0).iloc[0]
    assert weights.tolist() == [1.0, 0.0, 0.0]


def test_combine_renormalises_over_available_experts() -> None:
    predictions = pd.DataFrame({"model": [100.0], "persistence": [200.0], "comparable": [np.nan]})
    weights = pd.DataFrame({"model": [0.25], "persistence": [0.25], "comparable": [0.5]})
    assert combine(predictions, weights).tolist() == [150.0]
