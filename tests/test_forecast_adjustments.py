import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from forecast_adjustments import (  # noqa: E402
    PERIODIC_EXPERTS, blend_with_reference, cap_growth, combine, expert_weights, production_adjust, reference_inputs,
    reference_lag_days,
)
from load_supabase import SupabaseLoader  # noqa: E402


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


def _cycle(model_value: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flat station at 100 for 4 days; the history says persistence was exact and the model 50 off."""
    times = pd.date_range("2026-09-14T00:00:00Z", periods=4 * 96, freq="15min", tz="UTC")
    data = pd.DataFrame({"station_id": "07111", "observed_at": times, "demand": 100.0})
    cutoff = times[-1]
    predictions = pd.DataFrame({
        "station_id": ["07111"], "observed_at": [cutoff], "target_at": [cutoff + timedelta(minutes=60)],
        "horizon_minutes": [60], "model": [model_value],
    })
    targets = [cutoff - timedelta(hours=h) for h in range(1, 6)]
    history = pd.DataFrame({"station_id": "07111", "horizon_minutes": 60, "target_at": targets, "model": 150.0, "actual": 100.0})
    return data, predictions, history


def test_production_adjust_modes() -> None:
    data, predictions, history = _cycle(model_value=200.0)
    none, _ = production_adjust(data, predictions, history, "none", timedelta(hours=24), 2.0)
    assert none.tolist() == [200.0]
    # comparable day is 100 at the target: h60 blend puts a quarter of the weight on it
    blended, _ = production_adjust(data, predictions, history, "blend_cap", timedelta(hours=24), 2.0)
    assert np.allclose(blended, [175.0])
    # persistence and comparable day were exact lately, the model wasn't: the ensemble follows them
    ensembled, weights = production_adjust(data, predictions, history, "ensemble", timedelta(hours=24), 2.0)
    assert abs(ensembled[0] - 100.0) < 1.0
    assert weights["model"] < 0.01


def test_production_adjust_without_history_submits_the_adjusted_model() -> None:
    data, predictions, history = _cycle(model_value=200.0)
    ensembled, weights = production_adjust(data, predictions, history.iloc[0:0], "ensemble", timedelta(hours=24), 2.0)
    assert np.allclose(ensembled, [175.0])
    assert weights["model"] == 1.0


def test_production_ensemble_pooled_over_stations_uses_every_station_history() -> None:
    data, predictions, history = _cycle(model_value=200.0)
    other = data.assign(station_id="06000")
    data = pd.concat([data, other], ignore_index=True)
    # 06000 has no history of its own: pooled per horizon, it still learns from 07111's errors
    predictions = predictions.assign(station_id="06000")
    pooled, weights = production_adjust(data, predictions, history, "ensemble", timedelta(hours=24), 3.0,
                                        ensemble_on_adjusted=False, pool_stations=True)
    assert abs(pooled[0] - 100.0) < 1.0
    alone, _ = production_adjust(data, predictions, history, "ensemble", timedelta(hours=24), 3.0,
                                 ensemble_on_adjusted=False, pool_stations=False)
    assert alone.tolist() == [200.0]


def test_production_adjust_mixes_timestamp_resolutions() -> None:
    # live cycle: cutoff parsed at second resolution, observations and history at nanoseconds
    data, predictions, history = _cycle(model_value=200.0)
    data["observed_at"] = data["observed_at"].astype("datetime64[ns, UTC]")
    predictions["observed_at"] = predictions["observed_at"].astype("datetime64[s, UTC]")
    predictions["target_at"] = predictions["target_at"].astype("datetime64[ns, UTC]")
    history["target_at"] = history["target_at"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    for mode in ("blend_cap", "ensemble"):
        values, _ = production_adjust(data, predictions, history, mode, timedelta(hours=6), 3.0,
                                      ensemble_on_adjusted=False, pool_stations=True)
        assert np.isfinite(values).all()


def test_select_all_pages_past_the_row_cap() -> None:
    loader = SupabaseLoader.__new__(SupabaseLoader)
    rows = [{"id": i} for i in range(2500)]
    calls = []

    def fake_select(table, params):
        calls.append(params)
        return rows[params["offset"]:params["offset"] + params["limit"]]

    loader.select = fake_select
    assert loader.select_all("predictions", {"select": "id"}, order="id.asc") == rows
    assert [c["offset"] for c in calls] == [0, 1000, 2000]
    assert all(c["order"] == "id.asc" for c in calls)


def test_reference_inputs_add_the_periodic_lags() -> None:
    times = pd.date_range("2026-09-14T00:00:00Z", periods=4 * 96, freq="15min", tz="UTC")
    data = pd.DataFrame({"station_id": "07111", "observed_at": times, "demand": np.arange(len(times), dtype=float)})
    origin = pd.Timestamp("2026-09-16T17:00:00Z")
    frame = pd.DataFrame({"station_id": ["07111"], "observed_at": [origin], "target_at": [origin + timedelta(minutes=45)]})
    out = reference_inputs(data, frame).iloc[0]
    index = {t: i for i, t in enumerate(times)}
    target = origin + timedelta(minutes=45)
    assert out["lag_4h"] == index[target - timedelta(hours=4)]
    # the periodic expert averages the 4h period over the last 24h (6 copies)
    assert out["per_4h"] == np.mean([index[target - timedelta(hours=4 * k)] for k in range(1, 7)])
    assert set(PERIODIC_EXPERTS) <= set(out.index)


def test_ensemble_follows_a_periodic_regime() -> None:
    # flat comparable day, then a 4h-periodic oscillation: the 4h copy is exact, the model is not
    times = pd.date_range("2026-09-14T00:00:00Z", periods=4 * 96, freq="15min", tz="UTC")
    demand = np.full(len(times), 100.0)
    oscillating = times >= pd.Timestamp("2026-09-16T00:00:00Z")  # > 24h, so every averaged copy is in-regime
    demand[oscillating] = 100 * np.exp(1.5 * np.sin(2 * np.pi * np.arange(oscillating.sum()) / 16))
    data = pd.DataFrame({"station_id": "07111", "observed_at": times, "demand": demand})
    cutoff = times[-5]
    target = cutoff + timedelta(minutes=60)
    predictions = pd.DataFrame({"station_id": ["07111"], "observed_at": [cutoff], "target_at": [target], "horizon_minutes": [60], "model": [100.0]})
    past_targets = [cutoff - timedelta(hours=k) for k in range(0, 6)]
    actual = data.set_index("observed_at")["demand"]
    history = pd.DataFrame({"station_id": "07111", "horizon_minutes": 60, "target_at": past_targets,
                            "model": 100.0, "actual": [actual[t] for t in past_targets]})
    experts = ("model", "persistence", "comparable", *PERIODIC_EXPERTS)
    values, weights = production_adjust(data, predictions, history, "ensemble", timedelta(hours=6), 3.0,
                                        ensemble_on_adjusted=False, pool_stations=True, experts=experts)
    # the 4h-period average is exact (so is the comparable day: 24h is 6 periods); the model isn't
    assert weights["per_4h"] > 0.4 and weights["model"] < 0.01
    assert abs(values[0] - actual[target]) < 0.05 * actual[target]


def test_periodic_break_fires_only_when_the_period_stops_repeating() -> None:
    from forecast_adjustments import periodic_break

    times = pd.date_range("2026-09-18T00:00Z", "2026-09-20T12:00Z", freq="15min")
    phase = (times - times[0]) / pd.Timedelta(hours=4) * 2 * np.pi
    periodic = pd.DataFrame({"station_id": "A", "observed_at": times, "demand": 500 + 400 * np.sin(phase)})
    broke, info = periodic_break(periodic, times[-1])
    assert not broke and info["recent_wape"] < 0.01

    after = pd.date_range(times[-1] + pd.Timedelta(minutes=15), periods=4, freq="15min")
    flat = pd.DataFrame({"station_id": "A", "observed_at": after, "demand": [100.0, 1500.0, 100.0, 1500.0]})
    broke, info = periodic_break(pd.concat([periodic, flat]), after[-1])
    assert broke and info["recent_wape"] > 0.25


def test_break_guard_drops_the_periodic_experts() -> None:
    from forecast_adjustments import EXPERTS, LAG_EXPERTS, production_adjust

    times = pd.date_range("2026-09-10T00:00Z", "2026-09-20T13:00Z", freq="15min")
    phase = (times - times[0]) / pd.Timedelta(hours=4) * 2 * np.pi
    demand = np.asarray(500 + 400 * np.sin(phase), dtype=float)
    demand[times > pd.Timestamp("2026-09-20T12:00Z")] = 1500.0  # pattern stops at a new level
    data = pd.DataFrame({"station_id": "A", "observed_at": times, "demand": demand})
    cutoff = times[-1]
    frame = pd.DataFrame({"station_id": "A", "observed_at": cutoff, "target_at": [cutoff + pd.Timedelta(minutes=15)], "horizon_minutes": [15], "model": [0.0]})
    hist_t = [cutoff - pd.Timedelta(minutes=15 * k) for k in range(4)]
    history = pd.DataFrame({"station_id": "A", "horizon_minutes": 15, "target_at": hist_t, "model": 0.0, "actual": 1500.0})
    values, info = production_adjust(data, frame, history, "ensemble", pd.Timedelta(hours=4), 6.0, ensemble_on_adjusted=False,
                                     pool_stations=True, experts=(*EXPERTS, *LAG_EXPERTS), break_guard=True)
    assert info["regime_break"] == 1.0
    assert not any(k.startswith("lag_") for k in info)
    assert abs(values[0] - 1500.0) < 5.0  # persistence, the only expert right over the last hour


def test_trend_expert_extends_the_last_hour_slope_damped() -> None:
    from forecast_adjustments import reference_inputs

    times = pd.date_range("2026-09-20T12:00Z", periods=5, freq="15min")  # 100, 110, ..., 140
    data = pd.DataFrame({"station_id": "A", "observed_at": times, "demand": [100.0, 110.0, 120.0, 130.0, 140.0]})
    cutoff = times[-1]
    frame = pd.DataFrame({"station_id": "A", "observed_at": cutoff, "target_at": [cutoff + pd.Timedelta(minutes=15 * k) for k in (1, 4)]})
    out = reference_inputs(data, frame)
    assert np.allclose(out["trend"], [140 + 0.5 * 10 * 1, 140 + 0.5 * 10 * 4])
    assert np.allclose(out["trend_full"], [150, 180])
