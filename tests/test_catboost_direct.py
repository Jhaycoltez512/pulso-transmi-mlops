import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from train_catboost_direct import add_origin_features, add_target_calendar


def test_direct_features_use_only_past_demand_and_target_calendar() -> None:
    timestamps = pd.date_range("2026-01-01", periods=800, freq="15min", tz="UTC")
    source = pd.DataFrame({
        "station_id": "03000", "observed_at": timestamps, "demand": np.arange(len(timestamps)) + 10,
    })
    features = add_origin_features(source)
    supervised = add_target_calendar(features, 15)
    row = supervised.iloc[0]
    assert row["lag_672"] == row["current_demand"] - 672
    assert row["target_demand"] == row["current_demand"] + 1
    assert row["target_at"] == row["observed_at"] + pd.to_timedelta(15, unit="m")


def _synthetic(periods: int = 3 * 672, stations: tuple[str, ...] = ("03000", "05000"), scale: float = 1.0) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC")
    rng = np.random.default_rng(7)
    frames = []
    for offset, station in enumerate(stations):
        daily = 200 + 150 * np.sin(2 * np.pi * np.arange(periods) / 96) ** 2 + 40 * offset
        frames.append(pd.DataFrame({
            "station_id": station, "observed_at": timestamps,
            "demand": np.round((daily + rng.normal(0, 10, periods)) * scale),
        }))
    return pd.concat(frames, ignore_index=True)


def test_normalised_features_do_not_change_under_a_level_shift() -> None:
    from train_catboost_direct import FEATURES

    base = add_origin_features(_synthetic())
    shifted = add_origin_features(_synthetic().assign(demand=lambda f: f["demand"] * 1.35))
    numeric = [feature for feature in FEATURES if feature.startswith("norm_")]
    pd.testing.assert_frame_equal(base[numeric].reset_index(drop=True), shifted[numeric].reset_index(drop=True), rtol=1e-9)
    assert np.allclose(shifted["level"].dropna(), base["level"].dropna() * 1.35)


def test_week_ratio_tracks_a_recent_level_shift() -> None:
    data = _synthetic()
    shift_at = data["observed_at"].max() - pd.Timedelta(days=1)
    data.loc[data["observed_at"] > shift_at, "demand"] *= 1.35
    features = add_origin_features(data)
    last = features.loc[features["observed_at"] == features["observed_at"].max(), "week_ratio"]
    assert ((last > 1.25) & (last < 1.45)).all()


def test_production_model_is_refit_on_all_history_and_follows_a_shift() -> None:
    from train_catboost_direct import predict_demand, train_models

    data = add_origin_features(_synthetic(periods=5 * 672))
    models, report = train_models(data, [15])
    assert report["15"]["production_rows"] > report["15"]["train_rows"] + report["15"]["validation_rows"]
    # same model, inputs 35% higher: predictions scale with them instead of saturating
    frame = add_target_calendar(data, 15).tail(200)
    shifted = frame.assign(level=frame["level"] * 1.35)
    assert np.allclose(predict_demand(models[15], shifted), predict_demand(models[15], frame) * 1.35)
