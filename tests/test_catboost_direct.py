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
        "rain_forecast": np.arange(len(timestamps)) * 0.1,
        "temperature_forecast": np.full(len(timestamps), 18.0),
        "event_intensity": np.zeros(len(timestamps)),
    })
    features = add_origin_features(source)
    supervised = add_target_calendar(features, 15)
    row = supervised.iloc[0]
    assert row["lag_672"] == row["current_demand"] - 672
    assert row["target_demand"] == row["current_demand"] + 1
    assert row["target_at"] == row["observed_at"] + pd.to_timedelta(15, unit="m")
    # target_at falls inside the known context range, so the forecast should match that timestamp, not the origin.
    expected_rain = source.loc[source["observed_at"] == row["target_at"], "rain_forecast"].iloc[0]
    assert row["rain_forecast"] == expected_rain
