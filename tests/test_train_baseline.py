import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


if importlib.util.find_spec("sklearn") is None:
    pytest.skip("ML dependencies are optional", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from train_baseline import build_features, station_metrics, temporal_splits
from train_weekly_naive import LAG_PERIODS, build_weekly_naive_data


def source_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=3_000, freq="15min", tz="UTC")
    return pd.concat([
        pd.DataFrame({
            "station_id": station,
            "observed_at": timestamps,
            "demand": np.arange(len(timestamps)) % 100 + 10,
            "rain_forecast": 0.0,
            "temperature_forecast": 15.0,
            "event_intensity": 0.0,
        })
        for station in ("03000", "05000")
    ], ignore_index=True)


def test_features_use_required_lags_and_temporal_split_is_disjoint() -> None:
    features = build_features(source_frame())
    train, validation, test = temporal_splits(features)
    assert features["lag_96"].notna().all()
    assert features["lag_672"].notna().all()
    assert train["observed_at"].max() < validation["observed_at"].min()
    assert validation["observed_at"].max() < test["observed_at"].min()


def test_metrics_perfect_prediction_has_full_accuracy() -> None:
    frame = source_frame().iloc[:10]
    metrics = station_metrics(frame, frame["demand"].to_numpy())
    assert metrics["mean_station_accuracy"] == 100.0
    assert metrics["mean_station_wape"] == 0.0


def test_weekly_naive_uses_exactly_672_previous_periods() -> None:
    data = build_weekly_naive_data(source_frame())
    assert LAG_PERIODS == 672
    first_station = data.loc[data["station_id"] == "03000"].iloc[0]
    assert first_station["prediction"] == 10
