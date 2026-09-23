import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from backtest_seasonal_features import add_seasonal_features
from train_catboost_direct import add_origin_features, add_target_calendar


def synthetic(periods: int = 3000) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC")
    return pd.DataFrame({"station_id": "03000", "observed_at": timestamps, "demand": np.arange(periods) + 10})


def test_target_aligned_lags_point_k_weeks_before_target() -> None:
    origin = add_origin_features(synthetic())
    supervised = add_target_calendar(add_seasonal_features(origin, horizon_periods=4, weeks=3), 60).dropna(subset=["tw_lag_3"])
    row = supervised.iloc[0]
    # demand is a counter, so "k weeks before target" is target_demand - k*672 exactly
    assert row["tw_lag_1"] == row["target_demand"] - 672
    assert row["tw_lag_3"] == row["target_demand"] - 3 * 672
    assert row["tw_trend"] == 672


def test_features_never_use_the_target_itself() -> None:
    origin = add_origin_features(synthetic())
    supervised = add_target_calendar(add_seasonal_features(origin, horizon_periods=1, weeks=2), 15).dropna(subset=["level_ratio_16"])
    row = supervised.iloc[0]
    # every seasonal lag must be strictly older than the value being predicted
    assert row["tw_lag_1"] < row["target_demand"]
    assert row["current_demand"] < row["target_demand"]
