import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from backtest_bias_correction import apply_correction, correction_factors


def frame(periods: int = 40) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC")
    # target_demand = row index + 1, so each window sum is easy to compute by hand
    return pd.DataFrame({"station_id": "03000", "observed_at": timestamps, "target_demand": np.arange(periods) + 1.0})


def test_feedback_only_uses_targets_known_at_the_origin() -> None:
    test = frame()
    predictions = np.ones(len(test))
    horizon, window = 4, 8
    factors = correction_factors(test, predictions, horizon, window, "station")
    i = 20
    # row i predicts target i+4 from origin i; known targets are rows <= i-4, window of 8 -> rows 9..16
    expected = test["target_demand"].iloc[i - horizon - window + 1 : i - horizon + 1].sum() / window
    assert factors.iloc[i] == expected
    # and nothing from row i-3 onwards (targets after the origin) may enter the sum
    assert factors.iloc[i] < test["target_demand"].iloc[i - horizon + 1]


def test_rows_without_full_feedback_stay_uncorrected() -> None:
    test = frame()
    predictions = np.full(len(test), 10.0)
    factors = correction_factors(test, predictions, 4, 8, "station")
    corrected = apply_correction(predictions, factors, alpha=1.0)
    assert np.isnan(factors.iloc[0])
    assert corrected[0] == 10.0


def test_correction_is_clipped() -> None:
    predictions = np.array([100.0, 100.0])
    corrected = apply_correction(predictions, pd.Series([5.0, 0.1]), alpha=1.0)
    assert corrected.tolist() == [125.0, 80.0]
