"""Evaluate a weekly seasonal-naive baseline using demand from seven days ago.

The model is persisted with Joblib for a uniform artifact interface, even though
the seasonal-naive predictor itself has no learned parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_baseline import ARTIFACTS_DIR, load_training_data, station_metrics, temporal_splits

LAG_PERIODS = 7 * 24 * 4  # 7 days at 15-minute frequency.


def build_weekly_naive_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the last observed demand at the same quarter-hour one week before."""
    data = frame.sort_values(["station_id", "observed_at"]).copy()
    data["prediction"] = data.groupby("station_id")["demand"].shift(LAG_PERIODS)
    return data.dropna(subset=["prediction"]).reset_index(drop=True)


def evaluate(frame: pd.DataFrame) -> dict:
    return station_metrics(frame, frame["prediction"].to_numpy())


def main() -> None:
    data = build_weekly_naive_data(load_training_data())
    train, validation, test = temporal_splits(data)
    validation_metrics = evaluate(validation)
    test_metrics = evaluate(test)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    metadata = {
        "model": "seasonal_naive_weekly",
        "lag_periods": LAG_PERIODS,
        "lag_minutes": LAG_PERIODS * 15,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_end": train["observed_at"].max().isoformat(),
        "validation_end": validation["observed_at"].max().isoformat(),
        "test_end": test["observed_at"].max().isoformat(),
    }
    joblib.dump({"model": "seasonal_naive_weekly", "metadata": metadata}, ARTIFACTS_DIR / "weekly_naive.joblib")
    (ARTIFACTS_DIR / "weekly_naive_metrics.json").write_text(
        json.dumps({"metadata": metadata, "validation": validation_metrics, "test": test_metrics}, indent=2),
        encoding="utf-8",
    )
    print(f"Validation accuracy: {validation_metrics['mean_station_accuracy']:.2f}")
    print(f"Test accuracy: {test_metrics['mean_station_accuracy']:.2f}")
    print(f"Artifact: {ARTIFACTS_DIR / 'weekly_naive.joblib'}")


if __name__ == "__main__":
    main()
