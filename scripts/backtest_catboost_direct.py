"""Walk-forward backtest for the direct multi-horizon CatBoost model.

A single validation/test split is noisy: the same code change can improve one
7-day window and worsen the next just from randomness in that window. This
script retrains on an expanding window and evaluates on successive rolling
7-day test windows instead, so WAPE differences can be judged against the
spread across folds rather than a single draw.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_baseline import ARTIFACTS_DIR, load_training_data, station_metrics
from train_catboost_direct import FEATURES, add_origin_features, add_target_calendar, model

TEST_DAYS = 7
MIN_TRAIN_ROWS = 8064 * 2  # at least two weeks, so lag_672/rolling_672 are not mostly empty


def build_folds(supervised: pd.DataFrame, test_days: int = TEST_DAYS) -> list[dict[str, Any]]:
    """Walk backward from the most recent target timestamp in non-overlapping test windows."""
    folds = []
    fold_end = supervised["target_at"].max()
    while True:
        test_start = fold_end - timedelta(days=test_days)
        train = supervised.loc[supervised["target_at"] <= test_start]
        test = supervised.loc[(supervised["target_at"] > test_start) & (supervised["target_at"] <= fold_end)]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            break
        folds.append({"train": train, "test": test, "window_start": test_start, "window_end": fold_end})
        fold_end = test_start
    folds.reverse()
    return folds


def run_backtest(horizons: list[int]) -> dict[str, Any]:
    data = add_origin_features(load_training_data())
    report: dict[str, Any] = {}
    for horizon in horizons:
        supervised = add_target_calendar(data, horizon)
        fold_reports = []
        for fold in build_folds(supervised):
            fitted = model()
            fitted.fit(fold["train"][FEATURES], fold["train"]["target_demand"], cat_features=["station_id"])
            metrics = station_metrics(
                fold["test"][["station_id", "target_demand"]].rename(columns={"target_demand": "demand"}),
                fitted.predict(fold["test"][FEATURES]),
            )
            fold_reports.append({
                "window_start": fold["window_start"].isoformat(),
                "window_end": fold["window_end"].isoformat(),
                "train_rows": len(fold["train"]),
                "test_rows": len(fold["test"]),
                "mean_station_wape": metrics["mean_station_wape"],
                "mean_station_accuracy": metrics["mean_station_accuracy"],
            })
        wapes = np.array([f["mean_station_wape"] for f in fold_reports])
        report[str(horizon)] = {
            "folds": fold_reports,
            "wape_mean": float(wapes.mean()) if len(wapes) else None,
            "wape_std": float(wapes.std(ddof=1)) if len(wapes) > 1 else 0.0,
            "n_folds": len(fold_reports),
        }
    return report


def main() -> None:
    report = run_backtest([15, 30, 45, 60])
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "catboost_direct_backtest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for horizon, result in report.items():
        if result["n_folds"] == 0:
            print(f"H={horizon}: not enough history for any fold yet.")
            continue
        print(f"H={horizon}: {result['n_folds']} folds, WAPE mean={result['wape_mean']:.4f} std={result['wape_std']:.4f}")


if __name__ == "__main__":
    main()
