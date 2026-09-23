"""Walk-forward backtest for recency-weighted training, to see if it helps against the
gradual concept drift found in production (docs/collector-and-lineage.md has the writeup):
demand kept flat, no PSI drift in any monitored feature, degradation spread across ~10/12
stations -- consistent with the relationship between features and demand shifting slowly,
not a one-off event.

Weights each training row by an exponential decay of its distance (in days) from the fold's
training cutoff, so CatBoost's loss favors recent, more-representative rows without fully
discarding older history the way a hard rolling window would.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from train_baseline import ARTIFACTS_DIR, load_training_data, station_metrics
from train_catboost_direct import FEATURES, add_origin_features, add_target_calendar, model
from backtest_catboost_direct import build_folds

HALF_LIVES_DAYS = [None, 30, 14, 7, 3]  # None = current behaviour, unweighted


def recency_weights(train: pd.DataFrame, half_life_days: float | None) -> np.ndarray | None:
    if half_life_days is None:
        return None
    cutoff = train["target_at"].max()
    age_days = (cutoff - train["target_at"]).dt.total_seconds() / 86400
    return np.power(0.5, age_days / half_life_days).to_numpy()


def run_backtest(horizons: list[int]) -> dict[str, Any]:
    data = add_origin_features(load_training_data())
    report: dict[str, Any] = {}
    for horizon in horizons:
        supervised = add_target_calendar(data, horizon)
        fold_reports = []
        for fold in build_folds(supervised):
            row = {"window_start": fold["window_start"].isoformat(), "window_end": fold["window_end"].isoformat()}
            for half_life in HALF_LIVES_DAYS:
                fitted = model()
                weights = recency_weights(fold["train"], half_life)
                fitted.fit(fold["train"][FEATURES], fold["train"]["target_demand"], cat_features=["station_id"], sample_weight=weights)
                metrics = station_metrics(
                    fold["test"][["station_id", "target_demand"]].rename(columns={"target_demand": "demand"}),
                    fitted.predict(fold["test"][FEATURES]),
                )
                row[f"wape_hl{half_life}"] = metrics["mean_station_wape"]
            fold_reports.append(row)
        summary = {}
        for half_life in HALF_LIVES_DAYS:
            values = np.array([f[f"wape_hl{half_life}"] for f in fold_reports])
            summary[str(half_life)] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
        report[str(horizon)] = {"folds": fold_reports, "by_half_life": summary, "n_folds": len(fold_reports)}
    return report


def main() -> None:
    import json

    report = run_backtest([15, 30, 45, 60])
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "recency_weight_backtest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for horizon, result in report.items():
        print(f"H={horizon} ({result['n_folds']} folds):")
        for half_life, stats in result["by_half_life"].items():
            label = "sin ponderar" if half_life == "None" else f"half-life={half_life}d"
            print(f"  {label:>16}: WAPE mean={stats['mean']:.4f} std={stats['std']:.4f}")


if __name__ == "__main__":
    main()
