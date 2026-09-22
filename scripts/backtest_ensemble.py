"""Walk-forward backtest for blending CatBoost with the weekly seasonal-naive baseline.

For each fold, scores a sweep of blend weights (0 = pure naive, 1 = pure
CatBoost) so a winning weight can be picked on evidence across several
7-day test windows, not a single noisy split.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from train_baseline import ARTIFACTS_DIR, load_training_data, station_metrics
from train_catboost_direct import FEATURES, add_origin_features, add_target_calendar, model, naive_prediction_at_target
from backtest_catboost_direct import build_folds

WEIGHTS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]  # share given to the CatBoost prediction


def run_backtest(horizons: list[int]) -> dict[str, Any]:
    data = add_origin_features(load_training_data())
    report: dict[str, Any] = {}
    for horizon in horizons:
        supervised = add_target_calendar(data, horizon)
        fold_reports = []
        for fold in build_folds(supervised):
            fitted = model()
            fitted.fit(fold["train"][FEATURES], fold["train"]["target_demand"], cat_features=["station_id"])
            catboost_pred = fitted.predict(fold["test"][FEATURES])
            naive_pred = naive_prediction_at_target(data, fold["test"])
            valid = ~np.isnan(naive_pred)
            frame = fold["test"].loc[valid, ["station_id", "target_demand"]].rename(columns={"target_demand": "demand"})
            row = {
                "window_start": fold["window_start"].isoformat(), "window_end": fold["window_end"].isoformat(),
                "test_rows": int(valid.sum()), "dropped_no_naive": int((~valid).sum()),
            }
            for weight in WEIGHTS:
                blended = weight * catboost_pred[valid] + (1 - weight) * naive_pred[valid]
                row[f"wape_w{weight}"] = station_metrics(frame, blended)["mean_station_wape"]
            fold_reports.append(row)
        summary = {}
        for weight in WEIGHTS:
            values = np.array([f[f"wape_w{weight}"] for f in fold_reports])
            summary[str(weight)] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
        report[str(horizon)] = {"folds": fold_reports, "by_weight": summary, "n_folds": len(fold_reports)}
    return report


def main() -> None:
    report = run_backtest([15, 30, 45, 60])
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "ensemble_backtest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for horizon, result in report.items():
        print(f"H={horizon} ({result['n_folds']} folds):")
        for weight, stats in result["by_weight"].items():
            label = "naive" if weight == "0.0" else "catboost" if weight == "1.0" else f"blend w={weight}"
            print(f"  {label:>14}: WAPE mean={stats['mean']:.4f} std={stats['std']:.4f}")


if __name__ == "__main__":
    main()
