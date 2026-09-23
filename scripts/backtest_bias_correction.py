"""Walk-forward backtest: online bias correction on top of the production ensemble.

At each forecast origin, measure how far recent predictions were from actual demand
(only actuals already known at that origin), and scale the next prediction by that
ratio. Aimed at the slow concept drift found in production: a post-hoc correction
that reacts between retrains, instead of changing what the model is trained on.

Mirrors production: same model, features and CatBoost + weekly-naive blend. Only the
first part of each test window runs uncorrected (not enough feedback yet), so no
in-sample training residuals ever leak into the correction.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from backtest_catboost_direct import build_folds
from train_baseline import ARTIFACTS_DIR, load_training_data, station_metrics
from train_catboost_direct import FEATURES, add_origin_features, add_target_calendar, blend_predictions, model, naive_prediction_at_target

WINDOW_HOURS = (4, 12, 24)
ALPHAS = (0.5, 1.0)
SCOPES = ("station", "global")
CLIP = (0.8, 1.25)


def correction_factors(test: pd.DataFrame, predictions: np.ndarray, horizon_periods: int, window: int, scope: str) -> pd.Series:
    """actual/predicted over the last `window` targets already observed at each row's origin.

    A row forecasting target t from origin o = t - horizon can only use targets <= o, i.e.
    rows at least `horizon_periods` earlier: rolling sum ending there, shifted by horizon.
    Rows without a full window of feedback get NaN (left uncorrected by the caller).
    """
    frame = test[["station_id", "observed_at", "target_demand"]].copy()
    frame["pred"] = predictions
    if scope == "station":
        frame = frame.sort_values(["station_id", "observed_at"])
        grouped = frame.groupby("station_id")
        actual = grouped["target_demand"].transform(lambda s: s.rolling(window, min_periods=window).sum().shift(horizon_periods))
        predicted = grouped["pred"].transform(lambda s: s.rolling(window, min_periods=window).sum().shift(horizon_periods))
        return (actual / predicted).reindex(test.index)
    totals = frame.groupby("observed_at")[["target_demand", "pred"]].sum().sort_index()
    actual = totals["target_demand"].rolling(window, min_periods=window).sum().shift(horizon_periods)
    predicted = totals["pred"].rolling(window, min_periods=window).sum().shift(horizon_periods)
    return frame["observed_at"].map(actual / predicted).reindex(test.index)


def apply_correction(predictions: np.ndarray, factors: pd.Series, alpha: float) -> np.ndarray:
    scale = (1 + alpha * (factors.to_numpy() - 1)).clip(*CLIP)
    return predictions * np.where(np.isnan(scale), 1.0, scale)


def score(test: pd.DataFrame, predictions: np.ndarray) -> float:
    frame = test[["station_id", "target_demand"]].rename(columns={"target_demand": "demand"})
    return station_metrics(frame, predictions)["mean_station_wape"]


def run(horizons: list[int]) -> dict[str, Any]:
    data = add_origin_features(load_training_data())
    report: dict[str, Any] = {}
    for horizon in horizons:
        h = horizon // 15
        supervised = add_target_calendar(data, horizon)
        folds = []
        for fold in build_folds(supervised):
            train, test = fold["train"], fold["test"]
            fitted = model()
            fitted.fit(train[FEATURES], train["target_demand"], cat_features=["station_id"])
            production = blend_predictions(fitted.predict(test[FEATURES]), naive_prediction_at_target(data, test))
            row: dict[str, Any] = {"window_start": fold["window_start"].isoformat(), "production": score(test, production)}
            for scope in SCOPES:
                for hours in WINDOW_HOURS:
                    factors = correction_factors(test, production, h, hours * 4, scope)
                    for alpha in ALPHAS:
                        row[f"{scope}/{hours}h/a{alpha}"] = score(test, apply_correction(production, factors, alpha))
            folds.append(row)
        report[str(horizon)] = folds
    return report


def summarize(report: dict[str, Any]) -> None:
    for horizon, folds in report.items():
        production = np.array([f["production"] for f in folds])
        print(f"\nH={horizon} ({len(folds)} folds)  producción={production.mean():.4f}  por fold={[round(v, 4) for v in production]}")
        configs = [key for key in folds[0] if "/" in key]
        ranked = sorted(configs, key=lambda key: np.mean([f[key] for f in folds]))
        for key in ranked[:4]:
            values = np.array([f[key] for f in folds])
            deltas = values - production
            print(f"  {key:>18}: {values.mean():.4f}  delta={deltas.mean():+.4f}  por fold={[f'{d:+.4f}' for d in deltas]}")


def main() -> None:
    report = run([15, 30, 45, 60])
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "bias_correction_backtest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summarize(report)


if __name__ == "__main__":
    main()
