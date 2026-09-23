"""Walk-forward backtest: multi-week seasonal profile + recent-level features vs the current model.

The production model drifts slowly (concept drift, see docs/ml-baselines.md). Reweighting
training rows didn't help; this tests changing what the model sees instead:
  - target-aligned weekly lags (demand at target_at - k weeks) and their median/trend,
  - a "level" ratio: how far current demand sits above/below its own multi-week profile,
    averaged over the last 1h and 4h, so the model can see and correct an ongoing shift.
The weekly-naive half of the ensemble also switches to the multi-week median.

Features are computed on the contiguous per-station series before any dropna, so shifts
never cross gaps. All lags are strictly in the past relative to the forecast origin.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from train_baseline import ARTIFACTS_DIR, load_training_data, station_metrics
from train_catboost_direct import ENSEMBLE_WEIGHT, FEATURES, add_origin_features, add_target_calendar, model

WEEK = 672
TEST_DAYS = 7
MIN_TRAIN_ROWS = 8064  # one week; variants lose 2-3 weeks of history to warmup
MIN_BASELINE_TRAIN_ROWS = 8064 * 2


def seasonal_columns(weeks: int) -> list[str]:
    return [*(f"tw_lag_{k}" for k in range(1, weeks + 1)), "tw_median", "tw_trend", "level_ratio_4", "level_ratio_16", "tw_adjusted"]


def add_seasonal_features(data: pd.DataFrame, horizon_periods: int, weeks: int) -> pd.DataFrame:
    """Adds target-aligned multi-week lags and origin-level ratios. `data` must be sorted by station, time."""
    result = data.copy()
    grouped = result.groupby("station_id")["demand"]
    for k in range(1, weeks + 1):
        # demand at target_at - k weeks == origin shifted by (k*WEEK - horizon) periods
        result[f"tw_lag_{k}"] = grouped.shift(k * WEEK - horizon_periods)
    target_lags = result[[f"tw_lag_{k}" for k in range(1, weeks + 1)]]
    result["tw_median"] = target_lags.median(axis=1, skipna=False)
    result["tw_trend"] = result["tw_lag_1"] - result["tw_lag_2"]

    origin_profile = pd.concat([grouped.shift(k * WEEK) for k in range(1, weeks + 1)], axis=1).mean(axis=1, skipna=False)
    ratio = (result["demand"] + 1) / (origin_profile + 1)
    ratio_by_station = ratio.groupby(result["station_id"])
    result["level_ratio_4"] = ratio_by_station.transform(lambda s: s.rolling(4, min_periods=4).mean())
    result["level_ratio_16"] = ratio_by_station.transform(lambda s: s.rolling(16, min_periods=16).mean())
    result["tw_adjusted"] = result["tw_median"] * result["level_ratio_4"]
    return result


def blend(catboost_pred: np.ndarray, naive_pred: np.ndarray) -> np.ndarray:
    naive_filled = np.where(np.isnan(naive_pred), catboost_pred, naive_pred)
    return np.clip(ENSEMBLE_WEIGHT * catboost_pred + (1 - ENSEMBLE_WEIGHT) * naive_filled, 0, None)


def score(test: pd.DataFrame, predictions: np.ndarray) -> float:
    frame = test[["station_id", "target_demand"]].rename(columns={"target_demand": "demand"})
    return station_metrics(frame, predictions)["mean_station_wape"]


def run(horizons: list[int]) -> dict[str, Any]:
    origin = add_origin_features(load_training_data())
    variants = {"V2": 2, "V3": 3}
    report: dict[str, Any] = {}
    for horizon in horizons:
        h = horizon // 15
        max_weeks = max(variants.values())
        supervised = add_target_calendar(add_seasonal_features(origin, h, max_weeks), horizon)
        all_variant_cols = seasonal_columns(max_weeks)
        # Each variant gets its own frame so V2's median/level only use 2 weeks of history.
        variant_frames = {name: add_target_calendar(add_seasonal_features(origin, h, weeks), horizon) for name, weeks in variants.items()}

        folds = []
        fold_end = supervised["target_at"].max()
        while True:
            test_start = fold_end - timedelta(days=TEST_DAYS)
            baseline_train = supervised[supervised["target_at"] <= test_start]
            if len(baseline_train) < MIN_BASELINE_TRAIN_ROWS:
                break
            window = supervised[(supervised["target_at"] > test_start) & (supervised["target_at"] <= fold_end)]
            test = window.dropna(subset=all_variant_cols)
            fold: dict[str, Any] = {"window_start": test_start.isoformat(), "window_end": fold_end.isoformat(), "test_rows": len(test)}

            fitted = model()
            fitted.fit(baseline_train[FEATURES], baseline_train["target_demand"], cat_features=["station_id"])
            cat = fitted.predict(test[FEATURES])
            fold["baseline_catboost"] = score(test, cat)
            fold["baseline_blend"] = score(test, blend(cat, test["tw_lag_1"].to_numpy()))

            test_keys = test[["station_id", "observed_at"]]
            for name, weeks in variants.items():
                cols = [*FEATURES, *seasonal_columns(weeks)]
                frame = variant_frames[name]
                train = frame[frame["target_at"] <= test_start].dropna(subset=cols)
                if len(train) < MIN_TRAIN_ROWS:
                    fold[f"{name}_catboost"] = None
                    fold[f"{name}_blend"] = None
                    continue
                variant_test = test_keys.merge(frame, on=["station_id", "observed_at"], how="left")
                fitted = model()
                fitted.fit(train[cols], train["target_demand"], cat_features=["station_id"])
                cat = fitted.predict(variant_test[cols])
                fold[f"{name}_catboost"] = score(variant_test, cat)
                fold[f"{name}_blend"] = score(variant_test, blend(cat, variant_test["tw_median"].to_numpy()))
                fold[f"{name}_train_rows"] = len(train)
            folds.append(fold)
            fold_end = test_start
        folds.reverse()
        report[str(horizon)] = folds
    return report


def summarize(report: dict[str, Any]) -> None:
    for horizon, folds in report.items():
        print(f"\nH={horizon} ({len(folds)} folds)")
        for fold in folds:
            parts = [f"{fold['window_start'][:10]}"]
            for key in ("baseline_blend", "V2_blend", "V3_blend"):
                value = fold.get(key)
                parts.append(f"{key}={value:.4f}" if value is not None else f"{key}=n/a")
            print("  " + "  ".join(parts))
        for name in ("V2", "V3"):
            paired = [(f["baseline_blend"], f[f"{name}_blend"]) for f in folds if f.get(f"{name}_blend") is not None]
            if not paired:
                print(f"  {name}: sin folds válidos")
                continue
            deltas = np.array([v - b for b, v in paired])
            print(
                f"  {name} vs producción (blend): {len(paired)} folds, delta medio={deltas.mean():+.4f}, "
                f"deltas={', '.join(f'{d:+.4f}' for d in deltas)}"
            )


def sweep_blend_weights(horizons: list[int], weeks: int = 3, weights: tuple[float, ...] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0)) -> dict[str, Any]:
    """ENSEMBLE_WEIGHT=0.75 was tuned for the current feature set; give the variant its own tuned weight
    before judging it. Tries both naive anchors: last week only, or the multi-week median."""
    origin = add_origin_features(load_training_data())
    cols = [*FEATURES, *seasonal_columns(weeks)]
    report: dict[str, Any] = {}
    for horizon in horizons:
        frame = add_target_calendar(add_seasonal_features(origin, horizon // 15, weeks), horizon)
        fold_end = frame["target_at"].max()
        rows = []
        while True:
            test_start = fold_end - timedelta(days=TEST_DAYS)
            train = frame[frame["target_at"] <= test_start].dropna(subset=cols)
            if len(train) < MIN_TRAIN_ROWS:
                break
            test = frame[(frame["target_at"] > test_start) & (frame["target_at"] <= fold_end)].dropna(subset=cols)
            # Production model on the exact same test rows (data keeps growing, so recompute here
            # instead of comparing against an older run with shifted windows).
            baseline_train = frame[frame["target_at"] <= test_start]
            production = model()
            production.fit(baseline_train[FEATURES], baseline_train["target_demand"], cat_features=["station_id"])
            production_pred = blend(production.predict(test[FEATURES]), test["tw_lag_1"].to_numpy())
            fitted = model()
            fitted.fit(train[cols], train["target_demand"], cat_features=["station_id"])
            cat = fitted.predict(test[cols])
            row = {"window_start": test_start.isoformat(), "production": score(test, production_pred)}
            for anchor in ("tw_lag_1", "tw_median"):
                naive = test[anchor].to_numpy()
                for weight in weights:
                    blended = np.clip(weight * cat + (1 - weight) * naive, 0, None)
                    row[f"{anchor}@{weight}"] = score(test, blended)
            rows.append(row)
            fold_end = test_start
        report[str(horizon)] = rows
    return report


def main() -> None:
    report = run([15, 30, 45, 60])
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "seasonal_features_backtest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summarize(report)


if __name__ == "__main__":
    main()
