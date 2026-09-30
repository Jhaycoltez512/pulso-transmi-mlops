"""Walk-forward backtest over the production period where station demand changed.

From mid-September the competition moved demand between stations (05100 falling to
~0.2x of the previous week, 05000/02300 rising 2-3x, 03000 to ~0.4x, 07111 ~1.4x), and
production accuracy fell from ~85% to ~65%. This replays that period with real data:
every ORIGIN_STEP_HOURS a model is trained on everything known at that origin and scored
on the next ORIGIN_STEP_HOURS of forecast origins, exactly as production would have.
Predictions from consecutive windows form one out-of-sample series, so the online bias
correction can be replayed on it without leaking in-sample fits.

Read-only: loads observations/context from Supabase and prints tables, writes nothing.

    python scripts/backtest_production_shift.py [horizon] [start_utc]
"""

from __future__ import annotations

import sys
from datetime import timedelta

import numpy as np
import pandas as pd

from run_forecast_cycle import bias_scale
from train_baseline import load_training_data, station_metrics
from train_catboost_direct import (
    CALENDAR_FEATURES, DEMAND_FEATURES, RAW_FEATURES, add_origin_features, add_target_calendar,
    blend_predictions, model, naive_prediction_at_target,
)

ORIGIN_STEP_HOURS = 12
DEFAULT_START = "2026-09-12T05:00:00Z"  # local midnight, the day before the first changes
SHORT_DEMAND_FEATURES = ["current_demand", "lag_1", "lag_2", "lag_4", "lag_8", "rolling_mean_4", "rolling_mean_12"]


def normalised(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """Level and norm_* features for a given level window (production uses 16)."""
    out = data.copy()
    grouped = out.groupby("station_id")["demand"]
    out["level"] = grouped.transform(lambda s: s.rolling(window, min_periods=window).mean()).clip(lower=1.0)
    for feature in DEMAND_FEATURES:
        out[f"norm_{feature}"] = out[feature] / out["level"]
    recent = grouped.transform(lambda s: s.rolling(16, min_periods=16).sum())
    before = grouped.transform(lambda s: s.shift(672).rolling(16, min_periods=16).sum())
    out["week_ratio_raw"] = recent / before
    return out


VARIANTS = {
    # name: (feature list, level-normalised?)
    "raw": (RAW_FEATURES, False),
    "level16": (["station_id", *CALENDAR_FEATURES, *[f"norm_{f}" for f in DEMAND_FEATURES]], True),
    "level16_short": (["station_id", *CALENDAR_FEATURES, *[f"norm_{f}" for f in SHORT_DEMAND_FEATURES]], True),
}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], ratio: bool) -> np.ndarray:
    fitted = model()
    if ratio:
        fitted.fit(train[features], train["target_demand"] / train["level"], cat_features=["station_id"], sample_weight=train["level"])
        return fitted.predict(test[features]) * test["level"].to_numpy()
    fitted.fit(train[features], train["target_demand"], cat_features=["station_id"])
    return fitted.predict(test[features])


def correction(frame: pd.DataFrame, column: str, horizon_periods: int, scope: str, alpha: float, clip: tuple[float, float]) -> np.ndarray:
    """Replay the online bias correction: actual/pred over the last 16 targets known at each origin."""
    work = frame.sort_values(["station_id", "observed_at"])
    if scope == "global":
        totals = work.groupby("observed_at")[["target_demand", column]].sum().sort_index()
        factor = (totals["target_demand"].rolling(16, min_periods=16).sum() / totals[column].rolling(16, min_periods=16).sum()).shift(horizon_periods)
        factors = work["observed_at"].map(factor)
    else:
        grouped = work.groupby("station_id")
        actual = grouped["target_demand"].transform(lambda s: s.rolling(16, min_periods=16).sum().shift(horizon_periods))
        predicted = grouped[column].transform(lambda s: s.rolling(16, min_periods=16).sum().shift(horizon_periods))
        factors = actual / predicted
    scale = np.clip(1 + alpha * (factors.to_numpy() - 1), *clip)
    corrected = work[column].to_numpy() * np.where(np.isnan(scale), 1.0, scale)
    return pd.Series(corrected, index=work.index).reindex(frame.index).to_numpy()


def accuracy(frame: pd.DataFrame, column: str) -> float:
    scored = frame[["station_id", "target_demand"]].rename(columns={"target_demand": "demand"})
    return station_metrics(scored, frame[column].to_numpy())["mean_station_accuracy"]


def main() -> None:
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    start = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_START)
    base = add_origin_features(load_training_data())
    supervised = add_target_calendar(normalised(base, 16), horizon)
    naive = naive_prediction_at_target(base, supervised)
    ratio = supervised["week_ratio_raw"].fillna(1.0).to_numpy()
    supervised["naive"] = naive
    supervised["naive_adj_prod"] = naive * np.clip(ratio, 0.5, 2.0)
    supervised["naive_adj_wide"] = naive * np.clip(ratio, 0.1, 5.0)

    end = supervised["observed_at"].max()
    pieces = []
    origin = start
    while origin < end:
        train = supervised.loc[supervised["target_at"] <= origin]
        test = supervised.loc[(supervised["observed_at"] >= origin) & (supervised["observed_at"] < origin + timedelta(hours=ORIGIN_STEP_HOURS))].copy()
        if not test.empty:
            # production before this change: raw features, fitted on the train split only (14 days held out)
            prod_train = train.loc[train["target_at"] <= train["target_at"].max() - timedelta(days=14)]
            test["prod_old"] = blend_predictions(fit_predict(prod_train, test, RAW_FEATURES, False), test["naive"].to_numpy())
            for name, (features, ratio_model) in VARIANTS.items():
                test[f"cb_{name}"] = fit_predict(train, test, features, ratio_model)
            pieces.append(test)
            print(f"h={horizon} origin={origin.isoformat()} rows={len(test)}", flush=True)
        origin += timedelta(hours=ORIGIN_STEP_HOURS)
    result = pd.concat(pieces)

    candidates = {"prod_old": result["prod_old"].to_numpy()}
    for name in VARIANTS:
        cb = result[f"cb_{name}"].to_numpy()
        candidates[f"{name}_solo"] = np.clip(cb, 0, None)
        candidates[f"{name}+naive"] = blend_predictions(cb, result["naive"].to_numpy())
        candidates[f"{name}+adjnaive_prod"] = blend_predictions(cb, result["naive_adj_prod"].to_numpy())
        candidates[f"{name}+adjnaive_wide"] = blend_predictions(cb, result["naive_adj_wide"].to_numpy())
    h = horizon // 15
    rows = []
    for name, values in candidates.items():
        result[name] = values
        variants = {"none": values}
        variants["global_a0.5"] = correction(result, name, h, "global", 0.5, (0.8, 1.25))
        variants["station_a0.5"] = correction(result, name, h, "station", 0.5, (0.8, 1.25))
        variants["station_a1.0"] = correction(result, name, h, "station", 1.0, (0.5, 2.0))
        for corr, predicted in variants.items():
            column = f"{name}|{corr}"
            result[column] = predicted
            row = {"candidate": name, "correction": corr, "overall": accuracy(result, column)}
            for day, part in result.groupby(result["target_at"].dt.tz_convert("America/Bogota").dt.date):
                row[str(day)[5:]] = accuracy(part, column)
            rows.append(row)
    table = pd.DataFrame(rows).sort_values("overall", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== h={horizon}: mean-station accuracy (%) by target day, {len(result)} rows ===")
    print(table.round(2).to_string(index=False))

    best = f"{table.iloc[0]['candidate']}|{table.iloc[0]['correction']}"
    per_station = pd.DataFrame({
        column: {station: accuracy(part, column) for station, part in result.groupby("station_id")}
        for column in ["prod_old|global_a0.5", best]
    })
    print(f"\n=== h={horizon}: per-station accuracy, production vs best ({best}) ===")
    print(per_station.round(1).to_string())


if __name__ == "__main__":
    main()
