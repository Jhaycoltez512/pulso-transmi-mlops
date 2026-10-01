"""Walk-forward backtest: which training data / memory suits manually injected drift?

The production model (level-normalised, short memory, trained on all history) handles
sustained station level changes, but the competition also injects drift by hand -- e.g.
a one-hour surge at 05:00-06:00 UTC on 2026-09-18 (09122 x4, 03000 x3.5, 10009 near zero)
seen on no earlier day. This compares, on the real period since the changes began:

- how much history to train on (all / last 14 days / last 7 days),
- recency weighting (half-life 3 or 7 days, multiplied into the level weight),
- adding the 1-day memory back (lag_96, rolling_mean_96), which could learn a new pattern
  that repeats daily but anchored stations to the old pattern in the earlier backtest.

Every ORIGIN_STEP_HOURS a model is trained on everything known at that origin and scored
on the next ORIGIN_STEP_HOURS of forecast origins, as production would have. Read-only:
loads observations from Supabase and prints tables.

    python scripts/backtest_training_window.py [horizon] [start_utc]
"""

from __future__ import annotations

import sys
from datetime import timedelta

import numpy as np
import pandas as pd

from train_baseline import load_training_data, station_metrics
from train_catboost_direct import FEATURES, add_origin_features, add_target_calendar, model

ORIGIN_STEP_HOURS = 12
DEFAULT_START = "2026-09-12T05:00:00Z"
WITH_DAY_MEMORY = [*FEATURES, "norm_lag_96", "norm_rolling_mean_96"]

# name: (features, training window in days or None for all, recency half-life in days or None)
VARIANTS = {
    "produccion (todo)": (FEATURES, None, None),
    "ultimos 14 dias": (FEATURES, 14, None),
    "ultimos 7 dias": (FEATURES, 7, None),
    "recencia hl=7d": (FEATURES, None, 7),
    "recencia hl=3d": (FEATURES, None, 3),
    "+lag 1 dia": (WITH_DAY_MEMORY, None, None),
    "+lag 1 dia, 14 dias": (WITH_DAY_MEMORY, 14, None),
}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], half_life: float | None, origin: pd.Timestamp) -> np.ndarray:
    """Same target/weighting as production fit_model, with an optional recency factor."""
    weight = train["level"].to_numpy()
    if half_life is not None:
        age_days = (origin - train["target_at"]).dt.total_seconds().to_numpy() / 86_400
        weight = weight * np.power(0.5, age_days / half_life)
    fitted = model()
    fitted.fit(train[features], train["target_demand"] / train["level"], cat_features=["station_id"], sample_weight=weight)
    return np.clip(fitted.predict(test[features]) * test["level"].to_numpy(), 0, None)


def accuracy(frame: pd.DataFrame, column: str) -> float:
    scored = frame[["station_id", "target_demand"]].rename(columns={"target_demand": "demand"})
    return station_metrics(scored, frame[column].to_numpy())["mean_station_accuracy"]


def main() -> None:
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    start = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_START)
    supervised = add_target_calendar(add_origin_features(load_training_data()), horizon)
    supervised = supervised.dropna(subset=WITH_DAY_MEMORY)
    end = supervised["observed_at"].max()
    pieces = []
    origin = start
    while origin < end:
        seen = supervised.loc[supervised["target_at"] <= origin]
        test = supervised.loc[(supervised["observed_at"] >= origin) & (supervised["observed_at"] < origin + timedelta(hours=ORIGIN_STEP_HOURS))].copy()
        if not test.empty:
            for name, (features, window_days, half_life) in VARIANTS.items():
                train = seen if window_days is None else seen.loc[seen["target_at"] > origin - timedelta(days=window_days)]
                test[name] = fit_predict(train, test, features, half_life, origin)
            pieces.append(test)
            print(f"h={horizon} origin={origin.isoformat()} rows={len(test)}", flush=True)
        origin += timedelta(hours=ORIGIN_STEP_HOURS)
    result = pd.concat(pieces)

    last_day = result["target_at"] > end - timedelta(days=1)
    rows = []
    for name in VARIANTS:
        row = {"variante": name, "todo": accuracy(result, name), "ultimas 24h": accuracy(result.loc[last_day], name)}
        for day, part in result.groupby(result["target_at"].dt.tz_convert("America/Bogota").dt.date):
            row[str(day)[5:]] = accuracy(part, name)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("todo", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== h={horizon}: accuracy media por estacion (%), {len(result)} filas, {start.date()} -> {end} ===")
    print(table.round(2).to_string(index=False))

    per_station = pd.DataFrame({
        name: {station: accuracy(part, name) for station, part in result.groupby("station_id")} for name in VARIANTS
    })
    print(f"\n=== h={horizon}: accuracy por estacion (%), todo el periodo ===")
    print(per_station.round(1).to_string())


if __name__ == "__main__":
    main()
