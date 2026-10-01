"""Walk-forward comparison of post-model adjustments on real data (scripts/forecast_adjustments.py).

Every ORIGIN_STEP_HOURS the production model is trained on everything known at that origin and
predicts the next ORIGIN_STEP_HOURS of origins, as production would. On top of those
predictions it compares: the simple references (persistence, comparable day), the mild blend,
the growth cap, both together, and adaptive expert ensembles (model / persistence / comparable
day weighted by recent inverse error), with different windows, sharpness and pooling.

As in production, ensemble weights only learn from hourly origins (one cycle per hour), and
only from targets already observed at each origin; scores are on hourly origins too.

Read-only: loads observations from Supabase and prints tables.

    python scripts/backtest_adjustments.py [horizon] [start_utc]
"""

from __future__ import annotations

import sys
from datetime import timedelta

import numpy as np
import pandas as pd

from forecast_adjustments import adjust, blend_with_reference, cap_growth, combine, expert_weights, reference_inputs
from train_baseline import load_training_data, station_metrics
from train_catboost_direct import add_origin_features, add_target_calendar, fit_model, predict_demand

ORIGIN_STEP_HOURS = 12
DEFAULT_START = "2026-09-12T05:00:00Z"
EVENT_START = pd.Timestamp("2026-09-18T05:00:00Z")
ENSEMBLES = {
    # name: (model expert column, pool by, window, power)
    "ensamble est+h 6h p1": ("produccion", ["station_id"], timedelta(hours=6), 1.0),
    "ensamble est+h 6h p2": ("produccion", ["station_id"], timedelta(hours=6), 2.0),
    "ensamble est+h 24h p1": ("produccion", ["station_id"], timedelta(hours=24), 1.0),
    "ensamble est+h 24h p2": ("produccion", ["station_id"], timedelta(hours=24), 2.0),
    "ensamble global 6h p2": ("produccion", [], timedelta(hours=6), 2.0),
    "ensamble global 24h p2": ("produccion", [], timedelta(hours=24), 2.0),
    "ensamble(mezcla+tope) est+h 6h p2": ("mezcla+tope", ["station_id"], timedelta(hours=6), 2.0),
    "ensamble(mezcla+tope) est+h 24h p2": ("mezcla+tope", ["station_id"], timedelta(hours=24), 2.0),
}


def accuracy(frame: pd.DataFrame, column: str) -> float:
    scored = frame[["station_id", "target_demand"]].rename(columns={"target_demand": "demand"})
    return station_metrics(scored, frame[column].to_numpy())["mean_station_accuracy"]


def main() -> None:
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    start = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_START)
    data = add_origin_features(load_training_data())
    supervised = add_target_calendar(data, horizon)
    end = supervised["observed_at"].max()
    pieces = []
    origin = start
    while origin < end:
        train = supervised.loc[supervised["target_at"] <= origin]
        test = supervised.loc[(supervised["observed_at"] >= origin) & (supervised["observed_at"] < origin + timedelta(hours=ORIGIN_STEP_HOURS))].copy()
        if not test.empty:
            test["produccion"] = np.clip(predict_demand(fit_model(train), test), 0, None)
            pieces.append(test)
            print(f"h={horizon} origin={origin.isoformat()} rows={len(test)}", flush=True)
        origin += timedelta(hours=ORIGIN_STEP_HOURS)
    result = reference_inputs(data, pd.concat(pieces, ignore_index=True)[["station_id", "observed_at", "target_at", "target_demand", "produccion"]])
    hourly = result.loc[result["observed_at"].dt.minute == 0].reset_index(drop=True)

    model = hourly["produccion"].to_numpy()
    hourly["persistencia"] = hourly["persistence"]
    hourly["dia comparable"] = hourly["reference"]
    hourly["mezcla"] = blend_with_reference(model, hourly["reference"].to_numpy(), horizon)
    hourly["tope"] = cap_growth(model, hourly["level_now"].to_numpy(), hourly["reference_level"].to_numpy(), hourly["reference"].to_numpy())
    hourly["mezcla+tope"] = adjust(hourly, model, horizon)
    hourly["mezcla+tope x2"] = blend_with_reference(
        cap_growth(model, hourly["level_now"].to_numpy(), hourly["reference_level"].to_numpy(), hourly["reference"].to_numpy(), threshold=2.0),
        hourly["reference"].to_numpy(), horizon)

    for name, (model_column, keys, window, power) in ENSEMBLES.items():
        experts = pd.DataFrame({"model": hourly[model_column], "persistence": hourly["persistence"], "comparable": hourly["reference"]})
        history = pd.concat([hourly[["station_id", "target_at"]], experts], axis=1).assign(actual=hourly["target_demand"])
        weights = expert_weights(history, hourly[["station_id", "observed_at"]], keys, window, power)
        hourly[name] = combine(experts, weights)
        if name == "ensamble est+h 24h p2":
            event = hourly["target_at"] >= EVENT_START
            print(f"pesos medios {name}: normal {weights.loc[~event.to_numpy()].mean().round(2).to_dict()} | evento {weights.loc[event.to_numpy()].mean().round(2).to_dict()}")

    candidates = ["produccion", "persistencia", "dia comparable", "mezcla", "tope", "mezcla+tope", "mezcla+tope x2", *ENSEMBLES]
    last_day = hourly["target_at"] > end - timedelta(days=1)
    event = hourly["target_at"] >= EVENT_START
    rows = []
    for name in candidates:
        valid = hourly[name].notna()
        frame = hourly.loc[valid]
        row = {
            "variante": name, "todo": accuracy(frame, name),
            "ultimas 24h": accuracy(frame.loc[last_day[valid]], name),
            "sin evento": accuracy(frame.loc[~event[valid]], name),
            "evento": accuracy(frame.loc[event[valid]], name) if event[valid].any() else np.nan,
        }
        for day, part in frame.groupby(frame["target_at"].dt.tz_convert("America/Bogota").dt.date):
            row[str(day)[5:]] = accuracy(part, name)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("todo", ascending=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== h={horizon}: accuracy media por estacion (%), {len(hourly)} filas horarias, {start.date()} -> {end} ===")
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
