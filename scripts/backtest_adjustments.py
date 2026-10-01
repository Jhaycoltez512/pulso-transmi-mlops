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

from forecast_adjustments import PERIODIC_EXPERTS, adjust, blend_with_reference, combine, expert_frame, expert_weights, reference_inputs
from train_baseline import load_training_data, station_metrics
from train_catboost_direct import add_origin_features, add_target_calendar, fit_model, predict_demand

ORIGIN_STEP_HOURS = 12
DEFAULT_START = "2026-09-12T05:00:00Z"
EVENT_START = pd.Timestamp("2026-09-18T05:00:00Z")
CAP_THRESHOLDS = (2.5,)
OLD = ("model", "persistence", "comparable")
PERIODIC = ("model", *PERIODIC_EXPERTS)
ALL = (*OLD, *PERIODIC_EXPERTS)
ENSEMBLES = {
    # name: (experts, pool by, window, power)
    "ens viejo global 6h p3": (OLD, [], timedelta(hours=6), 3.0),
    "ens modelo+periodicos global 6h p3": (PERIODIC, [], timedelta(hours=6), 3.0),
    "ens modelo+periodicos global 3h p3": (PERIODIC, [], timedelta(hours=3), 3.0),
    "ens modelo+periodicos global 6h p6": (PERIODIC, [], timedelta(hours=6), 6.0),
    "ens modelo+periodicos est+h 6h p3": (PERIODIC, ["station_id"], timedelta(hours=6), 3.0),
    "ens todos global 6h p3": (ALL, [], timedelta(hours=6), 3.0),
    "ens todos global 3h p3": (ALL, [], timedelta(hours=3), 3.0),
    "ens todos est+h 6h p3": (ALL, ["station_id"], timedelta(hours=6), 3.0),
    "ens modelo+lag4h global 6h p3": (("model", "lag_4h"), [], timedelta(hours=6), 3.0),
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
    event = hourly["target_at"] >= EVENT_START
    late = hourly["target_at"] >= EVENT_START + timedelta(hours=4)
    hourly["persistencia"] = hourly["persistence"]
    hourly["dia comparable"] = hourly["reference"]
    hourly["mezcla"] = blend_with_reference(model, hourly["reference"].to_numpy(), horizon)
    for threshold in CAP_THRESHOLDS:
        hourly[f"mezcla+tope x{threshold}"] = adjust(hourly, model, horizon, threshold=threshold)
    for name in PERIODIC_EXPERTS:
        hourly[f"copia {name}"] = hourly[name]

    for name, (experts, keys, window, power) in ENSEMBLES.items():
        table = expert_frame(hourly, model, experts)
        history = pd.concat([hourly[["station_id", "target_at"]], table], axis=1).assign(actual=hourly["target_demand"])
        weights = expert_weights(history, hourly[["station_id", "observed_at"]], keys, window, power, experts)
        hourly[name] = combine(table, weights)
        if name == "ens todos global 6h p3":
            for label, part in (("normal", ~event), ("evento 05-09h", event & ~late), ("evento 09h+", late)):
                print(f"pesos medios {name} [{label}]: {weights.loc[part.to_numpy()].mean().round(2).to_dict()}")

    candidates = ["produccion", "persistencia", "dia comparable", "mezcla", *(f"mezcla+tope x{t}" for t in CAP_THRESHOLDS),
                  *(f"copia {n}" for n in PERIODIC_EXPERTS), *ENSEMBLES]
    last_day = hourly["target_at"] > end - timedelta(days=1)
    rows = []
    for name in candidates:
        valid = hourly[name].notna()
        frame = hourly.loc[valid]
        row = {
            "variante": name, "todo": accuracy(frame, name),
            "ultimas 24h": accuracy(frame.loc[last_day[valid]], name),
            "sin evento": accuracy(frame.loc[~event[valid]], name),
            "evento": accuracy(frame.loc[event[valid]], name) if event[valid].any() else np.nan,
            "evento 05-09h": accuracy(frame.loc[(event & ~late)[valid]], name) if (event & ~late)[valid].any() else np.nan,
            "evento 09h+": accuracy(frame.loc[late[valid]], name) if late[valid].any() else np.nan,
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
