"""Walk-forward backtest: pull predictions back to the usual profile during short events.

The production model follows each station's recent level, which is what makes it robust to
sustained level changes (05100 down to 0.2x, 05000 up 3x) -- but the competition also injects
short bursts (2026-09-18 05:00-07:00 UTC: 07111 x5 then collapsing, 05000/02300 x2.5). There
the model reacts late and, once the burst ends, keeps predicting high because its 4h level
still contains the burst.

Candidate fix, applied on top of the production predictions: when a station's last hour is far
from the same hour on the previous comparable day (|ratio| > threshold), blend the prediction
towards that day's demand at the target time, more for longer horizons.

The comparable day is the most recent one of the same type, so a level change that has already
lasted a day is treated as the new normal while weekday/weekend profiles aren't mixed: Tue-Fri
use the day before, Monday uses Friday, Saturday and Sunday use the same day last week. (A
plain "yesterday" flagged 12-19% of ordinary rows, every Monday against a Sunday.)

    weight_on_reference = alpha * horizon / 60      (only when the anomaly flag is on)

Read-only: loads observations from Supabase and prints tables.

    python scripts/backtest_event_reversion.py [horizon] [start_utc]
"""

from __future__ import annotations

import sys
from datetime import timedelta

import numpy as np
import pandas as pd

from train_baseline import load_training_data, station_metrics
from train_catboost_direct import add_origin_features, add_target_calendar, fit_model, predict_demand

ORIGIN_STEP_HOURS = 12
DEFAULT_START = "2026-09-12T05:00:00Z"
DAY = timedelta(days=1)
PERIODS_PER_DAY = 96
EVENT_WINDOW = (pd.Timestamp("2026-09-18T05:00:00Z"), pd.Timestamp("2026-09-18T08:00:00Z"))

# name: (anomaly threshold as a ratio, or None = always blend; alpha = weight on the reference at h60)
VARIANTS = {
    "siempre, alpha=0.25": (None, 0.25),
    "siempre, alpha=0.5": (None, 0.5),
    "anomalia >x1.5, alpha=0.5": (1.5, 0.5),
    "anomalia >x1.5, alpha=1.0": (1.5, 1.0),
    "anomalia >x2.0, alpha=0.5": (2.0, 0.5),
    "anomalia >x2.0, alpha=1.0": (2.0, 1.0),
    "anomalia >x3.0, alpha=1.0": (3.0, 1.0),
}


def demand_at(data: pd.DataFrame, stations: pd.Series, times: pd.Series) -> np.ndarray:
    lookup = data[["station_id", "observed_at", "demand"]].rename(columns={"observed_at": "t", "demand": "value"})
    keys = pd.DataFrame({"station_id": stations.to_numpy(), "t": times.to_numpy()})
    return keys.merge(lookup, on=["station_id", "t"], how="left")["value"].to_numpy(dtype=float)


def reference_lag_days(times: pd.Series) -> np.ndarray:
    """Days back to the previous comparable day (see module docstring), by local weekday."""
    weekday = times.dt.tz_convert("America/Bogota").dt.dayofweek.to_numpy()
    return np.select([weekday == 0, weekday >= 5], [3, 7], default=1)


def add_anomaly_inputs(data: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Last-hour level vs the same hour on the comparable day, and that day's demand at the target."""
    out = frame.copy()
    hourly = data.sort_values(["station_id", "observed_at"]).copy()
    hourly["level4"] = hourly.groupby("station_id")["demand"].transform(lambda s: s.rolling(4, min_periods=4).mean())
    level = hourly[["station_id", "observed_at", "level4"]]
    out = out.merge(level, on=["station_id", "observed_at"], how="left")
    origin_ref = out["observed_at"] - pd.to_timedelta(reference_lag_days(out["observed_at"]), unit="D")
    ref_level = pd.DataFrame({"station_id": out["station_id"].to_numpy(), "observed_at": origin_ref.to_numpy()}).merge(
        level, on=["station_id", "observed_at"], how="left")["level4"].to_numpy()
    out["anomaly_ratio"] = pd.Series(out["level4"].to_numpy() / ref_level).replace([np.inf, -np.inf], np.nan).to_numpy()
    target_ref = out["target_at"] - pd.to_timedelta(reference_lag_days(out["target_at"]), unit="D")
    out["reference_at_target"] = demand_at(data, out["station_id"], target_ref)
    return out


def apply_reversion(frame: pd.DataFrame, base: np.ndarray, horizon: int, threshold: float | None, alpha: float) -> np.ndarray:
    weight = alpha * horizon / 60
    reference = frame["reference_at_target"].to_numpy()
    ratio = frame["anomaly_ratio"].to_numpy()
    if threshold is None:
        flagged = np.ones(len(frame), dtype=bool)
    else:
        flagged = np.abs(np.log(np.where(ratio > 0, ratio, np.nan))) > np.log(threshold)
    use = flagged & ~np.isnan(reference)
    return np.where(use, (1 - weight) * base + weight * reference, base)


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
    result = add_anomaly_inputs(data, pd.concat(pieces, ignore_index=True))
    for name, (threshold, alpha) in VARIANTS.items():
        result[name] = apply_reversion(result, result["produccion"].to_numpy(), horizon, threshold, alpha)

    last_day = result["target_at"] > end - DAY
    event = (result["target_at"] >= EVENT_WINDOW[0]) & (result["target_at"] < EVENT_WINDOW[1])
    rows = []
    for name in ["produccion", *VARIANTS]:
        row = {"variante": name, "todo": accuracy(result, name), "ultimas 24h": accuracy(result.loc[last_day], name)}
        if event.any():
            row["evento 18-sep"] = accuracy(result.loc[event], name)
        for day, part in result.groupby(result["target_at"].dt.tz_convert("America/Bogota").dt.date):
            row[str(day)[5:]] = accuracy(part, name)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("todo", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    ratio = result["anomaly_ratio"]
    flagged = {t: float((np.abs(np.log(ratio[ratio > 0])) > np.log(t)).mean() * 100) for t in (1.5, 2.0, 3.0)}
    print(f"\n=== h={horizon}: accuracy media por estacion (%), {len(result)} filas, {start.date()} -> {end} ===")
    print("filas marcadas como anomalia: " + ", ".join(f">x{t}: {pct:.1f}%" for t, pct in flagged.items()))
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
