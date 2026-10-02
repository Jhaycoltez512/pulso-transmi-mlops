"""Backtest refinements of the 4h-periodic copy on the live regime (read-only).

Since 2026-09-18 ~05:00 UTC demand oscillates with a 4h period; the production ensemble ends up
submitting ~the plain copy from 4h earlier (~90%). Rivals score 92-93% per cycle. This compares,
on hourly origins like the real cycles, cheap refinements of that copy:

- averaging several past periods (mean / median / exponentially weighted), to cancel noise;
- level adjustment: scale by the ratio of the last full period to the one before (amplitude drift);
- daily-profile adjustment: x comparable-day(t+h) / comparable-day(t+h-4h);
- a per-horizon ridge on the last 6 periods, refit walk-forward on regime data only.

Score per cycle = 1 - WAPE over the cycle's 48 predictions (what the leaderboard uses; matched it
exactly on 11 cycles), plus the mean per-station accuracy. Also scores what was actually
submitted, from the predictions table.

    python scripts/backtest_periodic.py
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from load_supabase import SupabaseLoader, load_dotenv
from train_baseline import load_training_data

import os

PERIOD = 16  # 4h in 15-min steps
HORIZONS = (1, 2, 3, 4)
REGIME_START = pd.Timestamp("2026-09-18T05:00:00Z")
EVAL_START = REGIME_START + timedelta(hours=24)  # 6 full periods of history for every variant


def series(data: pd.DataFrame) -> pd.DataFrame:
    wide = data.pivot_table(index="observed_at", columns="station_id", values="demand", aggfunc="last").sort_index()
    return wide.asfreq("15min")


def candidates(y: pd.DataFrame, t: pd.Timestamp, steps: int, ridge: dict[int, np.ndarray] | None) -> dict[str, pd.Series]:
    idx = y.index.get_loc(t)
    target = idx + steps
    lag = lambda k: y.iloc[target - PERIOD * k]  # noqa: E731  value k periods before the target
    lags = [lag(k) for k in range(1, 7)]
    out = {"copia 4h": lags[0]}
    for k in (2, 3, 4, 6):
        out[f"media {k} periodos"] = pd.concat(lags[:k], axis=1).mean(axis=1)
    for k in (3, 5):
        out[f"mediana {k} periodos"] = pd.concat(lags[:k], axis=1).median(axis=1)
    for decay in (0.5, 0.7):
        w = np.array([decay ** k for k in range(6)])
        out[f"ewm periodos d={decay}"] = (pd.concat(lags, axis=1) * w).sum(axis=1) / w.sum()
    last = y.iloc[idx - PERIOD + 1: idx + 1].mean()
    before = y.iloc[idx - 2 * PERIOD + 1: idx - PERIOD + 1].mean()
    ratio = (last / before).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.5, 2.0)
    for base in ("media 4 periodos", "ewm periodos d=0.7"):
        out[f"{base} x nivel"] = out[base] * ratio
        out[f"{base} x nivel^0.5"] = out[base] * np.sqrt(ratio)
    # anchor on the latest observation: shift the periodic shape by the current deviation
    resid = y.iloc[idx] - y.iloc[idx - PERIOD]
    for decay in (0.5, 0.8):
        out[f"copia 4h + desvio actual d={decay}"] = (lags[0] + resid * decay ** steps).clip(lower=0)
        out[f"media 4 periodos + desvio actual d={decay}"] = (out["media 4 periodos"] + (y.iloc[idx] - pd.concat([y.iloc[idx - PERIOD * k] for k in range(1, 5)], axis=1).mean(axis=1)) * decay ** steps).clip(lower=0)
    if ridge is not None and steps in ridge:
        X = np.column_stack([np.ones(len(lags[0]))] + [l.to_numpy() for l in lags] + [y.iloc[idx].to_numpy(), (y.iloc[idx] - y.iloc[idx - PERIOD]).to_numpy()])
        out["ridge 6 periodos"] = pd.Series(np.clip(X @ ridge[steps], 0, None), index=lags[0].index)
    return out


def fit_ridge(y: pd.DataFrame, origins: list[pd.Timestamp], lam: float = 1e3) -> dict[int, np.ndarray]:
    """Per-horizon ridge (pooled stations) on the 6 period lags + current value + current deviation."""
    coefs = {}
    for steps in HORIZONS:
        rows, targets = [], []
        for t in origins:
            idx = y.index.get_loc(t)
            target = idx + steps
            if target >= len(y) or y.iloc[target].isna().any():
                continue
            lags = [y.iloc[target - PERIOD * k].to_numpy() for k in range(1, 7)]
            rows.append(np.column_stack([np.ones(y.shape[1])] + lags + [y.iloc[idx].to_numpy(), (y.iloc[idx] - y.iloc[idx - PERIOD]).to_numpy()]))
            targets.append(y.iloc[target].to_numpy())
        if not rows:
            continue
        X, Y = np.vstack(rows), np.concatenate(targets)
        ok = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
        X, Y = X[ok], Y[ok]
        reg = lam * np.eye(X.shape[1])
        reg[0, 0] = 0
        coefs[steps] = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    return coefs


def main() -> None:
    load_dotenv()
    y = series(load_training_data())
    last_obs = y.dropna(how="any").index.max()
    origins = [t for t in pd.date_range(EVAL_START, last_obs - timedelta(hours=1), freq="1h")]
    regime_origins = list(pd.date_range(REGIME_START + timedelta(hours=24), last_obs - timedelta(hours=1), freq="15min"))
    print(f"evaluating {len(origins)} hourly origins {origins[0]} -> {origins[-1]}")

    loader = SupabaseLoader(os.environ["SUPABASE_URL"], os.getenv("SUPABASE_SECRET_KEY") or os.environ["SUPABASE_KEY"])
    sent = pd.DataFrame(loader.select_all("predictions", {
        "select": "station_id,target_at,horizon_minutes,predicted_demand,forecast_runs!inner(data_cutoff)",
        "target_at": f"gt.{EVAL_START.isoformat()}",
    }, order="id.asc"))
    loader.close()
    if not sent.empty:
        sent["cutoff"] = pd.to_datetime(sent["forecast_runs"].map(lambda r: r["data_cutoff"]), utc=True)
        sent["target_at"] = pd.to_datetime(sent["target_at"], utc=True)
        sent = sent.groupby(["cutoff", "target_at", "station_id"], as_index=False)["predicted_demand"].last()

    records = []
    for t in origins:
        train = [o for o in regime_origins if o + timedelta(hours=1) <= t]  # targets already observed at t
        ridge = fit_ridge(y, train) if len(train) >= 20 else None
        for steps in HORIZONS:
            target_at = t + timedelta(minutes=15 * steps)
            actual = y.loc[target_at]
            for name, pred in candidates(y, t, steps, ridge).items():
                for station in y.columns:
                    records.append((t, steps * 15, station, name, float(pred[station]), float(actual[station])))
            if not sent.empty:
                s = sent.loc[(sent["cutoff"] == t) & (sent["target_at"] == target_at)].set_index("station_id")["predicted_demand"]
                for station, value in s.items():
                    records.append((t, steps * 15, station, "enviado (produccion)", float(value), float(actual[station])))
    frame = pd.DataFrame(records, columns=["origin", "h", "station", "variant", "pred", "actual"])
    frame["err"] = (frame["pred"] - frame["actual"]).abs()

    per_cycle = frame.groupby(["variant", "origin"]).apply(lambda g: 100 * max(0.0, 1 - g["err"].sum() / g["actual"].sum()), include_groups=False)
    per_station = frame.groupby(["variant", "origin", "station"]).apply(lambda g: 100 * max(0.0, 1 - g["err"].sum() / g["actual"].sum()), include_groups=False).groupby(["variant", "origin"]).mean()
    per_h = frame.groupby(["variant", "h"]).apply(lambda g: 100 * max(0.0, 1 - g["err"].sum() / g["actual"].sum()), include_groups=False).unstack()
    summary = pd.DataFrame({
        "ciclo (1-WAPE) medio": per_cycle.groupby("variant").mean(),
        "min": per_cycle.groupby("variant").min(),
        "max": per_cycle.groupby("variant").max(),
        "media por estacion": per_station.groupby("variant").mean(),
        "ciclos": per_cycle.groupby("variant").size(),
    }).join(per_h.add_prefix("h")).sort_values("ciclo (1-WAPE) medio", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
