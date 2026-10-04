"""Read-only experiment: periodic forecasters vs what we submitted, on the 4h and the 8h regimes.

Every candidate predicts y[t + k] (k = 1..4 steps of 15 min) from data up to the origin t only.
Scored like the leaderboard: 1 - WAPE per hourly origin (48 predictions) then averaged per regime.
"auto" candidates pick the period themselves: the P (2-12h, 15-min grid) whose 3-point smoothed copy
had the lowest WAPE over the last AUTO_LOOKBACK at the origin.

    python scripts/experiment_8h.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from load_supabase import SupabaseLoader, load_dotenv
from run_forecast_cycle import fill_gaps
from train_baseline import load_training_data

warnings.simplefilter("ignore", category=RuntimeWarning)
REGIMES = {
    "4h (09-19 06:00 - 09-20 11:00)": (pd.Timestamp("2026-09-19T06:00Z"), pd.Timestamp("2026-09-20T11:00Z")),
    "8h (09-21 00:00 -)": (pd.Timestamp("2026-09-21T00:00Z"), pd.Timestamp("2030-01-01T00:00Z")),
}
PERIODS = list(range(8, 49))  # 2h..12h in 15-min steps
AUTO_LOOKBACK = 16  # 4h of origins to judge the period


def series():
    raw = load_training_data()[["station_id", "observed_at", "demand"]].copy()
    raw["observed_at"] = pd.to_datetime(raw["observed_at"], utc=True)
    truth = raw.pivot_table(index="observed_at", columns="station_id", values="demand", aggfunc="last")
    filled = fill_gaps(raw).pivot_table(index="observed_at", columns="station_id", values="demand", aggfunc="last")
    filled = filled.loc[filled.index >= pd.Timestamp("2026-09-17T00:00Z")]
    return filled.to_numpy(float), filled.index, list(filled.columns), truth.reindex(filled.index).to_numpy(float)


def sm(y, i, w):
    return np.nanmean(y[i - w:i + w + 1], axis=0)


def harmonic(y, t, k, period, window, m):
    s = np.arange(t - window + 1, t + 1)
    def basis(x):
        cols = [np.ones_like(x, dtype=float)]
        for j in range(1, m + 1):
            cols += [np.cos(2 * np.pi * j * x / period), np.sin(2 * np.pi * j * x / period)]
        return np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(basis(s), y[t - window + 1:t + 1], rcond=None)
    return (basis(np.array([t + k])) @ beta)[0]


def best_period(y, t):
    """Period whose smoothed copy fitted the last AUTO_LOOKBACK observed values best (pooled WAPE)."""
    obs = y[t - AUTO_LOOKBACK + 1:t + 1]
    scores = {}
    for p in PERIODS:
        pred = np.stack([sm(y, i - p, 1) for i in range(t - AUTO_LOOKBACK + 1, t + 1)])
        scores[p] = np.nansum(np.abs(pred - obs)) / np.nansum(obs)
    return min(scores, key=scores.get), min(scores.values())


def candidates(y, t, k, regime_period, auto_p):
    out = {"persistencia": y[t]}
    for name, p in (("oraculo", regime_period), ("auto", auto_p)):
        out[f"{name} copia"] = y[t + k - p]
        out[f"{name} sm w1"] = sm(y, t + k - p, 1)
        out[f"{name} sm w2"] = sm(y, t + k - p, 2)
        for wmult in (1, 1.5, 2):
            for m in (2, 3):
                out[f"{name} armonico {wmult}P m{m}"] = harmonic(y, t, k, p, int(wmult * p), m)
        h = harmonic(y, t, k, p, p, 2)
        out[f"{name} (armonico P m2 + sm w2)/2"] = (h + sm(y, t + k - p, 2)) / 2
        res = np.mean([y[t - j] - harmonic(y, t - j - k, k, p, p, 2) for j in range(2)], axis=0) if False else None
    return {n: np.clip(v, 0, None) for n, v in out.items()}


def main():
    load_dotenv()
    y, index, stations, truth = series()
    loader = SupabaseLoader(os.environ["SUPABASE_URL"], os.getenv("SUPABASE_SECRET_KEY") or os.environ["SUPABASE_KEY"])
    sent = pd.DataFrame(loader.select_all("predictions", {
        "select": "station_id,target_at,predicted_demand,forecast_runs!inner(data_cutoff),submissions:forecast_runs!inner(submissions!inner(status))",
        "target_at": "gt.2026-09-19T00:00:00Z",
    }, order="id.asc")) if False else pd.DataFrame(loader.select_all("predictions", {
        "select": "station_id,target_at,predicted_demand,forecast_runs!inner(data_cutoff)",
        "target_at": "gt.2026-09-19T00:00:00Z",
    }, order="id.asc"))
    loader.close()
    sent["cutoff"] = pd.to_datetime(sent["forecast_runs"].map(lambda r: r["data_cutoff"]), utc=True)
    sent["target_at"] = pd.to_datetime(sent["target_at"], utc=True)
    sent = sent.groupby(["cutoff", "target_at", "station_id"])["predicted_demand"].last()
    col = {s: i for i, s in enumerate(stations)}
    pos = {ts: i for i, ts in enumerate(index)}

    rows, chosen = [], []
    for label, (start, end) in REGIMES.items():
        regime_period = 16 if label.startswith("4h") else 32
        for t, ts in enumerate(index):
            if ts < start or ts > end or ts.minute != 0 or t + 4 >= len(index):
                continue
            auto_p, auto_err = best_period(y, t)
            chosen.append((label, ts, auto_p / 4, round(auto_err, 3)))
            for k in range(1, 5):
                actual = truth[t + k]
                preds = candidates(y, t, k, regime_period, auto_p)
                key = [(ts, index[t + k], s) for s in stations]
                if all(kk in sent.index for kk in key):
                    preds["ENVIADO"] = np.array([sent[kk] for kk in key], dtype=float)
                for name, p in preds.items():
                    rows.append((label, ts, k, name, np.nansum(np.abs(p - actual)), np.nansum(np.where(np.isnan(p), np.nan, actual))))
    frame = pd.DataFrame(rows, columns=["regime", "origin", "k", "cand", "err", "act"])
    cyc = frame.groupby(["regime", "cand", "origin"])[["err", "act"]].sum()
    cyc = 100 * (1 - cyc["err"] / cyc["act"])
    table = cyc.groupby(["regime", "cand"]).agg(["mean", "min", "size"]).round(2)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    for regime in REGIMES:
        print(f"\n=== regimen {regime} (1 - WAPE por ciclo) ===")
        print(table.loc[regime].sort_values("mean", ascending=False).to_string())
    ch = pd.DataFrame(chosen, columns=["regime", "origin", "auto_period_h", "copy_wape"])
    print("\nperiodo elegido por 'auto':")
    print(ch.groupby("regime")["auto_period_h"].value_counts().to_string())
    print("\nultimos ciclos (enviado vs mejores):")
    last = cyc.unstack("cand").loc["8h (09-21 00:00 -)"]
    keep = [c for c in ["ENVIADO", "oraculo sm w1", "oraculo sm w2", "oraculo armonico 1P m2", "auto armonico 1P m2", "auto sm w2"] if c in last.columns]
    print(last[keep].round(1).to_string())


if __name__ == "__main__":
    main()
