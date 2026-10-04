"""Replay the production ensemble on every live cycle with shorter weight windows (read-only).

At virtual 2026-09-20 12:15 (with the API's schema v2) the 4h oscillation stopped: the ensemble,
whose weights came from the last 4h of evaluated targets (mostly old regime), kept submitting the
periodic copies -- 34% and 11% on the next two cycles while persistence scored 75% and 69%.
This compares weight windows / powers on all live cycles since the 4h regime started, so a
faster-adapting setting is only adopted if it does not cost the stable periodic regime.

    python scripts/backtest_regime_switch.py
"""

from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd

from forecast_adjustments import EXPERTS, LAG_EXPERTS, PERIODIC_EXPERTS, production_adjust
from load_supabase import SupabaseLoader, load_dotenv
from train_baseline import load_training_data

REGIME_START = pd.Timestamp("2026-09-18T05:00:00Z")
BREAK = pd.Timestamp("2026-09-20T12:00:00Z")
EXPERT_SET = (*EXPERTS, *LAG_EXPERTS, *PERIODIC_EXPERTS)
VARIANTS = {"prod 4h p6": (4, 6.0), "2h p6": (2, 6.0), "1h p6": (1, 6.0), "1h p3": (1, 3.0)}
CYCLES = 30


def main() -> None:
    load_dotenv()
    data = load_training_data()[["station_id", "observed_at", "demand"]].copy()
    data["observed_at"] = pd.to_datetime(data["observed_at"], utc=True)
    actual = data.set_index(["station_id", "observed_at"])["demand"]
    last_obs = data["observed_at"].max()
    loader = SupabaseLoader(os.environ["SUPABASE_URL"], os.getenv("SUPABASE_SECRET_KEY") or os.environ["SUPABASE_KEY"])
    sent = pd.DataFrame(loader.select_all("predictions", {
        "select": "station_id,target_at,horizon_minutes,predicted_demand,raw_predicted_demand,forecast_runs!inner(data_cutoff)",
        "target_at": f"gt.{(REGIME_START - timedelta(hours=6)).isoformat()}",
    }, order="id.asc"))
    loader.close()
    sent["cutoff"] = pd.to_datetime(sent["forecast_runs"].map(lambda r: r["data_cutoff"]), utc=True)
    sent["target_at"] = pd.to_datetime(sent["target_at"], utc=True)
    sent["raw"] = sent["raw_predicted_demand"].fillna(sent["predicted_demand"])
    sent = sent.groupby(["cutoff", "target_at", "station_id", "horizon_minutes"], as_index=False)[["predicted_demand", "raw"]].last()

    rows = []
    cutoffs = sorted(c for c in sent["cutoff"].unique() if c >= REGIME_START)[-CYCLES:]
    for cutoff, cycle in sent.loc[sent["cutoff"].isin(cutoffs)].groupby("cutoff"):
        if cycle["target_at"].max() > last_obs:
            continue
        recent = data.loc[(data["observed_at"] > cutoff - timedelta(days=15)) & (data["observed_at"] <= cutoff)]
        frame = cycle.rename(columns={"raw": "model"})[["station_id", "horizon_minutes", "target_at", "model"]].assign(observed_at=cutoff).reset_index(drop=True)
        truth = np.array([actual.get((s, t), np.nan) for s, t in zip(frame["station_id"], frame["target_at"])], dtype=float)
        if np.isnan(truth).any():
            continue
        score = lambda v: 100 * max(0.0, 1 - np.abs(np.asarray(v, dtype=float) - truth).sum() / truth.sum())
        last = np.array([actual.get((s, cutoff), np.nan) for s in frame["station_id"]], dtype=float)
        row = {"cutoff": cutoff, "enviado": score(cycle["predicted_demand"]), "modelo": score(frame["model"]), "persistencia": score(last)}
        past_all = sent.loc[(sent["target_at"] <= cutoff) & (sent["target_at"] > cutoff - timedelta(hours=6))].rename(columns={"raw": "model"})
        past_all = past_all.assign(actual=[actual.get((s, t), np.nan) for s, t in zip(past_all["station_id"], past_all["target_at"])]).dropna(subset=["actual"])
        for name, (hours, power) in VARIANTS.items():
            values, _ = production_adjust(recent, frame, past_all[["station_id", "horizon_minutes", "target_at", "model", "actual"]],
                                          "ensemble", timedelta(hours=hours), power, ensemble_on_adjusted=False, pool_stations=True, experts=EXPERT_SET)
            row[name] = score(values)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("cutoff")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.round(1).to_string())
    for label, part in (("regimen 4h (desde 24h)", table.loc[(table.index >= REGIME_START + timedelta(hours=24)) & (table.index < BREAK)]),
                        ("ultimas 24 antes del corte", table.loc[table.index < BREAK].tail(24)),
                        ("tras el corte", table.loc[table.index >= BREAK]), ("todo", table)):
        print(f"\nmedia {label} ({len(part)} ciclos):")
        print(part.mean().round(2).to_string())


if __name__ == "__main__":
    main()
