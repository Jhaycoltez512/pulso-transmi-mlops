"""Snapshot the competition leaderboard into Supabase and derive per-cycle results.

GET /v1/leaderboard only serves two windows: cumulative (since the competition window opened,
`resolved_cycles` cycles so far) and rolling_24h. The cumulative figure is not a mean of
per-cycle scores: `raw_wape` is the participant's total absolute error over the total actual
demand of everything resolved. Every participant is scored on the same targets, so with A_n the
actual demand summed over the targets of the first n resolved cycles (computed from our own
forecast_targets + observations), the error on the newest cycle is

    W_n * c_n * A_n - W_{n-1} * c_{n-1} * A_{n-1}

(c = coverage, exact when it is 1), divided by that cycle's demand A_n - A_{n-1}. A participant
whose submitted-cycle count didn't grow missed the cycle and gets no result.

Runs at the end of the collector; it only writes when the leaderboard changed.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd

from load_supabase import SupabaseLoader, load_dotenv

TABLE = "leaderboard_snapshots"
WINDOWS = ("cumulative", "rolling_24h")


def cycle_results(previous: list[dict[str, Any]], current: list[dict[str, Any]], resolved: int,
                  demand_previous: float | None, demand_now: float | None) -> dict[str, tuple[float, float] | None]:
    """{display_name: (cycle_wape, cycle_accuracy) or None} for the cycle resolved between two snapshots."""
    if not demand_previous or not demand_now or demand_now <= demand_previous:
        return {}
    cycle_demand = demand_now - demand_previous
    before = {row["display_name"]: row for row in previous}
    out: dict[str, tuple[float, float] | None] = {}
    for row in current:
        prev = before.get(row["display_name"])
        if prev is None or row.get("raw_wape") is None or prev.get("raw_wape") is None:
            continue
        submitted_now = round(row["coverage"] * resolved)
        submitted_before = round(prev["coverage"] * (resolved - 1))
        if submitted_now <= submitted_before:
            out[row["display_name"]] = None  # missed this cycle
            continue
        error = row["raw_wape"] * row["coverage"] * demand_now - prev["raw_wape"] * prev["coverage"] * demand_previous
        wape = max(0.0, error / cycle_demand)
        out[row["display_name"]] = (wape, 100 * max(0.0, 1 - wape))
    return out


def demand_totals(loader: SupabaseLoader, starts_at: str, counts: list[int]) -> dict[int, float | None]:
    """Actual demand over every target of the first n cycles opened since starts_at, for each n."""
    targets = pd.DataFrame(loader.select_all("forecast_targets", {
        "select": "cycle_id,station_id,target_at,forecast_cycles!inner(opens_at)",
        "forecast_cycles.opens_at": f"gte.{starts_at}",
    }, order="cycle_id.asc,station_id.asc,target_at.asc"))
    if targets.empty:
        return {n: None for n in counts}
    targets["opens_at"] = pd.to_datetime(targets["forecast_cycles"].map(lambda c: c["opens_at"]), utc=True)
    targets["target_at"] = pd.to_datetime(targets["target_at"], utc=True)
    order = targets.drop_duplicates("cycle_id").sort_values("opens_at")["cycle_id"].tolist()
    observations = pd.DataFrame(loader.select_all("observations", {
        "select": "station_id,observed_at,demand",
        "observed_at": [f"gte.{targets['target_at'].min().isoformat()}", f"lte.{targets['target_at'].max().isoformat()}"],
    }, order="observed_at.asc,station_id.asc"))
    if observations.empty:
        return {n: None for n in counts}
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    merged = targets.merge(observations.rename(columns={"observed_at": "target_at"}), on=["station_id", "target_at"], how="left")
    result: dict[int, float | None] = {}
    for n in counts:
        part = merged.loc[merged["cycle_id"].isin(order[:n])]
        complete = n <= len(order) and part["demand"].notna().all()
        result[n] = float(part["demand"].sum()) if complete else None
    return result


def main() -> None:
    load_dotenv()
    loader = SupabaseLoader(os.environ["SUPABASE_URL"], os.getenv("SUPABASE_SECRET_KEY") or os.environ["SUPABASE_KEY"])
    base = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
    headers = {"Authorization": f"Bearer {os.environ['PULSO_API_KEY']}"}
    try:
        with httpx.Client(base_url=base, headers=headers, timeout=30) as client:
            me = client.get("/v1/me").json().get("display_name")
            boards = {window: client.get("/v1/leaderboard", params={"window": window}).json() for window in WINDOWS}
        for window, board in boards.items():
            data = board.get("data") or []
            if not data:
                continue
            resolved = board.get("resolved_cycles")
            key = str(resolved) if window == "cumulative" else str(data[0].get("calculated_at"))
            if loader.select(TABLE, {"board_window": f"eq.{window}", "snapshot_key": f"eq.{key}", "select": "id", "limit": 1}):
                print(f"Leaderboard {window}: snapshot {key} already stored.")
                continue
            demand_now, results = None, {}
            if window == "cumulative" and resolved:
                totals = demand_totals(loader, board["starts_at"], [resolved - 1, resolved])
                demand_now = totals[resolved]
                previous = loader.select(TABLE, {"board_window": "eq.cumulative", "snapshot_key": f"eq.{resolved - 1}"})
                results = cycle_results(previous, data, resolved, totals[resolved - 1], demand_now)
            rows = []
            for row in data:
                cycle = results.get(row["display_name"])
                rows.append({
                    "board_window": window, "snapshot_key": key, "resolved_cycles": resolved, "starts_at": board.get("starts_at"),
                    "demand_total": demand_now, "display_name": row["display_name"], "kind": row.get("kind"),
                    "eligible": row.get("eligible"), "accuracy": row.get("accuracy"), "raw_wape": row.get("raw_wape"),
                    "accuracy_at_20": row.get("accuracy_at_20"), "coverage": row.get("coverage"), "rank": row.get("rank"),
                    "calculated_at": row.get("calculated_at"), "is_me": row["display_name"] == me,
                    "cycle_wape": cycle[0] if cycle else None, "cycle_accuracy": cycle[1] if cycle else None,
                })
            loader.upsert(TABLE, rows, "board_window,snapshot_key,display_name")
            mine = next((r for r in rows if r["is_me"]), None)
            print(f"Leaderboard {window}: stored {len(rows)} rows (key {key}); me rank {mine and mine['rank']}, "
                  f"cycle accuracy {mine and mine['cycle_accuracy']}")
    finally:
        loader.close()


if __name__ == "__main__":
    main()
