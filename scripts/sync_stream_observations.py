"""Collect Pulso stream data with durable, idempotent lineage in Supabase.

Runs every 10 minutes, so it does not log to MLflow itself -- that would create
a noisy run per sync regardless of whether anything downstream changes. Data
and model versions are logged together to MLflow only when a model actually
retrains (see train_catboost_direct.record_lineage), driven by the drift/
performance rule in run_forecast_cycle.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from load_supabase import SupabaseLoader, git_commit, load_dotenv


def released_marker(row: dict[str, Any]) -> str:
    """Use release time to retain late-arriving records; observed_at is a fallback."""
    return row.get("released_at") or row["observed_at"]


def new_rows_since(rows: list[dict[str, Any]], last_released_at: str | None) -> list[dict[str, Any]]:
    return rows if last_released_at is None else [row for row in rows if released_marker(row) > last_released_at]


def version_for(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: (released_marker(row), row["station_id"], row["observed_at"]))
    digest = hashlib.sha256(json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"stream-{max(released_marker(row) for row in rows)}-{digest}"


def main() -> None:
    load_dotenv()
    database_url = os.getenv("SUPABASE_URL")
    database_key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY")
    pulso_url = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
    pulso_key = os.getenv("PULSO_API_KEY")
    if not database_url or not database_key or not pulso_key:
        raise SystemExit("Configure SUPABASE_URL, SUPABASE_SECRET_KEY and PULSO_API_KEY.")

    loader = SupabaseLoader(database_url, database_key)
    # Created before the network call -- a timeout or connection error while fetching from the
    # Pulso API must still leave a record, not vanish with an unhandled traceback (found live:
    # httpx.ConnectTimeout crashed the run with no trace anywhere in ingestion_runs).
    run = loader.insert("ingestion_runs", {"source_name": "pulso-transmi-stream", "git_commit": git_commit()})
    try:
        state = loader.select("sync_state", {"stream_name": "eq.observations", "limit": 1})
        last_released_at = state[0].get("last_released_at") if state else None
        rows: list[dict[str, Any]] = []
        cursor = None
        # Retries only connection failures (the request never reached the API), with backoff:
        # transient ConnectTimeouts to the Pulso API killed the whole job twice on 2026-09-23.
        with httpx.Client(
            base_url=pulso_url, headers={"Authorization": f"Bearer {pulso_key}"},
            timeout=httpx.Timeout(60, connect=15), transport=httpx.HTTPTransport(retries=3),
        ) as client:
            while True:
                response = client.get("/v1/stream/observations", params={"limit": 5000, **({"cursor": cursor} if cursor else {})})
                response.raise_for_status()
                page = response.json()
                rows.extend(page["data"])
                cursor = page.get("next_cursor")
                if cursor is None:
                    break
        fresh_rows = new_rows_since(rows, last_released_at)
        data_version = version_for(fresh_rows)
        loader.patch("ingestion_runs", run["id"], {"data_version": data_version})
        latest_observed_at = max((row["observed_at"] for row in fresh_rows), default=(state[0].get("last_observed_at") if state else None))
        latest_released_at = max((released_marker(row) for row in fresh_rows), default=last_released_at)
        if fresh_rows:
            for row in fresh_rows:
                row["ingestion_run_id"] = run["id"]
            loader.upsert("observations", fresh_rows, "station_id,observed_at")
            loader.upsert("sync_state", [{
                "stream_name": "observations", "last_observed_at": latest_observed_at,
                "last_released_at": latest_released_at, "last_ingestion_run_id": run["id"],
            }], "stream_name")
        loader.patch("ingestion_runs", run["id"], {
            "finished_at": datetime.now(timezone.utc).isoformat(), "status": "succeeded",
            "last_observed_at": latest_observed_at, "observation_rows_read": len(fresh_rows),
        })
        print(f"Stream synchronized: {len(rows)} rows seen, {len(fresh_rows)} new rows upserted.")
    except httpx.HTTPError as error:
        # HTTPStatusError has a response body worth saving; connection/timeout errors (no
        # response was ever received) fall back to the exception's own message.
        detail = error.response.text if isinstance(error, httpx.HTTPStatusError) else str(error)
        loader.patch("ingestion_runs", run["id"], {"finished_at": datetime.now(timezone.utc).isoformat(), "status": "failed", "error_message": detail})
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    main()
