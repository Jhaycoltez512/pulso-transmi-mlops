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


# The API moved to schema_version 2 on 2026-10-03 (virtual 2026-09-20 12:15): rows carry
# measurement={"value": "546.00", "unit": "passengers", "quality": "observed"} instead of demand,
# plus a schema_version field the observations table doesn't have -- the raw upsert then failed
# with PGRST204 and stopped the whole cycle. Rows are normalized to the table's own columns.
UNIT_SCALE = {"passengers": 1.0, "passenger": 1.0, "pax": 1.0, "trips": 1.0,
              "hundred_passengers": 100.0, "hundreds_of_passengers": 100.0,
              "thousand_passengers": 1000.0, "thousands_of_passengers": 1000.0, "kpassengers": 1000.0}
OBSERVATION_COLUMNS = ("station_id", "observed_at", "released_at", "demand")


def utc_iso(value: str | None) -> str | None:
    return None if value is None else as_instant(value).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def demand_of(row: dict[str, Any]) -> int | None:
    """Passengers in the period from either schema; None when the API reports no value."""
    if row.get("demand") is not None:
        return int(round(float(row["demand"])))
    measurement = row.get("measurement")
    if not isinstance(measurement, dict):
        if "measurement" in row or "demand" in row:
            return None
        raise ValueError(f"Stream row has neither demand nor measurement: {sorted(row)}")
    if measurement.get("value") in (None, ""):
        return None
    unit = str(measurement.get("unit") or "passengers").strip().lower().replace(" ", "_")
    if unit not in UNIT_SCALE:
        # Storing a value on the wrong scale would silently corrupt history and every prediction
        # built on it; failing leaves a readable error in ingestion_runs instead.
        raise ValueError(f"Unknown measurement unit {measurement.get('unit')!r} for {row.get('station_id')} {row.get('observed_at')}")
    return int(round(float(measurement["value"]) * UNIT_SCALE[unit]))


def normalize_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Map v1 and v2 stream rows onto the observations columns; returns (rows, skipped without a value)."""
    normalized, skipped = [], 0
    for row in rows:
        demand = demand_of(row)
        if demand is None:
            skipped += 1
            continue
        normalized.append({"station_id": str(row["station_id"]), "observed_at": utc_iso(row["observed_at"]),
                           "released_at": utc_iso(row.get("released_at")), "demand": demand})
    return normalized, skipped


def released_marker(row: dict[str, Any]) -> str:
    """Use release time to retain late-arriving records; observed_at is a fallback."""
    return row.get("released_at") or row["observed_at"]


def as_instant(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; 'Z' and '+00:00' must compare as the same instant."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def new_rows_since(rows: list[dict[str, Any]], last_released_at: str | None) -> list[dict[str, Any]]:
    # Compared as instants, not strings: the API returns "...Z" while Supabase hands the stored
    # cursor back as "...+00:00", and "Z" sorts after "+" -- so the latest release kept being
    # re-read as "new" on every run until the next one arrived (24 phantom rows per run).
    if last_released_at is None:
        return rows
    cursor = as_instant(last_released_at)
    return [row for row in rows if as_instant(released_marker(row)) > cursor]


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
        fresh_rows, skipped = normalize_rows(new_rows_since(rows, last_released_at))
        schemas = sorted({str(row.get("schema_version", 1)) for row in rows})
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
        print(f"Stream synchronized: {len(rows)} rows seen (schema {', '.join(schemas)}), {len(fresh_rows)} new rows upserted, {skipped} without a value skipped.")
    except ValueError as error:
        loader.patch("ingestion_runs", run["id"], {"finished_at": datetime.now(timezone.utc).isoformat(), "status": "failed", "error_message": str(error)})
        raise
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
