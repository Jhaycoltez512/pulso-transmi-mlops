"""Load the Pulso TransMi source data into an already-migrated Supabase project.

Requires SUPABASE_URL and a service-role SUPABASE_SECRET_KEY (or SUPABASE_KEY)
in .env or the environment.
The load is idempotent: source records are upserted by their natural keys.
"""

from __future__ import annotations

import os
import subprocess
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx

from pulso_transmi import PulsoTransmiClient

BATCH_SIZE = 1_000


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs without adding a runtime dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def batches(rows: list[dict[str, Any]]) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), BATCH_SIZE):
        yield rows[start : start + BATCH_SIZE]


def rest_url(url: str) -> str:
    return url.rstrip("/") if url.rstrip("/").endswith("/rest/v1") else f"{url.rstrip('/')}/rest/v1"


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def frame_records(frame: Any) -> list[dict[str, Any]]:
    """Serialize pandas timestamps to JSON-compatible ISO-8601 strings."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


class SupabaseLoader:
    def __init__(self, url: str, key: str) -> None:
        self.client = httpx.Client(
            base_url=rest_url(url), timeout=60,
            headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )

    def close(self) -> None:
        self.client.close()

    def upsert(self, table: str, rows: list[dict[str, Any]], keys: str) -> None:
        for batch in batches(rows):
            response = self.client.post(
                f"/{table}", params={"on_conflict": keys}, json=batch,
                headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            )
            response.raise_for_status()

    def insert(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(f"/{table}", json=row, headers={"Prefer": "return=representation"})
        response.raise_for_status()
        return response.json()[0]

    def patch(self, table: str, identifier: str, row: dict[str, Any]) -> None:
        response = self.client.patch(f"/{table}", params={"id": f"eq.{identifier}"}, json=row)
        response.raise_for_status()

    def select(self, table: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        response = self.client.get(f"/{table}", params=params)
        response.raise_for_status()
        return response.json()


def main() -> None:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("SUPABASE_URL and SUPABASE_SECRET_KEY must be configured.")

    loader = SupabaseLoader(url, key)
    try:
        run = loader.insert("ingestion_runs", {"git_commit": git_commit()})
        with PulsoTransmiClient() as source:
            stations = source.stations()
            observations = source.observations_dataframe(page_size=5000)
            context = source.context_dataframe(page_size=5000)

        station_rows = stations.to_dict("records")
        # PostgreSQL timestamps are serialized as ISO-8601 strings for PostgREST.
        observation_rows = frame_records(observations.assign(ingestion_run_id=run["id"]))
        context_rows = frame_records(context.assign(ingestion_run_id=run["id"]))
        loader.upsert("stations", station_rows, "station_id")
        loader.upsert("context", context_rows, "observed_at")
        loader.upsert("observations", observation_rows, "station_id,observed_at")
        latest = observations["observed_at"].max().isoformat()
        loader.upsert("sync_state", [
            {"stream_name": "observations", "last_observed_at": latest, "last_ingestion_run_id": run["id"]},
            {"stream_name": "context", "last_observed_at": context["observed_at"].max().isoformat(), "last_ingestion_run_id": run["id"]},
        ], "stream_name")
        loader.patch("ingestion_runs", run["id"], {
            "finished_at": datetime.now(timezone.utc).isoformat(), "status": "succeeded", "last_observed_at": latest,
            "observation_rows_read": len(observation_rows), "context_rows_read": len(context_rows),
        })
        print(f"Loaded {len(station_rows)} stations, {len(context_rows)} context rows and {len(observation_rows)} observations.")
    except httpx.HTTPStatusError as error:
        raise SystemExit(f"Supabase request failed ({error.response.status_code}): {error.response.text}") from error
    finally:
        loader.close()


if __name__ == "__main__":
    main()
