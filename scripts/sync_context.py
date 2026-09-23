"""Incrementally sync weather/event context into Supabase.

Unlike observations, /v1/context has no /v1/stream/ variant or released_at
semantics -- it's the plain paginated read endpoint (start/end/cursor/limit),
the same one the one-time initial load (load_supabase.py) used. That load ran
once on 2026-09-18 and nothing synced context again after it: sync_state for
'context' sat untouched for 5 days while 'observations' kept updating every
10 minutes, and context fell over 2 days behind. This script closes that gap
by being the context equivalent of sync_stream_observations.py, run on the
same schedule.

`start` is inclusive, so re-fetching the last synced row is expected; the
upsert on observed_at makes that harmless.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from load_supabase import SupabaseLoader, git_commit, load_dotenv


def main() -> None:
    load_dotenv()
    database_url = os.getenv("SUPABASE_URL")
    database_key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY")
    pulso_url = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
    pulso_key = os.getenv("PULSO_API_KEY")
    if not database_url or not database_key or not pulso_key:
        raise SystemExit("Configure SUPABASE_URL, SUPABASE_SECRET_KEY and PULSO_API_KEY.")

    loader = SupabaseLoader(database_url, database_key)
    # Created before the network call, same lesson as sync_stream_observations.py: a
    # connection failure must still leave a record instead of vanishing with a traceback.
    run = loader.insert("ingestion_runs", {"source_name": "pulso-transmi-context", "git_commit": git_commit()})
    try:
        state = loader.select("sync_state", {"stream_name": "eq.context", "limit": 1})
        start = state[0].get("last_observed_at") if state else None

        rows: list[dict[str, Any]] = []
        cursor = None
        with httpx.Client(
            base_url=pulso_url, headers={"Authorization": f"Bearer {pulso_key}"},
            timeout=httpx.Timeout(60, connect=15), transport=httpx.HTTPTransport(retries=3),
        ) as client:
            while True:
                params = {"limit": 5000, **({"start": start} if start else {}), **({"cursor": cursor} if cursor else {})}
                response = client.get("/v1/context", params=params)
                response.raise_for_status()
                page = response.json()
                rows.extend(page["data"])
                cursor = page.get("next_cursor")
                if cursor is None:
                    break

        loader.patch("ingestion_runs", run["id"], {"context_rows_read": len(rows)})
        if rows:
            for row in rows:
                row["ingestion_run_id"] = run["id"]
            loader.upsert("context", rows, "observed_at")
            latest_observed_at = max(row["observed_at"] for row in rows)
            loader.upsert("sync_state", [{
                "stream_name": "context", "last_observed_at": latest_observed_at, "last_ingestion_run_id": run["id"],
            }], "stream_name")
        loader.patch("ingestion_runs", run["id"], {"finished_at": datetime.now(timezone.utc).isoformat(), "status": "succeeded"})
        print(f"Context synchronized: {len(rows)} rows upserted.")
    except httpx.HTTPError as error:
        detail = error.response.text if isinstance(error, httpx.HTTPStatusError) else str(error)
        loader.patch("ingestion_runs", run["id"], {"finished_at": datetime.now(timezone.utc).isoformat(), "status": "failed", "error_message": detail})
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    main()
