"""Temporary, read-only: run the collector's normalization on the live stream without writing."""
import collections
import json
import os

import httpx

from sync_stream_observations import normalize_rows

client = httpx.Client(base_url=os.environ["PULSO_API_URL"], headers={"Authorization": f"Bearer {os.environ['PULSO_API_KEY']}"}, timeout=60)
rows, cursor = [], None
while True:
    page = client.get("/v1/stream/observations", params={"limit": 5000, **({"cursor": cursor} if cursor else {})}).json()
    rows.extend(page["data"]); cursor = page.get("next_cursor")
    if cursor is None:
        break
normalized, skipped = normalize_rows(rows)
print("rows", len(rows), "normalized", len(normalized), "skipped", skipped)
print("schemas", collections.Counter(r.get("schema_version", 1) for r in rows))
print("qualities", collections.Counter((r.get("measurement") or {}).get("quality") for r in rows if "measurement" in r))
print("units", collections.Counter((r.get("measurement") or {}).get("unit") for r in rows if "measurement" in r))
for r in normalized[-26:]:
    print(json.dumps(r))
me = client.get("/v1/me"); print("me", me.status_code, me.text[:600])
subs = client.get("/v1/submissions", params={"limit": 2}); print("submissions", subs.status_code, subs.text[:2500])
print("cycle", client.get("/v1/forecast-cycles/current").text[:2500])
print("clock", client.get("/v1/clock").text)
