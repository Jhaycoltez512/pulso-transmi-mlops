"""Temporary, read-only: dump samples of every Pulso API payload the pipeline reads."""
import collections
import json
import os

import httpx

client = httpx.Client(base_url=os.environ["PULSO_API_URL"], headers={"Authorization": f"Bearer {os.environ['PULSO_API_KEY']}"}, timeout=60)


def show(path, params=None, rows=3):
    r = client.get(path, params=params or {})
    print(f"\n===== {path} {params or ''} -> {r.status_code}")
    try:
        body = r.json()
    except Exception:
        print(r.text[:1500]); return None
    if isinstance(body, dict) and isinstance(body.get("data"), list):
        meta = {k: v for k, v in body.items() if k != "data"}
        print("meta:", json.dumps(meta)[:800], "| n data:", len(body["data"]))
        for row in body["data"][:rows]:
            print(json.dumps(row))
        for row in body["data"][-rows:]:
            print("tail:", json.dumps(row))
    else:
        print(json.dumps(body)[:3000])
    return body


show("/v1/meta")
show("/v1/clock")
show("/v1/forecast-cycles/current")
show("/v1/context", {"limit": 3})
show("/v1/observations", {"limit": 3})
spec = client.get("/openapi.json")
if spec.status_code == 200:
    s = spec.json()
    print("\n===== openapi paths:", list(s.get("paths", {})))
    for name, schema in s.get("components", {}).get("schemas", {}).items():
        if any(k in name.lower() for k in ("observ", "stream", "measure", "submission", "cycle")):
            print(name, json.dumps(schema)[:1500])
rows, cursor = [], None
while True:
    page = client.get("/v1/stream/observations", params={"limit": 5000, **({"cursor": cursor} if cursor else {})}).json()
    rows.extend(page["data"]); cursor = page.get("next_cursor")
    if cursor is None:
        break
print("\n===== stream total rows:", len(rows))
print("keys:", collections.Counter(tuple(sorted(r)) for r in rows).most_common(5))
meas = collections.Counter(json.dumps(r.get("measurement"), sort_keys=True)[:200] if not isinstance(r.get("measurement"), (int, float, str, type(None))) else type(r.get("measurement")).__name__ for r in rows)
print("measurement kinds:", meas.most_common(10))
with_m = [r for r in rows if "measurement" in r]
for r in rows[:3] + with_m[:5] + rows[-5:]:
    print(json.dumps(r))
print("max observed_at:", max((r.get("observed_at") or "") for r in rows), "max released_at:", max((r.get("released_at") or "") for r in rows))
