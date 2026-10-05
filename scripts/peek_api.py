"""Temporary, read-only: API clock, current cycle and leaderboard header."""
import json
import os

import httpx

c = httpx.Client(base_url=os.environ["PULSO_API_URL"], headers={"Authorization": f"Bearer {os.environ['PULSO_API_KEY']}"}, timeout=30)
for path, params in (("/health", {}), ("/v1/clock", {}), ("/v1/forecast-cycles/current", {}), ("/v1/meta", {})):
    r = c.get(path, params=params)
    print(path, r.status_code, r.text[:700])
lb = c.get("/v1/leaderboard", params={"window": "cumulative"}).json()
print("leaderboard", json.dumps({k: v for k, v in lb.items() if k != "data"})[:500])
