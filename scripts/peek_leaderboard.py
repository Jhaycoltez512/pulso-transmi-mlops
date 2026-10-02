"""Temporary, read-only: print the competition leaderboard and the API's cycle/score endpoints.

Uses PULSO_API_KEY from the environment; prints no secrets.
"""

from __future__ import annotations

import json
import os

import httpx

BASE = os.environ.get("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")


def show(client: httpx.Client, path: str, limit: int = 6000) -> dict | list | None:
    response = client.get(path)
    print(f"\n### GET {path} -> {response.status_code}")
    try:
        body = response.json()
    except ValueError:
        print(response.text[:500])
        return None
    print(json.dumps(body, ensure_ascii=False, indent=1)[:limit])
    return body


def main() -> None:
    headers = {"Authorization": f"Bearer {os.environ['PULSO_API_KEY']}"}
    with httpx.Client(base_url=BASE, headers=headers, timeout=30) as client:
        spec = client.get("/openapi.json")
        if spec.status_code == 200:
            paths = spec.json().get("paths", {})
            print("### endpoints:")
            for path, ops in paths.items():
                params = sorted({p["name"] for op in ops.values() if isinstance(op, dict) for p in op.get("parameters", [])})
                print(f"  {','.join(m.upper() for m in ops)} {path} params={params}")
        else:
            print(f"### /openapi.json -> {spec.status_code}")
        show(client, "/v1/me")
        show(client, "/v1/leaderboard?window=cumulative")
        show(client, "/v1/leaderboard?window=rolling_24h")


if __name__ == "__main__":
    main()
