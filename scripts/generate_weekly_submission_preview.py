"""Build and validate a local submission preview using the weekly-naive model.

This script never calls POST /v1/submissions. It only reads the active cycle and
observations, then writes a JSON preview under artifacts/.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pandas as pd

from train_baseline import ARTIFACTS_DIR, fetch_all, get_rest_url
from train_weekly_naive import LAG_PERIODS

PULSO_API_URL = "https://pulso-transmi.72-60-245-2.sslip.io"
STATION_ID = re.compile(r"^[0-9]{5}$")


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def active_cycle() -> dict:
    response = httpx.get(f"{PULSO_API_URL}/v1/forecast-cycles/current", timeout=30)
    response.raise_for_status()
    return response.json()


def weekly_predictions(targets: list[dict]) -> list[dict]:
    base_url, key = get_rest_url()
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=60) as client:
        rows = fetch_all(client, "observations", "station_id,observed_at,demand", "observed_at.asc,station_id.asc")
    observations = pd.DataFrame(rows)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    lookup = observations.set_index(["station_id", "observed_at"])["demand"]
    predictions = []
    for target in targets:
        target_at = pd.Timestamp(target["target_at"])
        source_at = target_at - timedelta(minutes=LAG_PERIODS * 15)
        key = (target["station_id"], source_at)
        if key not in lookup:
            raise RuntimeError(f"No weekly lag observation for station={key[0]}, observed_at={key[1].isoformat()}")
        predictions.append({
            "station_id": target["station_id"],
            "target_at": target_at.isoformat().replace("+00:00", "Z"),
            "value": float(lookup.loc[key]),
        })
    return predictions


def validate(payload: dict, cycle: dict) -> list[str]:
    """Validate the documented SubmissionInput contract without sending it."""
    errors: list[str] = []
    if payload.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    if payload.get("cycle_id") != cycle["cycle_id"]:
        errors.append("cycle_id does not match active cycle")
    if not payload.get("client_run_id") or len(payload["client_run_id"]) > 128:
        errors.append("client_run_id must contain 1 to 128 characters")
    try:
        pd.Timestamp(payload["data_cutoff"])
    except (KeyError, ValueError, TypeError):
        errors.append("data_cutoff must be a timezone-aware timestamp")
    model = payload.get("model", {})
    if not isinstance(model.get("version"), str) or not 1 <= len(model["version"]) <= 64:
        errors.append("model.version must contain 1 to 64 characters")

    expected = {(item["station_id"], item["target_at"]) for item in cycle["targets"]}
    actual = set()
    for prediction in payload.get("predictions", []):
        station_id = prediction.get("station_id")
        target_at = prediction.get("target_at")
        value = prediction.get("value")
        if not isinstance(station_id, str) or not STATION_ID.fullmatch(station_id):
            errors.append(f"invalid station_id: {station_id}")
        try:
            normalized_target = pd.Timestamp(target_at).isoformat().replace("+00:00", "Z")
        except (ValueError, TypeError):
            errors.append(f"invalid target_at: {target_at}")
            normalized_target = str(target_at)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 100000:
            errors.append(f"invalid prediction value for {station_id}")
        actual.add((station_id, normalized_target))
    if not 1 <= len(payload.get("predictions", [])) <= 100:
        errors.append("predictions must contain between 1 and 100 items")
    if len(payload.get("predictions", [])) != len(actual):
        errors.append("predictions contain duplicate station_id/target_at keys")
    if actual != expected:
        errors.append("prediction targets do not exactly match the active cycle")
    return errors


def main() -> None:
    cycle = active_cycle()
    if cycle["state"] != "open":
        raise RuntimeError(f"The active cycle is not open: {cycle['state']}")
    predictions = weekly_predictions(cycle["targets"])
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": f"weekly-naive-preview-{uuid4().hex[:12]}",
        "data_cutoff": cycle["data_cutoff"],
        "model": {
            "version": "weekly-naive-v1",
            "training_data_end": cycle["data_cutoff"],
            "git_commit": git_commit(),
        },
        "predictions": predictions,
    }
    errors = validate(payload, cycle)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "submission_weekly_naive_preview.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "submission_weekly_naive_validation.json").write_text(
        json.dumps({"valid": not errors, "errors": errors, "prediction_count": len(predictions)}, indent=2),
        encoding="utf-8",
    )
    if errors:
        raise SystemExit("Preview is invalid: " + "; ".join(errors))
    print(f"Preview valid: {len(predictions)} predictions for cycle {cycle['cycle_id']}")
    print("No submission was sent.")


if __name__ == "__main__":
    main()
