"""Operate the forecast cycle: evaluate recent accuracy, measure drift, decide whether
to keep or retrain the active model, predict, submit, and log the outcome.

Runs after scripts/sync_stream_observations.py in the same GitHub Actions job, so the
data it reads is already up to date. Writes to the operational tables the initial schema
already provisioned: forecast_runs, predictions, drift_measurements, submissions.
See docs/collector-and-lineage.md for the decision rule and its thresholds.

Set PULSO_SUBMIT_ENABLED=false to run the full pipeline (evaluate, decide, predict,
validate) without sending the real POST to the competition API.

Before submitting, predictions get an online bias correction (validated in
scripts/backtest_bias_correction.py): scaled by half of actual/predicted over the last 4h
of evaluated predictions. Set PULSO_BIAS_CORRECTION=false to submit raw model output.
"""

from __future__ import annotations

import io
import os
from datetime import timedelta
from typing import Any
from uuid import uuid4

import httpx
import joblib
import numpy as np
import pandas as pd

from generate_weekly_submission_preview import PULSO_API_URL, active_cycle, validate
from load_supabase import SupabaseLoader, download_object, load_dotenv, upload_object
from train_baseline import load_training_data, station_metrics
from train_catboost_direct import (
    FEATURES,
    ENSEMBLE_WEIGHT,
    add_origin_features,
    git_commit,
    prediction_rows,
    record_lineage,
    train_models,
)

MODEL_BUCKET = "models"
STALENESS_DAYS = 7
PERFORMANCE_DEGRADATION_FACTOR = 1.15
PERFORMANCE_MIN_SAMPLES = 20
PERFORMANCE_WINDOW_DAYS = 3
DRIFT_RECENT_WINDOW_DAYS = 3
DRIFT_REFERENCE_WINDOW_DAYS = 14
PSI_THRESHOLD = 0.25
DRIFT_FEATURES = ["demand", "rain_forecast", "temperature_forecast", "event_intensity"]
# Winner of the walk-forward sweep (global scope, 4h window, half correction): same config
# was best at every horizon, ~-0.0012 WAPE, never worse than production in any fold.
BIAS_WINDOW_HOURS = 4
BIAS_ALPHA = 0.5
BIAS_CLIP = (0.8, 1.25)
BIAS_MIN_SAMPLES = 48  # one full cycle: 12 stations x 4 horizons
BIAS_ALERT = 0.05  # |relative bias| worth flagging on the dashboard


def bias_factor(rows: list[dict[str, Any]]) -> tuple[float | None, int]:
    """actual / raw predicted demand over evaluated predictions, all stations and horizons pooled.

    Uses the model's raw output, not the corrected value that was submitted: correcting on
    top of a previous correction would compound. Rows from before the correction existed
    have no raw value, but those were never corrected, so predicted_demand is the raw one.
    """
    usable = [row for row in rows if row.get("actual_demand") is not None]
    if len(usable) < BIAS_MIN_SAMPLES:
        return None, len(usable)
    predicted = sum(row["raw_predicted_demand"] if row.get("raw_predicted_demand") is not None else row["predicted_demand"] for row in usable)
    if predicted <= 0:
        return None, len(usable)
    return sum(row["actual_demand"] for row in usable) / predicted, len(usable)


def bias_scale(factor: float | None) -> float:
    if factor is None:
        return 1.0
    low, high = BIAS_CLIP
    return float(min(high, max(low, 1 + BIAS_ALPHA * (factor - 1))))


def population_stability_index(reference: np.ndarray, recent: np.ndarray, bins: int = 10) -> float:
    """PSI between two samples. Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 significant shift.

    Rounds to 6 decimals first: some features (event_intensity) decay towards zero without ever
    reaching it exactly, and quantile edges computed on that long near-zero tail produce a PSI that
    is technically correct but numerically meaningless -- rounding treats "practically zero" as zero.
    """
    reference = np.round(reference[~np.isnan(reference)], 6)
    recent = np.round(recent[~np.isnan(recent)], 6)
    if len(reference) < bins or len(recent) == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    recent_counts, _ = np.histogram(recent, bins=edges)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    recent_pct = np.clip(recent_counts / recent_counts.sum(), 1e-6, None)
    return float(np.sum((recent_pct - ref_pct) * np.log(recent_pct / ref_pct)))


def evaluate_recent_predictions(loader: SupabaseLoader, data: pd.DataFrame) -> None:
    """Backfill actual_demand for past predictions whose target_at now has a real observation."""
    now = pd.Timestamp.now(tz="UTC")
    pending = loader.select("predictions", {
        "actual_demand": "is.null", "target_at": f"lte.{now.isoformat()}", "select": "id,station_id,target_at",
    })
    if not pending:
        return
    pending_frame = pd.DataFrame(pending)
    pending_frame["target_at"] = pd.to_datetime(pending_frame["target_at"], utc=True)
    lookup = data.set_index(["station_id", "observed_at"])["demand"]
    for row in pending_frame.itertuples():
        key = (row.station_id, row.target_at)
        if key in lookup.index:
            loader.patch("predictions", row.id, {"actual_demand": int(lookup.loc[key]), "evaluated_at": now.isoformat()})


def recent_performance(loader: SupabaseLoader, horizons: list[int]) -> dict[int, dict[str, Any]]:
    """WAPE per horizon from predictions evaluated within the last PERFORMANCE_WINDOW_DAYS.

    Scores the raw model output, so the bias correction can't hide model degradation from
    the retrain rule (the threshold it's compared to is the raw model's validation WAPE).
    """
    since = (pd.Timestamp.now(tz="UTC") - timedelta(days=PERFORMANCE_WINDOW_DAYS)).isoformat()
    rows = loader.select("predictions", {
        "evaluated_at": f"gte.{since}", "select": "station_id,horizon_minutes,predicted_demand,raw_predicted_demand,actual_demand",
    })
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    frame["predicted_demand"] = frame["raw_predicted_demand"].fillna(frame["predicted_demand"])
    result: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        subset = frame.loc[frame["horizon_minutes"] == horizon]
        if len(subset) < PERFORMANCE_MIN_SAMPLES:
            continue
        scored = subset.rename(columns={"actual_demand": "demand"})[["station_id", "demand"]]
        metrics = station_metrics(scored, subset["predicted_demand"].to_numpy())
        result[horizon] = {"wape": metrics["mean_station_wape"], "samples": len(subset)}
    return result


def compute_data_drift(data: pd.DataFrame, active_model: dict[str, Any] | None) -> list[dict[str, Any]]:
    """PSI per feature between the active model's training window and the recent window."""
    if active_model is None or not active_model.get("training_data_end"):
        return []
    reference_end = pd.Timestamp(active_model["training_data_end"])
    reference_start = reference_end - timedelta(days=DRIFT_REFERENCE_WINDOW_DAYS)
    recent_start = data["observed_at"].max() - timedelta(days=DRIFT_RECENT_WINDOW_DAYS)
    reference = data.loc[(data["observed_at"] > reference_start) & (data["observed_at"] <= reference_end)]
    recent = data.loc[data["observed_at"] > recent_start]
    rows = []
    for feature in DRIFT_FEATURES:
        if feature not in data.columns:
            continue
        psi = population_stability_index(reference[feature].to_numpy(dtype=float), recent[feature].to_numpy(dtype=float))
        rows.append({
            "feature_name": feature, "drift_type": "data", "method": "psi",
            "value": psi, "threshold": PSI_THRESHOLD, "triggered": psi > PSI_THRESHOLD,
        })
    return rows


def decide_action(
    active_model: dict[str, Any] | None,
    performance: dict[int, dict[str, Any]],
    thresholds: dict[str, float],
    drift_rows: list[dict[str, Any]],
    now: pd.Timestamp,
) -> tuple[str, str]:
    """Explicit retrain-vs-keep rule. The first condition that matches wins and is logged verbatim."""
    if active_model is None:
        return "retrain", "bootstrap: no active model"
    stale_days = (now - pd.Timestamp(active_model["trained_at"])).days
    if stale_days > STALENESS_DAYS:
        return "retrain", f"stale: last trained {stale_days} days ago"
    for horizon, result in performance.items():
        threshold = thresholds.get(str(horizon))
        if threshold is not None and result["wape"] > threshold:
            return "retrain", f"performance_drift: h{horizon} recent WAPE {result['wape']:.4f} > threshold {threshold:.4f}"
    for row in drift_rows:
        if row["triggered"]:
            return "retrain", f"data_drift: {row['feature_name']} PSI={row['value']:.4f} > {row['threshold']}"
    return "keep", "stable: no trigger met"


def fetch_active_model(loader: SupabaseLoader) -> dict[str, Any] | None:
    rows = loader.select("model_versions", {"is_active": "eq.true", "limit": 1})
    if not rows:
        return None
    active_model = rows[0]
    training_runs = loader.select("training_runs", {
        "model_version_id": f"eq.{active_model['id']}", "order": "started_at.desc", "limit": 1,
    })
    active_model["training_run"] = training_runs[0] if training_runs else None
    return active_model


def persist_model(url: str, key: str, version: str, models: dict[int, Any], horizons: list[int], metrics: dict[str, Any]) -> None:
    buffer = io.BytesIO()
    joblib.dump({"models": models, "features": FEATURES, "horizons": horizons, "metrics": metrics, "ensemble_weight": ENSEMBLE_WEIGHT}, buffer)
    upload_object(url, key, MODEL_BUCKET, f"{version}.joblib", buffer.getvalue())


def load_active_models(url: str, key: str, active_model: dict[str, Any]) -> dict[int, Any]:
    blob = download_object(url, key, MODEL_BUCKET, f"{active_model['version']}.joblib")
    return joblib.load(io.BytesIO(blob))["models"]


def retrain_and_persist(url: str, key: str, data: pd.DataFrame, horizons: list[int], cycle: dict, reason: str) -> tuple[dict[int, Any], str, str]:
    models, metrics = train_models(data, horizons)
    lineage = record_lineage(cycle["data_cutoff"], cycle["cycle_id"], metrics, trigger_reason=reason)
    if lineage is None:
        raise RuntimeError("record_lineage failed: Supabase credentials missing mid-run.")
    persist_model(url, key, lineage["version"], models, horizons, metrics)
    return models, lineage["model_version_id"], lineage["version"]


def submit_predictions(payload: dict[str, Any], idempotency_key: str, pulso_key: str) -> dict[str, Any]:
    try:
        response = httpx.post(
            f"{PULSO_API_URL}/v1/submissions", json=payload, timeout=30,
            headers={"Idempotency-Key": idempotency_key, "Authorization": f"Bearer {pulso_key}"},
        )
    except httpx.HTTPError as error:
        return {"status": "failed", "response_payload": None, "error_message": str(error), "external_submission_id": None}
    if response.status_code == 201:
        body = response.json()
        return {"status": body.get("status", "accepted"), "response_payload": body, "error_message": None, "external_submission_id": body.get("submission_id")}
    status = "rejected" if response.status_code == 422 else "failed"
    return {"status": status, "response_payload": None, "error_message": f"HTTP {response.status_code}: {response.text[:2000]}", "external_submission_id": None}


def upsert_cycle(loader: SupabaseLoader, cycle: dict) -> None:
    """forecast_runs.cycle_id has a foreign key to forecast_cycles; mirror the API's cycle here first."""
    loader.upsert("forecast_cycles", [{
        "cycle_id": cycle["cycle_id"], "state": cycle["state"], "origin_at": cycle["origin_at"],
        "data_cutoff": cycle["data_cutoff"], "opens_at": cycle.get("opens_at"), "closes_at": cycle.get("closes_at"),
        "expected_predictions": cycle["expected_predictions"],
    }], "cycle_id")
    loader.upsert("forecast_targets", [
        {"cycle_id": cycle["cycle_id"], "station_id": t["station_id"], "target_at": t["target_at"], "horizon_minutes": t["horizon_minutes"]}
        for t in cycle["targets"]
    ], "cycle_id,station_id,target_at")


def target_horizons(cycle: dict) -> dict[tuple[str, str], int]:
    lookup = {}
    for target in cycle["targets"]:
        normalized = pd.Timestamp(target["target_at"]).isoformat().replace("+00:00", "Z")
        lookup[(target["station_id"], normalized)] = int(target["horizon_minutes"])
    return lookup


def main() -> None:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_KEY")
    pulso_key = os.environ.get("PULSO_API_KEY")
    if not url or not key or not pulso_key:
        raise SystemExit("Configure SUPABASE_URL, SUPABASE_SECRET_KEY and PULSO_API_KEY.")

    try:
        cycle = active_cycle()
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 404:
            print("No active cycle: nothing to evaluate, predict or submit.")
            return
        raise
    if cycle["state"] != "open":
        print(f"Active cycle is not open (state={cycle['state']}): skipping.")
        return

    loader = SupabaseLoader(url, key)
    forecast_run_id: str | None = None
    try:
        # A still-open cycle gets checked again every 10 minutes; if it already has an
        # accepted submission there's nothing new to do. Retraining and resubmitting anyway
        # would reuse the same idempotency_key (cycle_id:model_version) with different
        # content (a freshly retrained model), which the API correctly rejects with 409 --
        # and it would burn through SUBMISSION_MAX_ATTEMPTS for no benefit.
        accepted = loader.select("submissions", {"cycle_id": f"eq.{cycle['cycle_id']}", "status": "eq.accepted", "limit": 1})
        if accepted:
            print(f"Cycle {cycle['cycle_id']} already has an accepted submission ({accepted[0]['external_submission_id']}); nothing to do.")
            return

        upsert_cycle(loader, cycle)
        data = add_origin_features(load_training_data())
        now = pd.Timestamp.now(tz="UTC")

        forecast_run = loader.insert("forecast_runs", {
            "cycle_id": cycle["cycle_id"], "data_cutoff": cycle["data_cutoff"], "status": "running", "git_commit": git_commit(),
        })
        forecast_run_id = forecast_run["id"]

        evaluate_recent_predictions(loader, data)

        active_model = fetch_active_model(loader)
        horizons = sorted({int(target["horizon_minutes"]) for target in cycle["targets"]})
        performance = recent_performance(loader, horizons)
        drift_rows = compute_data_drift(data, active_model)

        ingestions = loader.select("ingestion_runs", {"status": "eq.succeeded", "order": "finished_at.desc", "limit": 1})
        ingestion_run_id = ingestions[0]["id"] if ingestions else None
        if ingestion_run_id and drift_rows:
            loader.insert_many("drift_measurements", [
                {**row, "ingestion_run_id": ingestion_run_id, "details": {}} for row in drift_rows
            ])

        thresholds = ((active_model or {}).get("training_run") or {}).get("parameters", {}).get("performance_thresholds", {})
        action, reason = decide_action(active_model, performance, thresholds, drift_rows, now)
        print(f"Decision: {action} ({reason})")

        if action == "retrain":
            models, model_version_id, model_version = retrain_and_persist(url, key, data, horizons, cycle, reason)
        else:
            try:
                models = load_active_models(url, key, active_model)
                model_version_id, model_version = active_model["id"], active_model["version"]
            except Exception as error:
                reason = f"keep failed, fallback to retrain: {error}"
                action = "retrain"
                print(f"Decision revised: retrain ({reason})")
                models, model_version_id, model_version = retrain_and_persist(url, key, data, horizons, cycle, reason)

        loader.patch("forecast_runs", forecast_run_id, {"model_version_id": model_version_id, "trigger_reason": reason})

        predictions = prediction_rows(data, cycle, models)

        cutoff = pd.Timestamp(cycle["data_cutoff"])
        correction_enabled = os.environ.get("PULSO_BIAS_CORRECTION", "true").strip().lower() != "false"
        recent = loader.select("predictions", {
            "select": "predicted_demand,raw_predicted_demand,actual_demand", "actual_demand": "not.is.null",
            "target_at": [f"gt.{(cutoff - timedelta(hours=BIAS_WINDOW_HOURS)).isoformat()}", f"lte.{cutoff.isoformat()}"],
        })
        factor, samples = bias_factor(recent)
        scale = bias_scale(factor) if correction_enabled else 1.0
        raw_values = {(p["station_id"], p["target_at"]): p["value"] for p in predictions}
        for p in predictions:
            p["value"] = float(p["value"] * scale)
        print(f"Bias correction: factor={factor if factor is None else round(factor, 4)} samples={samples} scale={scale:.4f} enabled={correction_enabled}")
        if ingestion_run_id and factor is not None:
            loader.insert("drift_measurements", {
                "ingestion_run_id": ingestion_run_id, "feature_name": "prediction_bias", "drift_type": "performance",
                "method": f"relative_bias_{BIAS_WINDOW_HOURS}h", "value": factor - 1, "threshold": BIAS_ALERT,
                "triggered": abs(factor - 1) > BIAS_ALERT,
                "details": {"factor": factor, "applied_scale": scale, "samples": samples, "alpha": BIAS_ALPHA, "enabled": correction_enabled},
            })

        horizon_lookup = target_horizons(cycle)
        loader.insert_many("predictions", [
            {
                "forecast_run_id": forecast_run_id, "station_id": p["station_id"], "target_at": p["target_at"],
                "horizon_minutes": horizon_lookup[(p["station_id"], p["target_at"])], "predicted_demand": p["value"],
                "raw_predicted_demand": raw_values[(p["station_id"], p["target_at"])],
            }
            for p in predictions
        ])

        payload = {
            "schema_version": "1.0", "cycle_id": cycle["cycle_id"],
            "client_run_id": f"forecast-cycle-{uuid4().hex[:12]}", "data_cutoff": cycle["data_cutoff"],
            "model": {"version": model_version, "training_data_end": cycle["data_cutoff"], "git_commit": git_commit()},
            "predictions": predictions,
        }
        errors = validate(payload, cycle)
        if errors:
            raise RuntimeError("Local payload validation failed: " + "; ".join(errors))

        submit_enabled = os.environ.get("PULSO_SUBMIT_ENABLED", "true").strip().lower() != "false"
        idempotency_key = f"{cycle['cycle_id']}:{model_version}"[:128]
        if submit_enabled:
            submit_result = submit_predictions(payload, idempotency_key, pulso_key)
        else:
            submit_result = {"status": "pending", "response_payload": None, "error_message": "PULSO_SUBMIT_ENABLED=false", "external_submission_id": None}
            print("Submission skipped: PULSO_SUBMIT_ENABLED=false.")

        # Repeated runs within the same still-open cycle, with no reason to retrain again,
        # reuse the same idempotency_key -- upsert instead of insert so that doesn't 409.
        loader.upsert("submissions", [{
            "external_submission_id": submit_result["external_submission_id"],
            "forecast_run_id": forecast_run_id, "cycle_id": cycle["cycle_id"],
            "client_run_id": payload["client_run_id"], "idempotency_key": idempotency_key,
            "submitted_at": now.isoformat() if submit_enabled else None,
            "status": submit_result["status"], "response_payload": submit_result["response_payload"],
            "error_message": submit_result["error_message"],
        }], "idempotency_key")

        loader.patch("forecast_runs", forecast_run_id, {"status": "succeeded", "finished_at": pd.Timestamp.now(tz="UTC").isoformat()})
        print(f"Forecast run complete: action={action}, predictions={len(predictions)}, submission={submit_result['status']}.")
    except Exception as error:
        if forecast_run_id:
            loader.patch("forecast_runs", forecast_run_id, {
                "status": "failed", "finished_at": pd.Timestamp.now(tz="UTC").isoformat(), "error_message": str(error),
            })
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    main()
