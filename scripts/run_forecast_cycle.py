"""Operate the forecast cycle: evaluate recent accuracy, measure drift, decide whether
to keep or retrain the active model, predict, submit, and log the outcome.

Runs after scripts/sync_stream_observations.py in the same GitHub Actions job, so the
data it reads is already up to date. Writes to the operational tables the initial schema
already provisioned: forecast_runs, predictions, drift_measurements, submissions.
See docs/collector-and-lineage.md for the decision rule and its thresholds.

Set PULSO_SUBMIT_ENABLED=false to run the full pipeline (evaluate, decide, predict,
validate) without sending the real POST to the competition API.

The online bias correction (scale by half of actual/predicted over the last 4h of evaluated
predictions) is measured and logged every cycle but only applied with
PULSO_BIAS_CORRECTION=true: see docs/ml-baselines.md for why it is off by default.
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
from forecast_adjustments import (
    EXPERTS, LAG_EXPERTS, LONG_LAG_EXPERTS, LONG_PERIODIC_EXPERTS, PERIODIC_EXPERTS, SHIFT_EXPERTS, TREND_EXPERTS,
    production_adjust,
)
from load_supabase import SupabaseLoader, delete_objects, download_object, list_objects, load_dotenv, upload_object
from train_baseline import load_training_data, station_metrics
from train_catboost_direct import (
    FEATURES,
    ENSEMBLE_WEIGHT,
    LATEST_OBSERVATIONS_INGESTION,
    MODEL_FORMAT,
    add_origin_features,
    git_commit,
    prediction_rows,
    record_lineage,
    train_models,
)

MODEL_BUCKET = "models"
# Every retrain uploads a ~10 MB bundle and nothing ever removed old ones: with a retrain per
# cycle that filled the 1 GB free Storage tier (1.16 GB on 2026-09-30). Keep the active model,
# the most recent few (for rollback) and the first few (the original baselines); MLflow's
# Model Registry keeps the full history anyway.
MODELS_TO_KEEP = 5
OLDEST_MODELS_TO_KEEP = 5
STALENESS_DAYS = 7
PERFORMANCE_DEGRADATION_FACTOR = 1.15
PERFORMANCE_MIN_SAMPLES = 20
PERFORMANCE_WINDOW_DAYS = 3
DRIFT_RECENT_WINDOW_DAYS = 3
DRIFT_REFERENCE_WINDOW_DAYS = 14
# Below this many real (non-null) observations in the recent window, PSI is noise, not signal:
# context (weather/events) lags observations by days (scripts/sync_context.py), so a 3-day
# window can be mostly null -- found live on 2026-09-23, temperature_forecast's PSI swinging
# 0.07-0.29 between consecutive 10-min runs from a shrinking, shifting sample, not real weather
# change. context is one row per timestamp (not per station), so a full window is ~3*96=288;
# 150 is roughly half of that.
DRIFT_MIN_VALID_SAMPLES = 150
PSI_THRESHOLD = 0.25
# event_intensity is sparse (rare events): any 3-day window without events differs a lot from a
# 14-day one that had some, so its PSI sat at 1.16-1.18 in all 18 measurements and triggered a
# retrain every single cycle. 2.0 still catches a real jump above that background level.
PSI_THRESHOLD_OVERRIDES = {"event_intensity": 2.0}
DRIFT_FEATURES = ["demand", "rain_forecast", "temperature_forecast", "event_intensity"]
# demand varies per station; weather/events are one value per timestamp, broadcast to all 12
# stations by the merge in load_training_data() -- dedupe those before counting/scoring so 12
# copies of the same stale value don't look like a healthy sample.
PER_TIMESTAMP_FEATURES = {"rain_forecast", "temperature_forecast", "event_intensity"}
# Per-station level drift. PSI on `demand` pools all 12 stations, so when the competition moved
# riders between stations (05100 down to 0.2x of the previous week, 05000/02300 up 2-3x) the
# pooled distribution barely changed (PSI 0.01-0.04) and no data-drift alarm ever fired. This
# compares each station's last 24h against the same 24h a week earlier, now vs. at the active
# model's training_data_end: the change in that week-over-week ratio is what the model hasn't
# seen yet. After a retrain the reference moves to the new cutoff, so a persistent change fires
# once, not every cycle. On the real data, x1.5 clears the natural variation of stable stations
# (up to x1.45 within 12h) and catches 05100, 05000, 02300 and 03000.
STATION_DRIFT_WINDOW = timedelta(hours=24)
STATION_DRIFT_THRESHOLD = float(np.log(1.5))
STATION_DRIFT_PREFIX = "station_level:"
# Post-model adjustment (scripts/forecast_adjustments.py), chosen with
# scripts/backtest_adjustments.py on real data (12-18 sep, walk-forward, hourly origins) and a
# replay on the live predictions: "ensemble" weights the model, persistence, the comparable day
# and copies of the series 2-6h back by inverse error^6 over the last 4h, one set of weights per
# horizon across stations. From 18-sep 05:00 the injected drift is a 4h-periodic oscillation the
# model can't learn; once one period has been seen the 4h copy takes over (~90% vs ~60% for the
# model on the live cycles), and on normal days the model keeps ~all the weight (85.8% -> 86.0%).
# The first version (model / persistence / comparable day only, 6h, p3) lost to the model live
# and was reverted. "blend_cap" and "none" stay available through PULSO_ADJUSTMENT.
DEFAULT_ADJUSTMENT = "ensemble"
# 2h since 2026-10-04: after a regime change the weights catch up an hour or two sooner (87.4% vs
# 84.5% on the first 8h-regime cycles) for 0.26 points in a steady regime (92.80 vs 93.06).
ENSEMBLE_WINDOW = timedelta(hours=2)
ENSEMBLE_POWER = 6.0
ENSEMBLE_POOL_STATIONS = True
ENSEMBLE_ON_ADJUSTED = False
# Single-period copies (lag_Ph) and their averages over the last 24h (per_Ph): once a periodic
# regime has run for a day the averages cancel the copies' noise (+1.5 points per cycle on the
# live regime, 91.3% -> 92.8%), while at a regime's onset the averages still mix pre-regime
# periods and the copies lead (backtest_periodic.py replay, docs/ml-baselines.md).
# Drop the periodic experts for a cycle when copying any period has stopped working over the last
# hour (forecast_adjustments.periodic_break). Replayed on 29 live cycles: never fired during the
# 4h regime (93.05% either way), fired on the first cycle after it ended (68.8% vs 10.8% sent).
ENSEMBLE_BREAK_GUARD = True
# Damped-trend persistence (forecast_adjustments.TREND_DAMPING): replayed on 40 live cycles it is
# neutral in the 4h regime (92.97 vs 92.98) and adds ~1.2 points per post-break cycle.
# 7-12h periods and level-shifted copies: the regime after the 4h one repeats every 8h. Replayed on
# 44 live cycles with a 2h weight window: 87.4% vs 78.4% on the 6 cycles where the 8h copy is
# usable, 92.80 vs 93.06 in the 4h regime.
ENSEMBLE_EXPERTS = (
    *EXPERTS, *LAG_EXPERTS, *PERIODIC_EXPERTS, *TREND_EXPERTS, *LONG_LAG_EXPERTS, *LONG_PERIODIC_EXPERTS, *SHIFT_EXPERTS,
)
ADJUST_CAP_THRESHOLD = 2.5
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
    pending = loader.select_all("predictions", {
        "actual_demand": "is.null", "target_at": f"lte.{now.isoformat()}", "select": "id,station_id,target_at",
    }, order="id.asc")
    if not pending:
        return
    pending_frame = pd.DataFrame(pending)
    pending_frame["target_at"] = pd.to_datetime(pending_frame["target_at"], utc=True)
    lookup = data.set_index(["station_id", "observed_at"])["demand"]
    for row in pending_frame.itertuples():
        key = (row.station_id, row.target_at)
        if key in lookup.index:
            loader.patch("predictions", row.id, {"actual_demand": int(lookup.loc[key]), "evaluated_at": now.isoformat()})


def recent_performance_params(since: str, model_version_id: str | None) -> dict[str, Any]:
    """Only predictions made by the active model count against its own threshold.

    Pooling every model's predictions meant a replaced model's errors kept firing
    performance_drift for the whole 3-day window, retraining the new model every cycle no
    matter how it did (seen after the late-September demand shift: 60+ retrains in a row).
    A fresh model has no evaluated predictions yet, so it is kept until it has
    PERFORMANCE_MIN_SAMPLES per horizon (two cycles) and then judged on its own record.
    """
    params: dict[str, Any] = {
        "evaluated_at": f"gte.{since}", "select": "station_id,horizon_minutes,predicted_demand,raw_predicted_demand,actual_demand",
    }
    if model_version_id:
        params["select"] += ",forecast_runs!inner(model_version_id)"
        params["forecast_runs.model_version_id"] = f"eq.{model_version_id}"
    return params


def recent_performance(
    loader: SupabaseLoader, horizons: list[int], model_version_id: str | None = None, score_submitted: bool = False,
) -> dict[int, dict[str, Any]]:
    """WAPE per horizon from the active model's predictions evaluated within the last PERFORMANCE_WINDOW_DAYS.

    By default scores the raw model output, so the bias correction can't hide model degradation
    from the retrain rule (the threshold it's compared to is the raw model's validation WAPE).
    score_submitted=True scores what was actually submitted instead: with the adaptive ensemble
    on, the raw model can sit above its threshold for as long as an injected regime lasts (it
    can't learn the 4h oscillation) while the submitted values are fine, and judging the raw
    model then retrained every two cycles for nothing (12 retrains in 24h on 1-oct).
    """
    since = (pd.Timestamp.now(tz="UTC") - timedelta(days=PERFORMANCE_WINDOW_DAYS)).isoformat()
    # 3 days x 48 predictions per cycle passes PostgREST's 1000-row cap within a day: page it.
    rows = loader.select_all("predictions", recent_performance_params(since, model_version_id), order="id.asc")
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    if not score_submitted:
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
        ref_col, recent_col = reference[feature], recent[feature]
        if feature in PER_TIMESTAMP_FEATURES:
            ref_col = reference.drop_duplicates("observed_at")[feature]
            recent_col = recent.drop_duplicates("observed_at")[feature]
        valid_recent = int(recent_col.notna().sum())
        threshold = PSI_THRESHOLD_OVERRIDES.get(feature, PSI_THRESHOLD)
        psi = population_stability_index(ref_col.to_numpy(dtype=float), recent_col.to_numpy(dtype=float))
        insufficient = valid_recent < DRIFT_MIN_VALID_SAMPLES
        rows.append({
            "feature_name": feature, "drift_type": "data", "method": "psi",
            "value": psi, "threshold": threshold, "triggered": (not insufficient) and psi > threshold,
            "details": {"valid_samples": valid_recent, "min_required": DRIFT_MIN_VALID_SAMPLES, **({"insufficient_samples": True} if insufficient else {})},
        })
    return rows


def week_log_ratios(data: pd.DataFrame, end: pd.Timestamp) -> dict[str, float]:
    """ln(demand in the 24h up to `end` / demand in the same 24h a week earlier), per station.

    Stations missing any period in either window are left out rather than scored on a gap.
    """
    periods = int(STATION_DRIFT_WINDOW / timedelta(minutes=15))
    recent = data.loc[(data["observed_at"] > end - STATION_DRIFT_WINDOW) & (data["observed_at"] <= end)]
    week_end = end - timedelta(days=7)
    before = data.loc[(data["observed_at"] > week_end - STATION_DRIFT_WINDOW) & (data["observed_at"] <= week_end)]
    result = {}
    for station, now_rows in recent.groupby("station_id"):
        then_rows = before.loc[before["station_id"] == station]
        if len(now_rows) < periods or len(then_rows) < periods:
            continue
        now_sum, then_sum = float(now_rows["demand"].sum()), float(then_rows["demand"].sum())
        if now_sum > 0 and then_sum > 0:
            result[str(station)] = float(np.log(now_sum / then_sum))
    return result


def compute_station_drift(data: pd.DataFrame, active_model: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One row per station: |change in week-over-week log ratio| since the active model's training cutoff."""
    if active_model is None or not active_model.get("training_data_end"):
        return []
    at_training = week_log_ratios(data, pd.Timestamp(active_model["training_data_end"]))
    now = week_log_ratios(data, data["observed_at"].max())
    rows = []
    for station in sorted(set(at_training) & set(now)):
        change = now[station] - at_training[station]
        rows.append({
            "feature_name": f"{STATION_DRIFT_PREFIX}{station}", "drift_type": "data", "method": "week_ratio_shift_24h",
            "value": abs(change), "threshold": STATION_DRIFT_THRESHOLD, "triggered": abs(change) > STATION_DRIFT_THRESHOLD,
            "details": {
                "station_id": station, "ratio_now": float(np.exp(now[station])),
                "ratio_at_training": float(np.exp(at_training[station])), "relative_change": float(np.exp(change)),
            },
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
            if row["feature_name"].startswith(STATION_DRIFT_PREFIX):
                change = (row.get("details") or {}).get("relative_change")
                detail = f"x{change:.2f} vs training" if change is not None else f"{row['value']:.4f}"
                return "retrain", f"data_drift: {row['feature_name']} {detail} > x{np.exp(row['threshold']):.2f}"
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
    joblib.dump({
        "models": models, "features": FEATURES, "horizons": horizons, "metrics": metrics,
        "ensemble_weight": ENSEMBLE_WEIGHT, "model_format": MODEL_FORMAT,
    }, buffer)
    upload_object(url, key, MODEL_BUCKET, f"{version}.joblib", buffer.getvalue())


def models_to_prune(
    objects: list[dict[str, Any]], keep: set[str], keep_latest: int = MODELS_TO_KEEP, keep_oldest: int = OLDEST_MODELS_TO_KEEP,
) -> list[str]:
    """Names of stored model bundles to delete: all but `keep`, the `keep_latest` newest and the `keep_oldest` oldest."""
    bundles = sorted(
        (obj for obj in objects if str(obj.get("name", "")).endswith(".joblib")),
        key=lambda obj: obj.get("created_at") or "", reverse=True,
    )
    protected = {obj["name"] for obj in bundles[:keep_latest]} | {obj["name"] for obj in bundles[len(bundles) - keep_oldest:]} | keep
    return [obj["name"] for obj in bundles if obj["name"] not in protected]


def prune_stored_models(url: str, key: str, active_version: str) -> None:
    """Best effort: a failed cleanup must never block predicting or submitting."""
    try:
        stale = models_to_prune(list_objects(url, key, MODEL_BUCKET), {f"{active_version}.joblib"}) if active_version else []
        delete_objects(url, key, MODEL_BUCKET, stale)
        if stale:
            print(f"Pruned {len(stale)} old model bundles from Storage.")
    except Exception as error:
        print(f"Model storage cleanup skipped: {error}")


def load_active_models(url: str, key: str, active_model: dict[str, Any]) -> dict[int, Any]:
    blob = download_object(url, key, MODEL_BUCKET, f"{active_model['version']}.joblib")
    bundle = joblib.load(io.BytesIO(blob))
    # A bundle from before the level-ratio model predicts in different units with different
    # features: raising here sends the caller down its "keep failed, fallback to retrain" path.
    if bundle.get("model_format") != MODEL_FORMAT:
        raise RuntimeError(f"active model format {bundle.get('model_format') or 'raw-v0'} != {MODEL_FORMAT}")
    return bundle["models"]


def retrain_and_persist(url: str, key: str, data: pd.DataFrame, horizons: list[int], cycle: dict, reason: str) -> tuple[dict[int, Any], str, str]:
    models, metrics = train_models(data, horizons)
    lineage = record_lineage(cycle["data_cutoff"], cycle["cycle_id"], metrics, trigger_reason=reason, models=models, data=data)
    if lineage is None:
        raise RuntimeError("record_lineage failed: Supabase credentials missing mid-run.")
    persist_model(url, key, lineage["version"], models, horizons, metrics)
    prune_stored_models(url, key, lineage["version"])
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


def fill_gaps(data: pd.DataFrame) -> pd.DataFrame:
    """Put every station on the full 15-min grid, carrying its last known values forward.

    Since 2026-10-04 the API leaves some station-periods out of the stream (11 of 12 stations at
    the 15:00 cutoff); the model needs all 12 at the cutoff and gap-free lags, so the cycle failed.
    """
    times = pd.date_range(data["observed_at"].min(), data["observed_at"].max(), freq="15min")
    stations = sorted(data["station_id"].unique())
    grid = pd.MultiIndex.from_product([stations, times], names=["station_id", "observed_at"]).to_frame(index=False)
    filled = grid.merge(data, on=["station_id", "observed_at"], how="left").sort_values(["station_id", "observed_at"])
    missing = int(filled["demand"].isna().sum())
    if missing:
        value_columns = [c for c in filled.columns if c not in ("station_id", "observed_at")]
        filled[value_columns] = filled.groupby("station_id")[value_columns].ffill()
        print(f"Filled {missing} missing station-periods with each station's last known values.")
    return filled.dropna(subset=["demand"]).reset_index(drop=True)


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
        data = add_origin_features(fill_gaps(load_training_data()))
        now = pd.Timestamp.now(tz="UTC")

        forecast_run = loader.insert("forecast_runs", {
            "cycle_id": cycle["cycle_id"], "data_cutoff": cycle["data_cutoff"], "status": "running", "git_commit": git_commit(),
        })
        forecast_run_id = forecast_run["id"]

        evaluate_recent_predictions(loader, data)

        active_model = fetch_active_model(loader)
        horizons = sorted({int(target["horizon_minutes"]) for target in cycle["targets"]})
        adjustment = os.environ.get("PULSO_ADJUSTMENT", "").strip().lower() or DEFAULT_ADJUSTMENT
        # While the ensemble decides what is submitted, judge that: retraining can't fix a raw
        # model that the injected regime defeats, and the ensemble already routes around it.
        performance = recent_performance(loader, horizons, (active_model or {}).get("id"), score_submitted=adjustment == "ensemble")
        drift_rows = compute_data_drift(data, active_model) + compute_station_drift(data, active_model)

        ingestions = loader.select("ingestion_runs", LATEST_OBSERVATIONS_INGESTION)
        ingestion_run_id = ingestions[0]["id"] if ingestions else None
        if ingestion_run_id and drift_rows:
            loader.insert_many("drift_measurements", [
                {**row, "ingestion_run_id": ingestion_run_id} for row in drift_rows
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
        horizon_lookup = target_horizons(cycle)
        model_values = {(p["station_id"], p["target_at"]): p["value"] for p in predictions}
        if adjustment != "none":
            history = loader.select_all("predictions", {
                "select": "station_id,horizon_minutes,target_at,raw_predicted_demand,predicted_demand,actual_demand",
                "actual_demand": "not.is.null",
                "target_at": [f"gt.{(cutoff - ENSEMBLE_WINDOW).isoformat()}", f"lte.{cutoff.isoformat()}"],
            }, order="id.asc")
            history_frame = pd.DataFrame(history, columns=["station_id", "horizon_minutes", "target_at", "raw_predicted_demand", "predicted_demand", "actual_demand"])
            history_frame = (
                history_frame.assign(model=history_frame["raw_predicted_demand"].fillna(history_frame["predicted_demand"]), actual=history_frame["actual_demand"])
                .groupby(["station_id", "horizon_minutes", "target_at"], as_index=False)[["model", "actual"]].mean()
            )
            frame = pd.DataFrame({
                "station_id": [p["station_id"] for p in predictions], "observed_at": cutoff,
                "target_at": pd.to_datetime([p["target_at"] for p in predictions], utc=True),
                "horizon_minutes": [horizon_lookup[(p["station_id"], p["target_at"])] for p in predictions],
                "model": [p["value"] for p in predictions],
            })
            values, mean_weights = production_adjust(
                data, frame, history_frame, adjustment, ENSEMBLE_WINDOW, ENSEMBLE_POWER,
                ensemble_on_adjusted=ENSEMBLE_ON_ADJUSTED, threshold=ADJUST_CAP_THRESHOLD, pool_stations=ENSEMBLE_POOL_STATIONS,
                experts=ENSEMBLE_EXPERTS, break_guard=ENSEMBLE_BREAK_GUARD,
            )
            for p, value in zip(predictions, values, strict=True):
                p["value"] = float(value)
            print(f"Adjustment: mode={adjustment} history_rows={len(history_frame)} mean_weights={mean_weights}")

        # Off unless PULSO_BIAS_CORRECTION=true: with the level-normalised model it no longer
        # helped on the real post-change period (the pooled factor mixes stations moving in
        # opposite directions). The factor is still measured and logged below for monitoring.
        correction_enabled = os.environ.get("PULSO_BIAS_CORRECTION", "").strip().lower() == "true"
        recent = loader.select("predictions", {
            "select": "predicted_demand,raw_predicted_demand,actual_demand", "actual_demand": "not.is.null",
            "target_at": [f"gt.{(cutoff - timedelta(hours=BIAS_WINDOW_HOURS)).isoformat()}", f"lte.{cutoff.isoformat()}"],
        })
        factor, samples = bias_factor(recent)
        scale = bias_scale(factor) if correction_enabled else 1.0
        # raw_predicted_demand keeps the model's own output (before adjustment and correction): it is
        # what performance_drift and the bias factor measure, and the "model" expert's history.
        raw_values = model_values
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
