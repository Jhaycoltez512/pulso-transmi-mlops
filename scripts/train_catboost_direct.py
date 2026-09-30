"""Train direct multi-horizon CatBoost models and build a validated submission preview.

The active Pulso cycle defines the requested horizons. A separate model is fitted
for every horizon, so predictions never depend recursively on prior predictions.
This script only writes local artifacts; it never submits them to the API.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import httpx
from catboost import CatBoostRegressor

from generate_weekly_submission_preview import PULSO_API_URL, active_cycle, validate
from load_supabase import SupabaseLoader, load_dotenv
from mlflow_tracking import log_run
from train_baseline import ARTIFACTS_DIR, load_training_data, station_metrics

LAGS = [1, 2, 4, 8, 96, 192, 672]
ROLLING_WINDOWS = [4, 12, 96, 672]
CALENDAR_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_weekend"]
DEMAND_FEATURES = ["current_demand", *[f"lag_{lag}" for lag in LAGS], *[f"rolling_mean_{window}" for window in ROLLING_WINDOWS]]
# Features of the original model, in raw demand units. Kept for the backtests that document
# experiments run on that model (recency weighting, seasonal features).
RAW_FEATURES = ["station_id", *CALENDAR_FEATURES, *DEMAND_FEATURES]
# Level-normalised model (adopted 2026-09-30). Every demand feature and the target are divided
# by the station's mean demand over the last LEVEL_WINDOW periods, and the prediction is
# multiplied back. A tree model can't extrapolate: when the competition shifted demand ~35%
# up, the raw-units model kept predicting inside its old range (recent WAPE 0.26-0.28 against
# a 0.13 validation, bias factor 1.3-1.4). Ratios are unchanged by a level shift, so the same
# model keeps working. Validated offline on real data with an injected shift (see
# docs/ml-baselines.md): same WAPE as the raw model when stable, ~0.12-0.14 instead of
# ~0.15-0.17 (bias-corrected) or ~0.19-0.22 (uncorrected) under the shift.
LEVEL_WINDOW = 16  # 4 hours; 1 hour was noisier at h60
FEATURES = ["station_id", *CALENDAR_FEATURES, *[f"norm_{feature}" for feature in DEMAND_FEATURES]]
MODEL_FORMAT = "level-ratio-v1"
WEEKLY_NAIVE_LAG = timedelta(days=7)
WEEK_PERIODS = 672
# The weekly naive is rescaled by how the last LEVEL_WINDOW periods compare with the same
# periods a week earlier, so it follows a level shift too instead of anchoring on last week.
WEEK_RATIO_CLIP = (0.5, 2.0)
# Share given to the CatBoost prediction vs. the weekly seasonal-naive prediction. Picked with
# scripts/backtest_ensemble.py: a walk-forward sweep found 0.70-0.85 near-optimal at every horizon,
# with WAPE gains over pure CatBoost well above the noise between folds (see docs/ml-baselines.md).
ENSEMBLE_WEIGHT = 0.75
# MLflow Model Registry name; the `champion` alias always points at the active version.
REGISTERED_MODEL_NAME = "pulso-catboost"
# ingestion_runs.source_name of the observations collector. sync_context.py also writes
# ingestion_runs (without data_version) right after it, so lineage must filter by source or
# it links the context sync instead of the data the model was trained on -- which is what
# left data_version=unknown on every retrain since context sync was added. Collector runs
# that found no new rows also have no data_version; the latest one that did is the version.
OBSERVATIONS_SOURCE = "pulso-transmi-stream"
LATEST_OBSERVATIONS_INGESTION = {
    "status": "eq.succeeded", "source_name": f"eq.{OBSERVATIONS_SOURCE}", "data_version": "not.is.null",
    "order": "finished_at.desc", "limit": 1,
}


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def performance_thresholds(metrics: dict[str, Any], factor: float = 1.15) -> dict[str, float]:
    """Recent-WAPE ceiling per horizon: this model's own validation WAPE plus a relative margin."""
    return {horizon: result["validation"]["mean_station_wape"] * factor for horizon, result in metrics.items()}


def record_lineage(
    data_cutoff: str, cycle_id: str, metrics: dict[str, Any], trigger_reason: str = "direct-multihorizon-training",
    models: dict[int, CatBoostRegressor] | None = None, data: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Link the model artifact, dataset version, metrics and optional MLflow run. Marks the new version active."""
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("Supabase model lineage skipped: credentials are not configured.")
        return None
    loader = SupabaseLoader(url, key)
    try:
        ingestions = loader.select("ingestion_runs", LATEST_OBSERVATIONS_INGESTION)
        data_version = ingestions[0].get("data_version") if ingestions else None
        version = f"catboost-{pd.Timestamp(data_cutoff).strftime('%Y%m%dT%H%M%SZ')}"
        deactivate = loader.client.patch("/model_versions", params={"is_active": "eq.true"}, json={"is_active": False})
        deactivate.raise_for_status()
        model_row = {
            "name": "catboost-direct", "version": version, "algorithm": "CatBoostRegressor+WeeklyNaiveBlend",
            "artifact_uri": "artifacts/catboost_direct.joblib", "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "training_data_end": data_cutoff, "git_commit": git_commit(), "data_version": data_version, "is_active": True,
        }
        loader.upsert("model_versions", [model_row], "version")
        model_version = loader.select("model_versions", {"version": f"eq.{version}", "limit": 1})[0]
        training_run = loader.insert("training_runs", {
            "model_version_id": model_version["id"], "training_end": data_cutoff,
            "trigger_reason": trigger_reason, "status": "succeeded",
            "parameters": {
                "horizons": sorted(map(int, metrics)), "features": FEATURES, "ensemble_weight": ENSEMBLE_WEIGHT,
                "model_format": MODEL_FORMAT, "level_window": LEVEL_WINDOW,
                "performance_thresholds": performance_thresholds(metrics),
            },
        })
        for horizon, result in metrics.items():
            for split in ("validation", "test"):
                loader.upsert("model_metrics", [{
                    "training_run_id": training_run["id"], "station_id": None, "split_name": f"h{horizon}_{split}",
                    "metric_name": "mean_station_accuracy", "metric_value": result[split]["mean_station_accuracy"],
                }, {
                    "training_run_id": training_run["id"], "station_id": None, "split_name": f"h{horizon}_{split}",
                    "metric_name": "mean_station_wape", "metric_value": result[split]["mean_station_wape"],
                }], "training_run_id,station_id,split_name,metric_name")
        flat_metrics = {f"h{h}_{split}_{metric}": values[split][metric] for h, values in metrics.items() for split in ("validation", "test") for metric in ("mean_station_accuracy", "mean_station_wape")}
        # Data is versioned to MLflow here too, in the same run as the model -- not on every
        # collector sync -- so DagsHub only gets a new artifact set when a retrain actually happens.
        ingestion = ingestions[0] if ingestions else {}
        data_manifest = {
            "data_version": data_version, "cycle_id": cycle_id, "data_cutoff": data_cutoff,
            "ingestion_run_id": ingestion.get("id"), "observation_rows_read": ingestion.get("observation_rows_read"),
            "last_observed_at": ingestion.get("last_observed_at"), "ingestion_finished_at": ingestion.get("finished_at"),
        }
        try:
            mlflow_id = log_run(
                run_name=version, tags={"model": "catboost-direct", "data_version": data_version or "unknown", "git_commit": git_commit() or "unknown"},
                params={"horizons": sorted(map(int, metrics)), "feature_count": len(FEATURES)}, metrics=flat_metrics,
                artifacts={"metrics.json": metrics, "data_manifest.json": data_manifest},
                artifact_paths=[ARTIFACTS_DIR / "catboost_direct.joblib", ARTIFACTS_DIR / "catboost_direct_metrics.json"],
                model_bundle=(
                    {"models": models, "features": FEATURES, "horizons": sorted(models), "ensemble_weight": ENSEMBLE_WEIGHT, "model_format": MODEL_FORMAT}
                    if models else None
                ),
                registered_model_name=REGISTERED_MODEL_NAME,
                dataset=data, dataset_name=version,
            )
            if mlflow_id:
                loader.patch("model_versions", model_version["id"], {"mlflow_run_id": mlflow_id})
                loader.patch("training_runs", training_run["id"], {"mlflow_run_id": mlflow_id})
                if ingestion.get("id"):
                    loader.patch("ingestion_runs", ingestion["id"], {"mlflow_run_id": mlflow_id})
        except Exception as error:
            print(f"MLflow model tracking skipped: {error}")
        print(f"Model lineage recorded: version={version}, data_version={data_version or 'unknown'}.")
        return {"version": version, "model_version_id": model_version["id"], "training_run_id": training_run["id"]}
    finally:
        loader.close()


def add_origin_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features available at the forecast origin, never from future demand."""
    data = frame.sort_values(["station_id", "observed_at"]).copy()
    grouped = data.groupby("station_id")["demand"]
    data["current_demand"] = data["demand"]
    for lag in LAGS:
        data[f"lag_{lag}"] = grouped.shift(lag)
    for window in ROLLING_WINDOWS:
        data[f"rolling_mean_{window}"] = grouped.transform(lambda values: values.shift(1).rolling(window, min_periods=window).mean())
    # Recent level, including the origin itself: the scale every demand feature is divided by.
    data["level"] = grouped.transform(lambda values: values.rolling(LEVEL_WINDOW, min_periods=LEVEL_WINDOW).mean()).clip(lower=1.0)
    for feature in DEMAND_FEATURES:
        data[f"norm_{feature}"] = data[feature] / data["level"]
    recent_sum = grouped.transform(lambda values: values.rolling(LEVEL_WINDOW, min_periods=LEVEL_WINDOW).sum())
    week_before_sum = grouped.transform(lambda values: values.shift(WEEK_PERIODS).rolling(LEVEL_WINDOW, min_periods=LEVEL_WINDOW).sum())
    data["week_ratio"] = (recent_sum / week_before_sum).clip(*WEEK_RATIO_CLIP)
    return data


def add_target_calendar(data: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    result = data.copy()
    result["target_at"] = result["observed_at"] + pd.to_timedelta(horizon_minutes, unit="m")
    local = result["target_at"].dt.tz_convert("America/Bogota")
    quarter = local.dt.hour * 4 + local.dt.minute // 15
    result["hour_sin"] = np.sin(2 * np.pi * quarter / 96)
    result["hour_cos"] = np.cos(2 * np.pi * quarter / 96)
    result["weekday_sin"] = np.sin(2 * np.pi * local.dt.dayofweek / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * local.dt.dayofweek / 7)
    result["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
    result["target_demand"] = result.groupby("station_id")["demand"].shift(-horizon_minutes // 15)
    return result.dropna(subset=[*FEATURES, "target_demand"]).reset_index(drop=True)


def naive_prediction_at_target(data: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Demand seven days before each row's target_at: the weekly seasonal-naive forecast."""
    lookup = data[["station_id", "observed_at", "demand"]].rename(columns={"observed_at": "naive_at", "demand": "naive_prediction"})
    keys = pd.DataFrame({"station_id": frame["station_id"].to_numpy(), "naive_at": (frame["target_at"] - WEEKLY_NAIVE_LAG).to_numpy()})
    return keys.merge(lookup, on=["station_id", "naive_at"], how="left")["naive_prediction"].to_numpy()


def adjusted_naive_at_target(data: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Weekly naive scaled by the recent level vs. the same hours a week before (1.0 if unknown)."""
    return naive_prediction_at_target(data, frame) * frame["week_ratio"].fillna(1.0).to_numpy()


def blend_predictions(catboost_pred: np.ndarray, naive_pred: np.ndarray, weight: float = ENSEMBLE_WEIGHT) -> np.ndarray:
    """Average CatBoost with the weekly-naive forecast; fall back to CatBoost alone where naive is unavailable."""
    naive_filled = np.where(np.isnan(naive_pred), catboost_pred, naive_pred)
    return np.clip(weight * catboost_pred + (1 - weight) * naive_filled, 0, None)


def split_by_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maximum = frame["target_at"].max()
    test_start = maximum - timedelta(days=7)
    validation_start = test_start - timedelta(days=7)
    train = frame.loc[frame.target_at <= validation_start].copy()
    validation = frame.loc[(frame.target_at > validation_start) & (frame.target_at <= test_start)].copy()
    test = frame.loc[frame.target_at > test_start].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("Not enough chronological data for training, validation and test windows.")
    return train, validation, test


def model() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE", iterations=600, depth=8, learning_rate=0.05,
        l2_leaf_reg=5, random_seed=42, verbose=False, allow_writing_files=False,
    )


def fit_model(frame: pd.DataFrame) -> CatBoostRegressor:
    """Fit on target/level, weighted by level: MAE on the ratio then equals MAE in demand units."""
    fitted = model()
    fitted.fit(frame[FEATURES], frame["target_demand"] / frame["level"], cat_features=["station_id"], sample_weight=frame["level"])
    return fitted


def predict_demand(fitted: CatBoostRegressor, frame: pd.DataFrame) -> np.ndarray:
    return fitted.predict(frame[FEATURES]) * frame["level"].to_numpy()


def train_models(data: pd.DataFrame, horizons: list[int]) -> tuple[dict[int, CatBoostRegressor], dict[str, Any]]:
    """Score on chronological validation/test windows, then refit on all history for production.

    The production model used to be the one fitted on the train split only, so it never saw
    the most recent 14 days -- exactly where a change in the demand pattern shows up first.
    Metrics (and the performance_drift thresholds derived from them) still come from the
    held-out windows; only the model that gets deployed is refit with everything.
    """
    models: dict[int, CatBoostRegressor] = {}
    report: dict[str, Any] = {}
    for horizon in horizons:
        supervised = add_target_calendar(data, horizon)
        train, validation, test = split_by_target(supervised)
        fitted = fit_model(train)
        validation_scores = validation[["station_id", "target_demand"]].rename(columns={"target_demand": "demand"})
        test_scores = test[["station_id", "target_demand"]].rename(columns={"target_demand": "demand"})
        validation_predictions = blend_predictions(predict_demand(fitted, validation), adjusted_naive_at_target(data, validation))
        test_predictions = blend_predictions(predict_demand(fitted, test), adjusted_naive_at_target(data, test))
        validation_metrics = station_metrics(validation_scores, validation_predictions)
        test_metrics = station_metrics(test_scores, test_predictions)
        models[horizon] = fit_model(supervised)
        report[str(horizon)] = {
            "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test), "production_rows": len(supervised),
            "validation": validation_metrics, "test": test_metrics,
        }
    return models, report


def prediction_rows(data: pd.DataFrame, cycle: dict, models: dict[int, CatBoostRegressor]) -> list[dict[str, Any]]:
    cutoff = pd.Timestamp(cycle["data_cutoff"])
    origins = data.loc[data["observed_at"] == cutoff].copy()
    if origins["station_id"].nunique() != 12:
        raise RuntimeError(f"Expected 12 station rows at cutoff {cutoff.isoformat()}, found {len(origins)}.")
    rows = []
    for horizon, targets in pd.DataFrame(cycle["targets"]).groupby("horizon_minutes"):
        horizon = int(horizon)
        if horizon not in models:
            raise RuntimeError(f"No trained model for requested horizon {horizon}.")
        # At inference the target demand is unknown; build only the target-time calendar.
        inference = origins.copy()
        inference["target_at"] = inference["observed_at"] + pd.to_timedelta(horizon, unit="m")
        local = inference["target_at"].dt.tz_convert("America/Bogota")
        quarter = local.dt.hour * 4 + local.dt.minute // 15
        inference["hour_sin"] = np.sin(2 * np.pi * quarter / 96)
        inference["hour_cos"] = np.cos(2 * np.pi * quarter / 96)
        inference["weekday_sin"] = np.sin(2 * np.pi * local.dt.dayofweek / 7)
        inference["weekday_cos"] = np.cos(2 * np.pi * local.dt.dayofweek / 7)
        inference["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
        values = blend_predictions(predict_demand(models[horizon], inference), adjusted_naive_at_target(data, inference))
        target_lookup = {(item.station_id, pd.Timestamp(item.target_at)): item for item in targets.itertuples()}
        for row, value in zip(inference.itertuples(), values, strict=True):
            target = target_lookup.get((row.station_id, row.target_at))
            if target is None:
                raise RuntimeError(f"No API target for station={row.station_id}, target_at={row.target_at}")
            rows.append({"station_id": row.station_id, "target_at": row.target_at.isoformat().replace("+00:00", "Z"), "value": float(value)})
    return sorted(rows, key=lambda row: (row["target_at"], row["station_id"]))


def main() -> None:
    data = add_origin_features(load_training_data())
    cycle: dict[str, Any] | None = None
    try:
        candidate = active_cycle()
        if candidate["state"] == "open":
            cycle = candidate
    except httpx.HTTPStatusError as error:
        if error.response.status_code != 404:
            raise
    horizons = sorted({int(target["horizon_minutes"]) for target in cycle["targets"]}) if cycle else [15, 30, 45, 60]
    models, metrics = train_models(data, horizons)
    data_cutoff = cycle["data_cutoff"] if cycle else data["observed_at"].max().isoformat()
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"models": models, "features": FEATURES, "horizons": horizons, "metrics": metrics, "ensemble_weight": ENSEMBLE_WEIGHT, "model_format": MODEL_FORMAT},
        ARTIFACTS_DIR / "catboost_direct.joblib",
    )
    (ARTIFACTS_DIR / "catboost_direct_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    record_lineage(data_cutoff, cycle["cycle_id"] if cycle else "no-active-cycle", metrics, models=models, data=data)
    print(f"Trained direct CatBoost models for horizons {horizons}.")
    for horizon in horizons:
        print(f"H={horizon} test accuracy: {metrics[str(horizon)]['test']['mean_station_accuracy']:.2f}")
    if cycle:
        predictions = prediction_rows(data, cycle, models)
        payload = {
            "schema_version": "1.0", "cycle_id": cycle["cycle_id"],
            "client_run_id": f"catboost-direct-preview-{uuid4().hex[:12]}", "data_cutoff": cycle["data_cutoff"],
            "model": {"version": "catboost-direct-v1", "training_data_end": cycle["data_cutoff"], "git_commit": git_commit()},
            "predictions": predictions,
        }
        errors = validate(payload, cycle)
        (ARTIFACTS_DIR / "submission_catboost_preview.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (ARTIFACTS_DIR / "submission_catboost_validation.json").write_text(json.dumps({"valid": not errors, "errors": errors, "prediction_count": len(predictions)}, indent=2), encoding="utf-8")
        if errors:
            raise SystemExit("Invalid local preview: " + "; ".join(errors))
        print(f"Validated local submission preview with {len(predictions)} predictions. No submission sent.")
    else:
        print("No active cycle: model and lineage were recorded; submission preview was intentionally skipped.")


if __name__ == "__main__":
    main()
