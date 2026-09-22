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
NUMERIC_FEATURES = [
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_weekend", "current_demand",
    *[f"lag_{lag}" for lag in LAGS], *[f"rolling_mean_{window}" for window in ROLLING_WINDOWS],
]
FEATURES = ["station_id", *NUMERIC_FEATURES]
WEEKLY_NAIVE_LAG = timedelta(days=7)
# Share given to the CatBoost prediction vs. the weekly seasonal-naive prediction. Picked with
# scripts/backtest_ensemble.py: a walk-forward sweep found 0.70-0.85 near-optimal at every horizon,
# with WAPE gains over pure CatBoost well above the noise between folds (see docs/ml-baselines.md).
ENSEMBLE_WEIGHT = 0.75


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def performance_thresholds(metrics: dict[str, Any], factor: float = 1.15) -> dict[str, float]:
    """Recent-WAPE ceiling per horizon: this model's own validation WAPE plus a relative margin."""
    return {horizon: result["validation"]["mean_station_wape"] * factor for horizon, result in metrics.items()}


def record_lineage(
    data_cutoff: str, cycle_id: str, metrics: dict[str, Any], trigger_reason: str = "direct-multihorizon-training",
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
        ingestions = loader.select("ingestion_runs", {"status": "eq.succeeded", "order": "finished_at.desc", "limit": 1})
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
        try:
            mlflow_id = log_run(
                run_name=version, tags={"model": "catboost-direct", "data_version": data_version or "unknown", "git_commit": git_commit() or "unknown"},
                params={"horizons": sorted(map(int, metrics)), "feature_count": len(FEATURES)}, metrics=flat_metrics,
                artifacts={"metrics.json": metrics, "data_lineage.json": {"data_version": data_version, "cycle_id": cycle_id, "data_cutoff": data_cutoff}},
                artifact_paths=[ARTIFACTS_DIR / "catboost_direct.joblib", ARTIFACTS_DIR / "catboost_direct_metrics.json"],
            )
            if mlflow_id:
                loader.patch("model_versions", model_version["id"], {"mlflow_run_id": mlflow_id})
                loader.patch("training_runs", training_run["id"], {"mlflow_run_id": mlflow_id})
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


def train_models(data: pd.DataFrame, horizons: list[int]) -> tuple[dict[int, CatBoostRegressor], dict[str, Any]]:
    models: dict[int, CatBoostRegressor] = {}
    report: dict[str, Any] = {}
    for horizon in horizons:
        supervised = add_target_calendar(data, horizon)
        train, validation, test = split_by_target(supervised)
        fitted = model()
        fitted.fit(train[FEATURES], train["target_demand"], cat_features=["station_id"])
        validation_scores = validation[["station_id", "target_demand"]].rename(columns={"target_demand": "demand"})
        test_scores = test[["station_id", "target_demand"]].rename(columns={"target_demand": "demand"})
        validation_predictions = blend_predictions(fitted.predict(validation[FEATURES]), naive_prediction_at_target(data, validation))
        test_predictions = blend_predictions(fitted.predict(test[FEATURES]), naive_prediction_at_target(data, test))
        validation_metrics = station_metrics(validation_scores, validation_predictions)
        test_metrics = station_metrics(test_scores, test_predictions)
        models[horizon] = fitted
        report[str(horizon)] = {
            "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
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
        values = blend_predictions(models[horizon].predict(inference[FEATURES]), naive_prediction_at_target(data, inference))
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
        {"models": models, "features": FEATURES, "horizons": horizons, "metrics": metrics, "ensemble_weight": ENSEMBLE_WEIGHT},
        ARTIFACTS_DIR / "catboost_direct.joblib",
    )
    (ARTIFACTS_DIR / "catboost_direct_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    record_lineage(data_cutoff, cycle["cycle_id"] if cycle else "no-active-cycle", metrics)
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
