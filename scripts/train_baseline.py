"""Train and evaluate a demand-forecasting baseline from Supabase data.

Usage:
    python -m pip install -e '.[ml]'
    python scripts/train_baseline.py

The script expects SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_KEY) in
the environment or in .env. It writes a joblib artifact and evaluation metrics
to artifacts/ (which is intentionally ignored by Git).
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ARTIFACTS_DIR = Path("artifacts")
TIMEZONE = "America/Bogota"
NUMERIC_FEATURES = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "lag_96",
    "lag_672",
    "rain_forecast",
    "temperature_forecast",
    "event_intensity",
]
CATEGORICAL_FEATURES = ["station_id"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple local environment variables without an extra dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_rest_url() -> tuple[str, str]:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Configure SUPABASE_URL and SUPABASE_SECRET_KEY in .env.")
    base_url = url.rstrip("/")
    if not base_url.endswith("/rest/v1"):
        base_url = f"{base_url}/rest/v1"
    return base_url, key


def fetch_all(client: httpx.Client, table: str, columns: str, order: str) -> list[dict[str, Any]]:
    """Read all rows through PostgREST's bounded pagination."""
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 1_000
    while True:
        response = client.get(
            f"/{table}",
            params={"select": columns, "order": order, "limit": page_size, "offset": offset},
        )
        response.raise_for_status()
        page = response.json()
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def load_training_data() -> pd.DataFrame:
    base_url, key = get_rest_url()
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=60) as client:
        observations = pd.DataFrame(
            fetch_all(client, "observations", "station_id,observed_at,demand", "observed_at.asc,station_id.asc")
        )
        context = pd.DataFrame(
            fetch_all(
                client,
                "context",
                "observed_at,rain_forecast,temperature_forecast,event_intensity",
                "observed_at.asc",
            )
        )
    if observations.empty or context.empty:
        raise RuntimeError("Supabase returned no observations or context data.")
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)
    observations["station_id"] = observations["station_id"].astype("string")
    return observations.merge(context, on="observed_at", how="left", validate="many_to_one")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only features known at the timestamp being forecast."""
    data = frame.sort_values(["station_id", "observed_at"]).copy()
    local_time = data["observed_at"].dt.tz_convert(TIMEZONE)
    data["hour_sin"] = np.sin(2 * np.pi * (local_time.dt.hour * 4 + local_time.dt.minute // 15) / 96)
    data["hour_cos"] = np.cos(2 * np.pi * (local_time.dt.hour * 4 + local_time.dt.minute // 15) / 96)
    data["weekday_sin"] = np.sin(2 * np.pi * local_time.dt.dayofweek / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * local_time.dt.dayofweek / 7)
    data["is_weekend"] = (local_time.dt.dayofweek >= 5).astype(int)
    data["lag_96"] = data.groupby("station_id")["demand"].shift(96)
    data["lag_672"] = data.groupby("station_id")["demand"].shift(672)
    required = FEATURES + ["demand", "observed_at"]
    return data.dropna(subset=required).reset_index(drop=True)


def temporal_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use chronological windows: training, 7-day validation, then 7-day test."""
    max_timestamp = frame["observed_at"].max()
    test_start = max_timestamp - timedelta(days=7)
    validation_start = test_start - timedelta(days=7)
    train = frame.loc[frame["observed_at"] <= validation_start].copy()
    validation = frame.loc[(frame["observed_at"] > validation_start) & (frame["observed_at"] <= test_start)].copy()
    test = frame.loc[frame["observed_at"] > test_start].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("Insufficient history for the requested train/validation/test split.")
    return train, validation, test


def station_metrics(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, Any]:
    scored = frame[["station_id", "demand"]].copy()
    scored["prediction"] = np.clip(predictions, 0, None)
    aggregate = scored.groupby("station_id").apply(
        lambda group: pd.Series({
            "wape": float((group.demand - group.prediction).abs().sum() / group.demand.sum()),
            "accuracy": float(100 * max(0, 1 - (group.demand - group.prediction).abs().sum() / group.demand.sum())),
        }),
        include_groups=False,
    )
    return {
        "mean_station_accuracy": float(aggregate["accuracy"].mean()),
        "mean_station_wape": float(aggregate["wape"].mean()),
        "by_station": aggregate.round(6).to_dict(orient="index"),
    }


def make_model() -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("station", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("features", preprocessing),
        ("model", HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=42)),
    ])


def main() -> None:
    data = build_features(load_training_data())
    train, validation, test = temporal_splits(data)
    model = make_model()
    model.fit(train[FEATURES], train["demand"])
    validation_metrics = station_metrics(validation, model.predict(validation[FEATURES]))
    test_metrics = station_metrics(test, model.predict(test[FEATURES]))

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    metadata = {
        "model": "HistGradientBoostingRegressor",
        "features": FEATURES,
        "timezone": TIMEZONE,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_end": train["observed_at"].max().isoformat(),
        "validation_end": validation["observed_at"].max().isoformat(),
        "test_end": test["observed_at"].max().isoformat(),
    }
    joblib.dump({"model": model, "metadata": metadata}, ARTIFACTS_DIR / "demand_baseline.joblib")
    (ARTIFACTS_DIR / "demand_baseline_metrics.json").write_text(
        json.dumps({"metadata": metadata, "validation": validation_metrics, "test": test_metrics}, indent=2), encoding="utf-8"
    )
    print(f"Validation accuracy: {validation_metrics['mean_station_accuracy']:.2f}")
    print(f"Test accuracy: {test_metrics['mean_station_accuracy']:.2f}")
    print(f"Artifact: {ARTIFACTS_DIR / 'demand_baseline.joblib'}")


if __name__ == "__main__":
    main()
