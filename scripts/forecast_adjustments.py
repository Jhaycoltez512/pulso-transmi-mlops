"""Post-model adjustments: comparable-day reference, mild blend, growth cap, adaptive experts.

The level-normalised CatBoost (train_catboost_direct.py) follows each station's recent level,
which handles sustained level changes but fails in two ways under the competition's injected
events (2026-09-18 05:00-10:00 UTC: the whole system x2.6-3.4, station groups pulsing x3-11):

- once a station is already inflated by an event, the model applies the usual ramp towards the
  morning peak on top of it (07111: level 3759, predicted 4690 -> 6718, actual 2775 -> 1751);
- no single predictor wins everywhere: the comparable day is ~90% in normal hours and ~40% in
  event hours, persistence is the reverse, and the model wins under sustained level changes.

This module holds the pieces that sit after the model:

- comparable-day reference: the most recent day of the same type (Tue-Fri: the day before,
  Monday: Friday, weekends: same day last week), so a level change that has lasted a day is the
  new normal while weekday/weekend profiles aren't mixed;
- mild blend towards that reference, heavier for longer horizons;
- growth cap: while a station is far above its comparable day, don't predict it climbing above
  max(current level, comparable day at the target);
- adaptive expert ensemble: weight model / persistence / comparable day per station and horizon
  by their recent errors, so the predictor that has been right lately leads.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

PERIOD = timedelta(minutes=15)
TIMEZONE = "America/Bogota"
LEVEL_PERIODS = 4  # last hour, for "how far above normal is this station right now"
BLEND_ALPHA = 0.25  # weight on the comparable day at h60; scales linearly with the horizon
CAP_THRESHOLD = 1.5  # current level / comparable-day level above which the growth cap applies
EXPERTS = ("model", "persistence", "comparable")


def reference_lag_days(times: pd.Series) -> np.ndarray:
    """Days back to the previous comparable day, by local weekday."""
    weekday = pd.DatetimeIndex(times).tz_convert(TIMEZONE).dayofweek.to_numpy()
    return np.select([weekday == 0, weekday >= 5], [3, 7], default=1)


def _lookup(table: pd.DataFrame, value: str, stations: pd.Series, times: pd.Series) -> np.ndarray:
    keys = pd.DataFrame({"station_id": np.asarray(stations), "observed_at": np.asarray(times)})
    keys["observed_at"] = pd.to_datetime(keys["observed_at"], utc=True)
    return keys.merge(table[["station_id", "observed_at", value]], on=["station_id", "observed_at"], how="left")[value].to_numpy(dtype=float)


def level_table(data: pd.DataFrame) -> pd.DataFrame:
    """demand and last-hour mean level per station and timestamp."""
    table = data[["station_id", "observed_at", "demand"]].sort_values(["station_id", "observed_at"]).copy()
    table["level_now"] = table.groupby("station_id")["demand"].transform(lambda s: s.rolling(LEVEL_PERIODS, min_periods=LEVEL_PERIODS).mean())
    return table


def reference_inputs(data: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Add comparable-day demand at the target, current level and comparable-day level at the origin.

    `frame` needs station_id, observed_at (the forecast origin) and target_at.
    """
    table = level_table(data)
    out = frame.copy()
    target_ref = pd.to_datetime(out["target_at"], utc=True) - pd.to_timedelta(reference_lag_days(out["target_at"]), unit="D")
    origin_ref = pd.to_datetime(out["observed_at"], utc=True) - pd.to_timedelta(reference_lag_days(out["observed_at"]), unit="D")
    out["reference"] = _lookup(table, "demand", out["station_id"], target_ref)
    out["level_now"] = _lookup(table, "level_now", out["station_id"], out["observed_at"])
    out["reference_level"] = _lookup(table, "level_now", out["station_id"], origin_ref)
    out["persistence"] = _lookup(table, "demand", out["station_id"], out["observed_at"])
    return out


def blend_with_reference(prediction: np.ndarray, reference: np.ndarray, horizon_minutes: np.ndarray | int, alpha: float = BLEND_ALPHA) -> np.ndarray:
    """(1 - w) * model + w * comparable day, w = alpha * horizon / 60; model alone where no reference."""
    weight = alpha * np.asarray(horizon_minutes, dtype=float) / 60
    return np.where(np.isnan(reference), prediction, (1 - weight) * prediction + weight * reference)


def cap_growth(prediction: np.ndarray, level_now: np.ndarray, reference_level: np.ndarray, reference: np.ndarray, threshold: float = CAP_THRESHOLD) -> np.ndarray:
    """While a station runs > threshold x its comparable day, cap at max(current level, comparable day at target)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inflated = (level_now / reference_level) > threshold
    ceiling = np.fmax(level_now, reference)
    return np.where(inflated & ~np.isnan(ceiling), np.minimum(prediction, ceiling), prediction)


def adjust(
    frame: pd.DataFrame, prediction: np.ndarray, horizon_minutes: np.ndarray | int,
    blend: bool = True, cap: bool = True, threshold: float = CAP_THRESHOLD,
) -> np.ndarray:
    """Production adjustment on top of the model: optional growth cap, then the mild blend."""
    out = np.asarray(prediction, dtype=float)
    if cap:
        out = cap_growth(out, frame["level_now"].to_numpy(), frame["reference_level"].to_numpy(), frame["reference"].to_numpy(), threshold)
    if blend:
        out = blend_with_reference(out, frame["reference"].to_numpy(), horizon_minutes)
    return np.clip(out, 0, None)


def expert_weights(history: pd.DataFrame, queries: pd.DataFrame, keys: list[str], window: timedelta, power: float) -> pd.DataFrame:
    """Per-query expert weights from errors on targets already known at the query's origin.

    history: rows with keys, target_at, actual and one column per expert (its prediction).
    queries: rows with keys and observed_at (origin). Only history targets in
    (origin - window, origin] count. Weight_i = MAE_i^-power / sum; experts with no history get 0;
    if nothing is known yet, the model gets all the weight.
    """
    hist = history.sort_values("target_at").copy()
    for expert in EXPERTS:
        hist[f"err_{expert}"] = (hist[expert] - hist["actual"]).abs()
        hist[f"n_{expert}"] = hist[f"err_{expert}"].notna().astype(float)
        hist[f"err_{expert}"] = hist[f"err_{expert}"].fillna(0.0)
    cum_cols = [f"{p}_{e}" for e in EXPERTS for p in ("err", "n")]
    hist[cum_cols] = hist[cum_cols].astype(float)
    hist[cum_cols] = hist.groupby(keys, sort=False)[cum_cols].cumsum() if keys else hist[cum_cols].cumsum()
    hist = hist[[*keys, "target_at", *cum_cols]]

    q = queries[[*keys, "observed_at"]].copy()
    q["_row"] = np.arange(len(q))
    q["_end"] = pd.to_datetime(q["observed_at"], utc=True)
    q["_start"] = q["_end"] - window

    def as_of(column: str) -> pd.DataFrame:
        left = q.sort_values(column)
        merged = pd.merge_asof(left, hist, left_on=column, right_on="target_at", by=keys or None, direction="backward")
        return merged.set_index("_row")[cum_cols].reindex(np.arange(len(q))).fillna(0.0)

    end, start = as_of("_end"), as_of("_start")
    window_sums = end - start
    inv = {}
    for expert in EXPERTS:
        count = window_sums[f"n_{expert}"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            mae = window_sums[f"err_{expert}"].to_numpy() / count
            inv[expert] = np.where(count > 0, np.power(np.maximum(mae, 1e-6), -power), 0.0)
    total = sum(inv.values())
    weights = pd.DataFrame({e: np.where(total > 0, inv[e] / np.where(total > 0, total, 1), 1.0 if e == "model" else 0.0) for e in EXPERTS})
    return weights


def combine(predictions: pd.DataFrame, weights: pd.DataFrame) -> np.ndarray:
    """Weighted mean of the experts, renormalised over experts that have a prediction for the row."""
    values = predictions[list(EXPERTS)].to_numpy(dtype=float)
    w = weights[list(EXPERTS)].to_numpy(dtype=float) * ~np.isnan(values)
    norm = w.sum(axis=1)
    combined = np.nansum(np.nan_to_num(values) * w, axis=1) / np.where(norm > 0, norm, 1)
    return np.where(norm > 0, combined, predictions["model"].to_numpy(dtype=float))


def production_adjust(
    data: pd.DataFrame, predictions: pd.DataFrame, history: pd.DataFrame, mode: str,
    window: timedelta, power: float, ensemble_on_adjusted: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Final values for one cycle.

    predictions: station_id, observed_at (cycle cutoff), target_at, horizon_minutes, model.
    history: already evaluated predictions -- station_id, horizon_minutes, target_at, model (the
    raw model output at the time) and actual. Returns the values to submit and, for logging, the
    mean expert weights ({} when not ensembling).
    """
    frame = reference_inputs(data, predictions)
    model = frame["model"].to_numpy(dtype=float)
    horizons = frame["horizon_minutes"].to_numpy()
    if mode == "none":
        return np.clip(model, 0, None), {}
    adjusted = adjust(frame, model, horizons)
    if mode == "blend_cap":
        return adjusted, {}
    if mode != "ensemble":
        raise ValueError(f"unknown adjustment mode {mode!r}")
    past = history.copy()
    past["target_at"] = pd.to_datetime(past["target_at"], utc=True)
    past["observed_at"] = past["target_at"] - pd.to_timedelta(past["horizon_minutes"], unit="m")
    past = reference_inputs(data, past)
    past_model = past["model"].to_numpy(dtype=float)
    experts_past = pd.DataFrame({
        "model": adjust(past, past_model, past["horizon_minutes"].to_numpy()) if ensemble_on_adjusted else past_model,
        "persistence": past["persistence"].to_numpy(), "comparable": past["reference"].to_numpy(),
    })
    keys = ["station_id", "horizon_minutes"]
    hist = pd.concat([past[[*keys, "target_at"]].reset_index(drop=True), experts_past], axis=1)
    hist["actual"] = past["actual"].to_numpy(dtype=float)
    weights = expert_weights(hist, frame[[*keys, "observed_at"]].reset_index(drop=True), keys, window, power)
    experts_now = pd.DataFrame({
        "model": adjusted if ensemble_on_adjusted else model,
        "persistence": frame["persistence"].to_numpy(), "comparable": frame["reference"].to_numpy(),
    })
    return np.clip(combine(experts_now, weights), 0, None), weights.mean().round(3).to_dict()
