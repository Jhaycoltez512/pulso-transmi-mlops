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
  by their recent errors, so the predictor that has been right lately leads;
- periodic experts: demand P hours before the target (P in PERIOD_HOURS). From 2026-09-18
  05:00 UTC the injected drift is a 4h-periodic oscillation (four station groups, one hour
  apart, x5-11 peaks and x0.1-0.2 troughs vs the comparable day): copying the series from 4h
  earlier scores ~90% there and ~36% on a normal day, so the recent-error weights pick it only
  while such a regime lasts. Each periodic expert averages its period over the last
  PERIODIC_WINDOW (6 periods for 4h): averaging cancels the noise of copying a single period,
  ~91.3% -> ~93.0% per cycle on the live regime (backtest_periodic.py).
"""

from __future__ import annotations

import re

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

PERIOD = timedelta(minutes=15)
TIMEZONE = "America/Bogota"
LEVEL_PERIODS = 4  # last hour, for "how far above normal is this station right now"
BLEND_ALPHA = 0.25  # weight on the comparable day at h60; scales linearly with the horizon
CAP_THRESHOLD = 1.5  # current level / comparable-day level above which the growth cap applies
EXPERTS = ("model", "persistence", "comparable")
PERIOD_HOURS = (2, 3, 4, 5, 6)
PERIODIC_WINDOW = timedelta(hours=24)  # how far back each periodic expert averages its period
# Regime-break guard. At virtual 2026-09-20 12:15 the 4h oscillation stopped; weights learnt over
# the last 4h kept the periodic experts on top and the next two cycles scored 34% and 11% while
# persistence scored 75% and 69%. Copying any period P is checked directly on the observations:
# when even the best copy is badly wrong over the last hour (and far worse than over the day
# before), the periodic experts are dropped and the rest are weighted on that last hour only.
BREAK_RECENT = timedelta(hours=1)
BREAK_BASELINE = timedelta(hours=24)
BREAK_FLOOR = 0.25  # WAPE of the best period copy over the last hour
BREAK_RATIO = 2.0   # ... and that many times its WAPE over the previous day
LAG_EXPERTS = tuple(f"lag_{p}h" for p in PERIOD_HOURS)  # single-period copies (kept for backtests)
PERIODIC_EXPERTS = tuple(f"per_{p}h" for p in PERIOD_HOURS)  # period averaged over PERIODIC_WINDOW
# Longer periods. The regime that followed the 4h one (from virtual 2026-09-20 12:15) repeats every
# 8h: copying the value 8h before the target scored 85.4% on 1015 dense forecasts vs 79.6% for the
# trend expert, and our 2-6h experts could not see it (nor could the break guard, which kept
# firing). 7-12h also covers the next change the organisers may inject.
LONG_PERIOD_HOURS = (7, 8, 9, 10, 12)
ALL_PERIOD_HOURS = (*PERIOD_HOURS, *LONG_PERIOD_HOURS)
LONG_LAG_EXPERTS = tuple(f"lag_{p}h" for p in LONG_PERIOD_HOURS)
LONG_PERIODIC_EXPERTS = tuple(f"per_{p}h" for p in LONG_PERIOD_HOURS)
# Copy of the period, moved half-way to the current level: copy(target - P) + SHIFT_WEIGHT x
# (now - value P before now). 86.7% on the same 1015 forecasts (full shift 80.8%).
SHIFT_WEIGHT = 0.5
SHIFT_EXPERTS = tuple(f"shift_{p}h" for p in ALL_PERIOD_HOURS)
# Smoothed copy: mean of the values P before the target and 15 min either side. The 8h wave is
# smooth and each 15-min value is noisy, so the 3-point mean cancels part of the noise without
# moving the wave: 91.93% vs 90.14% for the plain 8h copy (and 90.43% submitted) on the first 250
# live 8h-regime forecasts; ahead of the plain copy on 11 of 12 cycles.
SMOOTH_EXPERTS = tuple(f"sm_{p}h" for p in ALL_PERIOD_HOURS)
# Harmonic fit: per station, least squares on the last max(P, HARMONIC_MIN_WINDOW) of data with a
# period-P Fourier basis of order HARMONIC_ORDER, extrapolated to the target. It smooths the noise
# over a whole period instead of 3 points: on the live 8h-regime cycles 92.5% for P=8h vs 91.9%
# for the smoothed copy and 89.9% submitted; on the 4h regime (P=4h, 8h window) 93.2% vs 91.9%.
HARMONIC_ORDER = 2
HARMONIC_MIN_WINDOW = timedelta(hours=8)
HARMONIC_EXPERTS = tuple(f"harm_{p}h" for p in ALL_PERIOD_HOURS)


def harmonic_fits(data: pd.DataFrame, frame: pd.DataFrame, hours_list=ALL_PERIOD_HOURS) -> dict[str, np.ndarray]:
    """{harm_Ph: prediction per frame row} from a Fourier fit at each row's origin."""
    table = data[["station_id", "observed_at", "demand"]].assign(observed_at=_utc_ns(data["observed_at"]).array)
    y = table.pivot_table(index="observed_at", columns="station_id", values="demand", aggfunc="last").sort_index()
    if y.empty:
        return {f"harm_{h}h": np.full(len(frame), np.nan) for h in hours_list}
    y = y.reindex(pd.date_range(y.index.min(), y.index.max(), freq=PERIOD)).ffill()
    step0 = y.index[0]
    origins = _utc_ns(frame["observed_at"]).array
    targets = _utc_ns(frame["target_at"]).array
    stations = np.asarray(frame["station_id"])
    col = {s: i for i, s in enumerate(y.columns)}
    values = y.to_numpy(float)
    out = {f"harm_{h}h": np.full(len(frame), np.nan) for h in hours_list}
    for origin in pd.unique(origins):
        rows = np.flatnonzero(origins == origin)
        end = int((origin - step0) / PERIOD)
        if end < 0 or end >= len(values):
            continue
        for hours in hours_list:
            period = hours * 4
            window = max(period, int(HARMONIC_MIN_WINDOW / PERIOD))
            if end - window + 1 < 0:
                continue
            seg = values[end - window + 1:end + 1]
            ok = np.isfinite(seg).all(axis=0)
            if not ok.any():
                continue
            def basis(x):
                cols = [np.ones_like(x, dtype=float)]
                for j in range(1, HARMONIC_ORDER + 1):
                    cols += [np.cos(2 * np.pi * j * x / period), np.sin(2 * np.pi * j * x / period)]
                return np.column_stack(cols)
            beta = np.full((2 * HARMONIC_ORDER + 1, seg.shape[1]), np.nan)
            beta[:, ok], *_ = np.linalg.lstsq(basis(np.arange(end - window + 1, end + 1)), seg[:, ok], rcond=None)
            steps = ((targets[rows] - step0) / PERIOD).astype(float)
            idx = np.array([col.get(st, -1) for st in stations[rows]])
            pred = np.einsum("rk,kr->r", basis(steps), beta[:, np.clip(idx, 0, None)])
            pred[idx < 0] = np.nan
            out[f"harm_{hours}h"][rows] = np.clip(pred, 0, None)
    return out
# Damped-trend persistence: last value + damping x (last-hour slope) x horizon. After the 4h
# oscillation ended (virtual 2026-09-20 12:15) each station drifts smoothly for hours; on 814
# dense post-break forecasts damping 0.5 scored 78.0% vs 76.1% for plain persistence (full
# slope: 74.2%). Not periodic, so the regime-break guard keeps it.
TREND_DAMPING = {"trend": 0.5, "trend_03": 0.3, "trend_full": 1.0}  # only "trend" runs in production
TREND_EXPERTS = ("trend",)
TREND_SLOPE_WINDOW = timedelta(hours=1)


def reference_lag_days(times: pd.Series) -> np.ndarray:
    """Days back to the previous comparable day, by local weekday."""
    weekday = pd.DatetimeIndex(times).tz_convert(TIMEZONE).dayofweek.to_numpy()
    return np.select([weekday == 0, weekday >= 5], [3, 7], default=1)


def _utc_ns(values) -> pd.Series:
    """Timestamps as datetime64[ns, UTC]: merges refuse keys of different resolution ([s] vs [ns])."""
    return pd.Series(pd.to_datetime(np.asarray(values), utc=True)).astype("datetime64[ns, UTC]")


def _lookup(table: pd.DataFrame, value: str, stations: pd.Series, times: pd.Series) -> np.ndarray:
    keys = pd.DataFrame({"station_id": np.asarray(stations), "observed_at": _utc_ns(times)})
    right = table[["station_id", "observed_at", value]].assign(observed_at=_utc_ns(table["observed_at"]).array)
    return keys.merge(right, on=["station_id", "observed_at"], how="left")[value].to_numpy(dtype=float)


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
    target = pd.to_datetime(out["target_at"], utc=True)
    origin = pd.to_datetime(out["observed_at"], utc=True)
    hour_ago = _lookup(table, "demand", out["station_id"], origin - TREND_SLOPE_WINDOW)
    slope = (out["persistence"].to_numpy() - hour_ago) / (TREND_SLOPE_WINDOW / PERIOD)
    steps = ((target - origin) / PERIOD).to_numpy(dtype=float)
    for name, damping in TREND_DAMPING.items():
        # no slope known (gap an hour ago) -> plain persistence
        out[name] = np.clip(out["persistence"].to_numpy() + np.nan_to_num(damping * slope * steps), 0, None)
    for hours in ALL_PERIOD_HOURS:
        lag_name, mean_name = f"lag_{hours}h", f"per_{hours}h"
        copies = [
            _lookup(table, "demand", out["station_id"], target - timedelta(hours=hours * k))
            for k in range(1, int(PERIODIC_WINDOW / timedelta(hours=hours)) + 1)
        ]
        out[lag_name] = copies[0]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows -> NaN, as intended
            out[mean_name] = np.nanmean(np.vstack(copies), axis=0)
        neighbours = [_lookup(table, "demand", out["station_id"], target - timedelta(hours=hours) + step * PERIOD) for step in (-1, 1)]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[f"sm_{hours}h"] = np.nanmean(np.vstack([copies[0], *neighbours]), axis=0)
        then = _lookup(table, "demand", out["station_id"], origin - timedelta(hours=hours))
        out[f"shift_{hours}h"] = np.clip(copies[0] + SHIFT_WEIGHT * (out["persistence"].to_numpy() - then), 0, None)
    for name, values in harmonic_fits(data, out).items():
        out[name] = values
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


def expert_weights(
    history: pd.DataFrame, queries: pd.DataFrame, keys: list[str], window: timedelta, power: float,
    experts: tuple[str, ...] = EXPERTS,
) -> pd.DataFrame:
    """Per-query expert weights from errors on targets already known at the query's origin.

    history: rows with keys, target_at, actual and one column per expert (its prediction).
    queries: rows with keys and observed_at (origin). Only history targets in
    (origin - window, origin] count. Weight_i = MAE_i^-power / sum; experts with no history get 0;
    if nothing is known yet, the model gets all the weight.
    """
    hist = history.copy()
    hist["target_at"] = _utc_ns(hist["target_at"]).array
    hist = hist.sort_values("target_at")
    for expert in experts:
        hist[f"err_{expert}"] = (hist[expert] - hist["actual"]).abs()
        hist[f"n_{expert}"] = hist[f"err_{expert}"].notna().astype(float)
        hist[f"err_{expert}"] = hist[f"err_{expert}"].fillna(0.0)
    cum_cols = [f"{p}_{e}" for e in experts for p in ("err", "n")]
    hist[cum_cols] = hist[cum_cols].astype(float)
    hist[cum_cols] = hist.groupby(keys, sort=False)[cum_cols].cumsum() if keys else hist[cum_cols].cumsum()
    hist = hist[[*keys, "target_at", *cum_cols]]

    q = queries[[*keys, "observed_at"]].copy()
    q["_row"] = np.arange(len(q))
    q["_end"] = _utc_ns(q["observed_at"]).array
    q["_start"] = q["_end"] - window

    def as_of(column: str) -> pd.DataFrame:
        left = q.sort_values(column)
        merged = pd.merge_asof(left, hist, left_on=column, right_on="target_at", by=keys or None, direction="backward")
        return merged.set_index("_row")[cum_cols].reindex(np.arange(len(q))).fillna(0.0)

    end, start = as_of("_end"), as_of("_start")
    window_sums = end - start
    inv = {}
    for expert in experts:
        count = window_sums[f"n_{expert}"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            mae = window_sums[f"err_{expert}"].to_numpy() / count
            inv[expert] = np.where(count > 0, np.power(np.maximum(mae, 1e-6), -power), 0.0)
    total = sum(inv.values())
    weights = pd.DataFrame({e: np.where(total > 0, inv[e] / np.where(total > 0, total, 1), 1.0 if e == "model" else 0.0) for e in experts})
    return weights


def combine(predictions: pd.DataFrame, weights: pd.DataFrame) -> np.ndarray:
    """Weighted mean of the experts, renormalised over experts that have a prediction for the row."""
    experts = list(weights.columns)
    values = predictions[experts].to_numpy(dtype=float)
    w = weights[experts].to_numpy(dtype=float) * ~np.isnan(values)
    norm = w.sum(axis=1)
    combined = np.nansum(np.nan_to_num(values) * w, axis=1) / np.where(norm > 0, norm, 1)
    return np.where(norm > 0, combined, predictions["model"].to_numpy(dtype=float))


def expert_frame(frame: pd.DataFrame, model: np.ndarray, experts: tuple[str, ...]) -> pd.DataFrame:
    """One column per expert from a reference_inputs() frame; "comparable" is its "reference"."""
    columns = {"model": np.asarray(model, dtype=float)}
    for expert in experts:
        if expert != "model":
            columns[expert] = frame["reference" if expert == "comparable" else expert].to_numpy(dtype=float)
    return pd.DataFrame(columns)[list(experts)]


def periodic_break(data: pd.DataFrame, cutoff: pd.Timestamp, hours: tuple[int, ...] = PERIOD_HOURS) -> tuple[bool, dict[str, float]]:
    """Whether the periodic pattern broke in the last hour, from the observations up to cutoff.

    hours: the periods the ensemble can copy -- a break only matters if none of them still works."""
    table = data.assign(observed_at=_utc_ns(data["observed_at"]).array)
    y = table.pivot_table(index="observed_at", columns="station_id", values="demand", aggfunc="last").sort_index()
    end = pd.Timestamp(cutoff).tz_convert("UTC") if pd.Timestamp(cutoff).tzinfo else pd.Timestamp(cutoff).tz_localize("UTC")
    y = y.loc[(y.index > end - BREAK_BASELINE - BREAK_RECENT - timedelta(hours=max(hours))) & (y.index <= end)]

    def wape(hours: int, start: pd.Timestamp, stop: pd.Timestamp) -> float:
        now = y.loc[(y.index > start) & (y.index <= stop)]
        then = y.reindex(now.index - timedelta(hours=hours))
        then.index = now.index
        mask = now.notna() & then.notna()
        total = now.where(mask).sum().sum()
        return float((now - then).abs().where(mask).sum().sum() / total) if total > 0 else float("nan")

    recent = min((wape(h, end - BREAK_RECENT, end) for h in hours), default=float("nan"))
    baseline = min((wape(h, end - BREAK_RECENT - BREAK_BASELINE, end - BREAK_RECENT) for h in hours), default=float("nan"))
    broke = bool(np.isfinite(recent) and np.isfinite(baseline) and recent > BREAK_FLOOR and recent > BREAK_RATIO * baseline)
    return broke, {"recent_wape": round(recent, 4), "baseline_wape": round(baseline, 4)}


def production_adjust(
    data: pd.DataFrame, predictions: pd.DataFrame, history: pd.DataFrame, mode: str,
    window: timedelta, power: float, ensemble_on_adjusted: bool = True,
    threshold: float = CAP_THRESHOLD, pool_stations: bool = False, experts: tuple[str, ...] = EXPERTS,
    break_guard: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Final values for one cycle.

    predictions: station_id, observed_at (cycle cutoff), target_at, horizon_minutes, model.
    history: already evaluated predictions -- station_id, horizon_minutes, target_at, model (the
    raw model output at the time) and actual. Returns the values to submit and, for logging, the
    mean expert weights ({} when not ensembling). pool_stations: one set of weights per horizon
    across stations (more history per weight) instead of one per station and horizon. experts:
    which predictors compete ("model" must be one of them; see EXPERTS and PERIODIC_EXPERTS).
    break_guard: when periodic_break fires, only the non-periodic experts compete, weighted over
    BREAK_RECENT (the "regime_break" key of the returned dict says whether it fired).
    """
    frame = reference_inputs(data, predictions)
    model = frame["model"].to_numpy(dtype=float)
    horizons = frame["horizon_minutes"].to_numpy()
    if mode == "none":
        return np.clip(model, 0, None), {}
    adjusted = adjust(frame, model, horizons, threshold=threshold)
    if mode == "blend_cap":
        return adjusted, {}
    if mode != "ensemble":
        raise ValueError(f"unknown adjustment mode {mode!r}")
    broke = False
    periodic = tuple(e for e in experts if re.fullmatch(r"(lag|per|shift|sm|harm)_\d+h", e))
    if break_guard and periodic:
        hours = tuple(sorted({int(re.search(r"\d+", e).group()) for e in periodic}))
        broke, _ = periodic_break(data, frame["observed_at"].max(), hours)
        if broke:
            experts = tuple(e for e in experts if e not in periodic)
            window = min(window, BREAK_RECENT)
    past = history.copy()
    past["target_at"] = pd.to_datetime(past["target_at"], utc=True)
    past["observed_at"] = past["target_at"] - pd.to_timedelta(past["horizon_minutes"], unit="m")
    past = reference_inputs(data, past)
    past_model = past["model"].to_numpy(dtype=float)
    experts_past = expert_frame(past, adjust(past, past_model, past["horizon_minutes"].to_numpy(), threshold=threshold) if ensemble_on_adjusted else past_model, experts)
    keys = ["horizon_minutes"] if pool_stations else ["station_id", "horizon_minutes"]
    hist = pd.concat([past[["station_id", "horizon_minutes", "target_at"]].reset_index(drop=True), experts_past], axis=1)
    hist["actual"] = past["actual"].to_numpy(dtype=float)
    weights = expert_weights(hist, frame[[*keys, "observed_at"]].reset_index(drop=True), keys, window, power, experts)
    experts_now = expert_frame(frame, adjusted if ensemble_on_adjusted else model, experts)
    summary = weights.mean().round(3).to_dict()
    if break_guard:
        summary["regime_break"] = float(broke)
    return np.clip(combine(experts_now, weights), 0, None), summary
