"""Read-only experiment: candidate forecasters on the 8h regime (from virtual 2026-09-20 12:15).

Every candidate predicts y[t + k] (k = 1..4 steps of 15 min, the cycle horizons) from data up to
the origin t only. Scored like the leaderboard: 1 - WAPE per hourly origin (a "cycle", 48
predictions), then averaged; also on every 15-min origin. Missing station-periods are filled the
way production fills them (last known value).

    python scripts/experiment_8h.py
"""

from __future__ import annotations

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from load_supabase import load_dotenv
from run_forecast_cycle import fill_gaps
from train_baseline import load_training_data

warnings.simplefilter("ignore", category=RuntimeWarning)
STEP = pd.Timedelta(minutes=15)
REGIME = pd.Timestamp("2026-09-20T12:15:00Z")
P8 = 32  # 8h in steps
EVAL_FROM = pd.Timestamp("2026-09-21T00:00:00Z")  # 11h45 of regime: one full period + margin


def series() -> tuple[np.ndarray, pd.DatetimeIndex, list[str], np.ndarray]:
    raw = load_training_data()[["station_id", "observed_at", "demand"]].copy()
    raw["observed_at"] = pd.to_datetime(raw["observed_at"], utc=True)
    truth = raw.pivot_table(index="observed_at", columns="station_id", values="demand", aggfunc="last")
    filled = fill_gaps(raw).pivot_table(index="observed_at", columns="station_id", values="demand", aggfunc="last")
    filled = filled.loc[filled.index >= REGIME - pd.Timedelta(hours=30)]
    truth = truth.reindex(filled.index)
    return filled.to_numpy(float), filled.index, list(filled.columns), truth.to_numpy(float)


def sm(y: np.ndarray, i: int, w: int) -> np.ndarray:
    return np.nanmean(y[i - w:i + w + 1], axis=0)


def harmonic(y: np.ndarray, t: int, k: int, hours: int, m: int, trend: bool) -> np.ndarray:
    """Least squares per station on the last `hours` with an 8h Fourier basis of order m."""
    n = hours * 4
    s = np.arange(t - n + 1, t + 1)
    cols = [np.ones(n)]
    for j in range(1, m + 1):
        cols += [np.cos(2 * np.pi * j * s / P8), np.sin(2 * np.pi * j * s / P8)]
    if trend:
        cols.append((s - t) / n)
    X = np.column_stack(cols)
    tt = t + k
    xf = [1.0]
    for j in range(1, m + 1):
        xf += [np.cos(2 * np.pi * j * tt / P8), np.sin(2 * np.pi * j * tt / P8)]
    if trend:
        xf.append(k / n)
    Y = y[t - n + 1:t + 1]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return np.asarray(xf) @ beta


def candidates(y: np.ndarray, t: int, k: int) -> dict[str, np.ndarray]:
    last = y[t]
    out = {"persistencia": last, "lag8": y[t + k - P8]}
    for w in (1, 2, 3):
        out[f"sm8_w{w}"] = sm(y, t + k - P8, w)
    # residual correction: the copy's error at the origin persists for a while
    for w in (1, 2):
        base_t, base_k = sm(y, t - P8, w), sm(y, t + k - P8, w)
        resid = last - base_t
        resid4 = np.mean([y[t - j] - sm(y, t - j - P8, w) for j in range(4)], axis=0)
        for a in (0.3, 0.5, 0.7, 1.0):
            out[f"sm8_w{w}+res{a}"] = base_k + a * resid
            out[f"sm8_w{w}+res{a}_d"] = base_k + a * (0.8 ** (k - 1)) * resid
            out[f"sm8_w{w}+res4_{a}"] = base_k + a * resid4
        # amplitude/level ratio over the last hour
        lvl_now, lvl_then = np.mean(y[t - 3:t + 1], axis=0), np.mean(y[t - 3 - P8:t + 1 - P8], axis=0)
        ratio = np.clip((lvl_now + 20) / (lvl_then + 20), 0.5, 2.0)
        for a in (0.3, 0.5):
            out[f"sm8_w{w}*ratio{a}"] = base_k * (1 + a * (ratio - 1))
    # two periods back (amplitude may drift)
    if t + k - 2 * P8 - 2 >= 0:
        s1, s2 = sm(y, t + k - P8, 1), sm(y, t + k - 2 * P8, 1)
        out["sm8+sm16 (0.75/0.25)"] = 0.75 * s1 + 0.25 * s2
        out["sm8+sm16 (2s1-s2) amortiguado"] = s1 + 0.3 * (s1 - s2)
    for hours in (8, 12, 16):
        for m in (1, 2, 3):
            for tr in (False, True):
                try:
                    out[f"armonico {hours}h m{m}{' +tend' if tr else ''}"] = harmonic(y, t, k, hours, m, tr)
                except Exception:
                    pass
    return {name: np.clip(v, 0, None) for name, v in out.items()}


def ridge_preds(y: np.ndarray, origins: list[int], lam: float = 1e3) -> dict[tuple[int, int], np.ndarray]:
    """Per-horizon ridge on [y_t, y_t-1, y_t-2, sm8 at target (w1), lag8 target-1..+1, sm8 at origin],
    pooled over stations, refit walk-forward on regime origins whose target is already known."""
    def feats(t, k):
        return np.column_stack([y[t], y[t - 1], y[t - 2], sm(y, t + k - P8, 1), y[t + k - P8 - 1], y[t + k - P8], y[t + k - P8 + 1], sm(y, t - P8, 1)])
    out = {}
    for k in range(1, 5):
        for t in origins:
            train = [s for s in range(t - 40, t - k + 1) if s - P8 - 2 >= 0 and s + k <= t]
            if len(train) < 8:
                continue
            X = np.vstack([feats(s, k) for s in train])
            Y = np.concatenate([y[s + k] for s in train])
            mask = np.isfinite(X).all(1) & np.isfinite(Y)
            X, Y = X[mask], Y[mask]
            A = X.T @ X + lam * np.eye(X.shape[1])
            beta = np.linalg.solve(A, X.T @ Y)
            out[(t, k)] = np.clip(feats(t, k) @ beta, 0, None)
    return out


def main() -> None:
    load_dotenv()
    y, index, stations, truth = series()
    start = int(np.searchsorted(index, EVAL_FROM))
    origins = [t for t in range(start, len(index) - 4)]
    print(f"{len(stations)} stations, origins {index[origins[0]]} -> {index[origins[-1]]} ({len(origins)})")
    ridge = ridge_preds(y, origins)
    rows = []
    for t in origins:
        for k in range(1, 5):
            actual = truth[t + k]
            preds = candidates(y, t, k)
            if (t, k) in ridge:
                preds["ridge"] = ridge[(t, k)]
            if "sm8_w1+res0.5_d" in preds and "ridge" in preds:
                preds["media(sm8_w1+res, ridge)"] = (preds["sm8_w1+res0.5_d"] + preds["ridge"]) / 2
            for name, p in preds.items():
                rows.append((index[t], index[t].minute == 0, k, name, np.nansum(np.abs(p - actual)), np.nansum(np.where(np.isnan(actual), np.nan, actual))))
    frame = pd.DataFrame(rows, columns=["origin", "hourly", "k", "cand", "err", "act"])
    dense = frame.groupby("cand")[["err", "act"]].sum()
    dense = 100 * (1 - dense["err"] / dense["act"])
    hourly = frame.loc[frame["hourly"]].groupby(["cand", "origin"])[["err", "act"]].sum()
    per_cycle = (100 * (1 - hourly["err"] / hourly["act"])).groupby("cand")
    by_k = frame.groupby(["cand", "k"])[["err", "act"]].sum()
    by_k = (100 * (1 - by_k["err"] / by_k["act"])).unstack()
    table = pd.DataFrame({"ciclos (media)": per_cycle.mean(), "min ciclo": per_cycle.min(), "n ciclos": per_cycle.size(), "denso": dense}).join(by_k.add_prefix("h"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(table.sort_values("ciclos (media)", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
