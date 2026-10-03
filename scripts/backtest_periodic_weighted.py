"""Backtest weighted refinements of the period average on the live 4h regime (read-only).

Production (since PR #11) submits ~the plain mean of the last 6 periods (per_4h) once the regime
has run a day. Rivals reach 93-95% per cycle. Every candidate below only uses copies at least one
period (4h) old, so the error of any candidate on targets up to the origin is known at the origin
without leakage -- the adaptive variants pick weights from those recent errors.

Score per cycle = 1 - WAPE over the cycle's 48 predictions (the leaderboard metric).

    cd scripts && python backtest_periodic_weighted.py
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from train_baseline import load_training_data
from load_supabase import load_dotenv

P = 16  # 4h in 15-min steps
REGIME_START = pd.Timestamp("2026-09-18T05:00:00Z")
EVAL_START = REGIME_START + timedelta(hours=24)
DECAYS = (0.3, 0.5, 0.7, 0.85, 1.0)


def lag_stack(Y: np.ndarray, tau: np.ndarray, k_max: int) -> np.ndarray:
    """(k_max, len(tau), stations): Y at tau - k*P, NaN when before the series."""
    out = np.full((k_max, len(tau), Y.shape[1]), np.nan)
    for k in range(1, k_max + 1):
        idx = tau - k * P
        ok = idx >= 0
        out[k - 1, ok] = Y[idx[ok]]
    return out


def ewm(L: np.ndarray, decay: float, k: int = 6) -> np.ndarray:
    w = decay ** np.arange(k)
    stack = L[:k]
    weights = np.where(np.isnan(stack), 0, w[:, None, None])
    return np.nansum(np.nan_to_num(stack) * weights, axis=0) / np.where(weights.sum(0) > 0, weights.sum(0), np.nan)


def smooth(Y: np.ndarray) -> np.ndarray:
    """Centered [0.25, 0.5, 0.25] smoothing in time (only used on copies >= 4h old)."""
    S = Y.copy()
    S[1:-1] = 0.25 * Y[:-2] + 0.5 * Y[1:-1] + 0.25 * Y[2:]
    return S


def candidates(Y: np.ndarray, tau: np.ndarray, origin_idx: int) -> dict[str, np.ndarray]:
    """Predictions for targets tau (indices) made at origin_idx; each (len(tau), stations)."""
    L = lag_stack(Y, tau, 12)
    out = {
        "media 6 (produccion)": np.nanmean(L[:6], axis=0),
        "media 8": np.nanmean(L[:8], axis=0),
        "media 12": np.nanmean(L[:12], axis=0),
        "ewm 0.85": ewm(L, 0.85),
        "ewm 0.7": ewm(L, 0.7),
    }
    srt = np.sort(L[:6], axis=0)
    out["media recortada 6 (sin min/max)"] = np.nanmean(srt[1:5], axis=0)
    out["mediana 6"] = np.nanmedian(L[:6], axis=0)
    out["media 8 recortada"] = np.nanmean(np.sort(L[:8], axis=0)[1:7], axis=0)
    S = smooth(Y)
    out["media 6 suavizada"] = np.nanmean(lag_stack(S, tau, 6), axis=0)
    out["media 8 suavizada"] = np.nanmean(lag_stack(S, tau, 8), axis=0)

    # adaptive per station: recent targets (already observed at the origin), last 6h
    recent = np.arange(origin_idx - 23, origin_idx + 1)
    Lr = lag_stack(Y, recent, 12)
    Yr = Y[recent]
    errs = {d: np.nanmean(np.abs(ewm(Lr, d) - Yr), axis=0) for d in DECAYS}
    best = np.argmin(np.vstack([errs[d] for d in DECAYS]), axis=0)  # per station
    out["ewm decaimiento por estacion (6h)"] = np.column_stack([ewm(L, DECAYS[b])[:, s] for s, b in enumerate(best)])

    # per-station weights over 6 lags, ridge shrunk towards equal weights, fit on last 24h
    fit = np.arange(origin_idx - 95, origin_idx + 1)
    Lf, Yf = lag_stack(Y, fit, 6), Y[fit]
    for lam_name, lam in (("fuerte", 30.0), ("media", 5.0)):
        pred = np.full((len(tau), Y.shape[1]), np.nan)
        for s in range(Y.shape[1]):
            X, y = Lf[:, :, s].T, Yf[:, s]
            ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            scale = np.nanmean(y[ok]) ** 2 if ok.any() else 1.0
            prior = np.full(6, 1 / 6)
            w = np.linalg.solve(X[ok].T @ X[ok] + lam * scale * np.eye(6), X[ok].T @ y[ok] + lam * scale * prior) if ok.sum() > 10 else prior
            pred[:, s] = np.nan_to_num(L[:6, :, s].T) @ w
        out[f"pesos por estacion (ridge {lam_name})"] = np.clip(pred, 0, None)

    # per-station multiplicative bias of the 6-period mean over the last 4h, shrunk
    base_r = np.nanmean(Lr[:6, -16:], axis=0)
    ratio = np.nansum(Yr[-16:], axis=0) / np.nansum(base_r, axis=0)
    ratio = np.clip(np.nan_to_num(ratio, nan=1.0), 0.7, 1.4)
    out["media 6 x sesgo por estacion^0.5"] = out["media 6 (produccion)"] * np.sqrt(ratio)
    # blend of the best two families
    out["mezcla media 8 suav. + ewm por estacion"] = 0.5 * out["media 8 suavizada"] + 0.5 * out["ewm decaimiento por estacion (6h)"]
    return out


def main() -> None:
    load_dotenv()
    data = load_training_data()
    wide = data.pivot_table(index="observed_at", columns="station_id", values="demand", aggfunc="last").sort_index().asfreq("15min")
    Y = wide.to_numpy(dtype=float)
    times = wide.index
    last_full = wide.dropna(how="any").index.max()
    origins = [t for t in pd.date_range(EVAL_START, last_full - timedelta(hours=1), freq="1h")]
    print(f"{len(origins)} ciclos horarios {origins[0]} -> {origins[-1]}")
    rows = []
    for t in origins:
        o = times.get_loc(t)
        tau = np.array([o + 1, o + 2, o + 3, o + 4])
        actual = Y[tau]
        for name, pred in candidates(Y, tau, o).items():
            err = np.abs(np.nan_to_num(pred) - actual)
            per_h = 100 * np.clip(1 - err.sum(axis=1) / actual.sum(axis=1), 0, None)
            rows.append({"variante": name, "origin": t, "ciclo": 100 * max(0.0, 1 - err.sum() / actual.sum()),
                         **{f"h{15 * (i + 1)}": v for i, v in enumerate(per_h)}})
    frame = pd.DataFrame(rows)
    base = frame.loc[frame["variante"] == "media 6 (produccion)"].set_index("origin")["ciclo"]
    frame["gana"] = frame.apply(lambda r: r["ciclo"] > base[r["origin"]] + 1e-9, axis=1)
    summary = frame.groupby("variante").agg(
        ciclo=("ciclo", "mean"), minimo=("ciclo", "min"), maximo=("ciclo", "max"),
        gana_a_media6=("gana", "sum"), h15=("h15", "mean"), h30=("h30", "mean"), h45=("h45", "mean"), h60=("h60", "mean"),
    ).sort_values("ciclo", ascending=False)
    pd.set_option("display.width", 220)
    print(summary.round(2).to_string())
    last8 = frame.loc[frame["origin"] >= origins[-8]].groupby("variante")["ciclo"].mean().sort_values(ascending=False)
    print("\nultimos 8 ciclos:")
    print(last8.round(2).to_string())


if __name__ == "__main__":
    main()
