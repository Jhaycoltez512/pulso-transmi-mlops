import numpy as np
import pandas as pd
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_forecast_cycle import BIAS_MIN_SAMPLES, DRIFT_MIN_VALID_SAMPLES, bias_factor, bias_scale, compute_data_drift, decide_action, population_stability_index


def test_psi_is_near_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(100, 15, size=2000)
    recent = rng.normal(100, 15, size=500)
    assert population_stability_index(reference, recent) < 0.05


def test_psi_is_large_for_a_shifted_distribution() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(100, 15, size=2000)
    recent = rng.normal(160, 15, size=500)
    assert population_stability_index(reference, recent) > 0.25


def test_decide_action_bootstraps_without_an_active_model() -> None:
    action, reason = decide_action(None, {}, {}, [], pd.Timestamp.now(tz="UTC"))
    assert action == "retrain"
    assert "bootstrap" in reason


def test_decide_action_retrains_when_stale() -> None:
    now = pd.Timestamp.now(tz="UTC")
    active_model = {"trained_at": (now - timedelta(days=10)).isoformat()}
    action, reason = decide_action(active_model, {}, {}, [], now)
    assert action == "retrain"
    assert "stale" in reason


def test_decide_action_retrains_on_performance_drift() -> None:
    now = pd.Timestamp.now(tz="UTC")
    active_model = {"trained_at": now.isoformat()}
    performance = {15: {"wape": 0.20, "samples": 50}}
    thresholds = {"15": 0.15}
    action, reason = decide_action(active_model, performance, thresholds, [], now)
    assert action == "retrain"
    assert "performance_drift" in reason


def test_decide_action_retrains_on_triggered_data_drift() -> None:
    now = pd.Timestamp.now(tz="UTC")
    active_model = {"trained_at": now.isoformat()}
    drift_rows = [{"feature_name": "demand", "value": 0.4, "threshold": 0.25, "triggered": True}]
    action, reason = decide_action(active_model, {}, {}, drift_rows, now)
    assert action == "retrain"
    assert "data_drift" in reason


def test_decide_action_keeps_when_stable() -> None:
    now = pd.Timestamp.now(tz="UTC")
    active_model = {"trained_at": now.isoformat()}
    performance = {15: {"wape": 0.12, "samples": 50}}
    thresholds = {"15": 0.15}
    drift_rows = [{"feature_name": "demand", "value": 0.05, "threshold": 0.25, "triggered": False}]
    action, reason = decide_action(active_model, performance, thresholds, drift_rows, now)
    assert action == "keep"
    assert "stable" in reason


def test_bias_factor_uses_raw_predictions_not_the_corrected_ones() -> None:
    # submitted (corrected) value 110, raw model output 100, actual 105 -> factor from raw: 1.05
    rows = [{"actual_demand": 105, "predicted_demand": 110.0, "raw_predicted_demand": 100.0}] * BIAS_MIN_SAMPLES
    factor, samples = bias_factor(rows)
    assert samples == BIAS_MIN_SAMPLES
    assert abs(factor - 1.05) < 1e-9


def test_bias_factor_falls_back_to_predicted_for_rows_before_the_correction_existed() -> None:
    rows = [{"actual_demand": 90, "predicted_demand": 100.0, "raw_predicted_demand": None}] * BIAS_MIN_SAMPLES
    factor, _ = bias_factor(rows)
    assert abs(factor - 0.9) < 1e-9


def test_bias_factor_needs_enough_evaluated_samples() -> None:
    rows = [{"actual_demand": 90, "predicted_demand": 100.0, "raw_predicted_demand": 100.0}] * (BIAS_MIN_SAMPLES - 1)
    rows += [{"actual_demand": None, "predicted_demand": 100.0, "raw_predicted_demand": 100.0}] * 10
    factor, samples = bias_factor(rows)
    assert factor is None
    assert samples == BIAS_MIN_SAMPLES - 1


def test_bias_scale_applies_half_the_correction_and_clips() -> None:
    assert bias_scale(None) == 1.0
    assert abs(bias_scale(1.10) - 1.05) < 1e-9
    assert abs(bias_scale(0.90) - 0.95) < 1e-9
    assert bias_scale(3.0) == 1.25
    assert bias_scale(0.1) == 0.8


def test_event_intensity_uses_its_own_higher_drift_threshold() -> None:
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2026-01-01", periods=20 * 96, freq="15min", tz="UTC")
    data = pd.DataFrame({
        "observed_at": timestamps,
        "demand": rng.normal(300, 50, len(timestamps)),
        "rain_forecast": rng.random(len(timestamps)),
        "temperature_forecast": rng.normal(18, 2, len(timestamps)),
        "event_intensity": rng.random(len(timestamps)),
    })
    rows = compute_data_drift(data, {"training_data_end": timestamps[16 * 96].isoformat()})
    thresholds = {row["feature_name"]: row["threshold"] for row in rows}
    assert thresholds["event_intensity"] == 2.0
    assert thresholds["demand"] == 0.25


def _synthetic_data(days: int, stations: int = 12) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=days * 96, freq="15min", tz="UTC")
    frame = pd.concat(
        [pd.DataFrame({
            "observed_at": timestamps, "station_id": f"{i:05d}",
            "demand": np.random.default_rng(i).normal(300, 50, len(timestamps)),
            "rain_forecast": np.random.default_rng(0).random(len(timestamps)),
            "temperature_forecast": np.random.default_rng(0).normal(18, 2, len(timestamps)),
            "event_intensity": np.random.default_rng(0).random(len(timestamps)),
        }) for i in range(stations)],
        ignore_index=True,
    )
    return frame


def test_sparse_recent_context_does_not_trigger_even_if_psi_is_high() -> None:
    data = _synthetic_data(days=20)
    # blank most of the 3-day recent window, keeping only the last day -> well under the 150
    # unique-timestamp minimum, the way real context data lagging observations by days does.
    recent_start = data["observed_at"].max() - timedelta(days=3)
    cutoff = data["observed_at"].max() - timedelta(days=1)
    sparse_mask = (data["observed_at"] > recent_start) & (data["observed_at"] < cutoff)
    data.loc[sparse_mask, "temperature_forecast"] = np.nan

    rows = compute_data_drift(data, {"training_data_end": (data["observed_at"].max() - timedelta(days=3)).isoformat()})
    temp_row = next(r for r in rows if r["feature_name"] == "temperature_forecast")
    assert temp_row["details"]["insufficient_samples"] is True
    assert temp_row["details"]["valid_samples"] < DRIFT_MIN_VALID_SAMPLES
    assert temp_row["triggered"] is False  # even though blanking most of the window can inflate PSI


def test_context_features_are_deduplicated_across_stations_before_counting() -> None:
    data = _synthetic_data(days=20, stations=12)
    rows = compute_data_drift(data, {"training_data_end": (data["observed_at"].max() - timedelta(days=3)).isoformat()})
    temp_row = next(r for r in rows if r["feature_name"] == "temperature_forecast")
    demand_row = next(r for r in rows if r["feature_name"] == "demand")
    # 3 days * 96 periods = 288 unique timestamps, not 288*12 duplicated rows
    assert temp_row["details"]["valid_samples"] <= 288
    assert temp_row["details"]["valid_samples"] >= DRIFT_MIN_VALID_SAMPLES
    # demand is genuinely per-station, so its count is much larger
    assert demand_row["details"]["valid_samples"] > temp_row["details"]["valid_samples"]


def test_models_to_prune_keeps_active_and_newest_bundles() -> None:
    from run_forecast_cycle import models_to_prune

    objects = [{"name": f"catboost-{i:02d}.joblib", "created_at": f"2026-09-30T{i:02d}:00:00Z"} for i in range(10)]
    objects.append({"name": "README.txt", "created_at": "2026-09-30T23:00:00Z"})
    stale = models_to_prune(objects, keep={"catboost-05.joblib"}, keep_latest=3, keep_oldest=2)
    # oldest two (00, 01), newest three (07, 08, 09) and the active one (05) survive;
    # non-bundles are never touched
    assert sorted(stale) == [f"catboost-{i:02d}.joblib" for i in (2, 3, 4, 6)]


def test_lineage_reads_the_observations_collector_not_the_context_sync() -> None:
    from train_catboost_direct import LATEST_OBSERVATIONS_INGESTION

    assert LATEST_OBSERVATIONS_INGESTION["source_name"] == "eq.pulso-transmi-stream"
    assert LATEST_OBSERVATIONS_INGESTION["data_version"] == "not.is.null"


def test_recent_performance_only_scores_the_active_models_predictions() -> None:
    from run_forecast_cycle import recent_performance_params

    params = recent_performance_params("2026-09-30T00:00:00+00:00", "model-123")
    assert params["forecast_runs.model_version_id"] == "eq.model-123"
    assert "forecast_runs!inner(model_version_id)" in params["select"]
    assert "forecast_runs.model_version_id" not in recent_performance_params("2026-09-30T00:00:00+00:00", None)


def test_recent_performance_reads_every_page() -> None:
    from load_supabase import SupabaseLoader
    from run_forecast_cycle import recent_performance

    # 2500 evaluated predictions (past the 1000-row cap): h15 is perfect, h60 always 50% off
    rows = [{"id": i, "station_id": "07111", "horizon_minutes": 15 if i % 2 else 60, "predicted_demand": 100.0,
             "raw_predicted_demand": 100.0 if i % 2 else 150.0, "actual_demand": 100} for i in range(2500)]
    loader = SupabaseLoader.__new__(SupabaseLoader)
    loader.select = lambda table, params: rows[params["offset"]:params["offset"] + params["limit"]]
    result = recent_performance(loader, [15, 60], "model-123")
    assert result[15]["samples"] + result[60]["samples"] == 2500
    assert result[15]["wape"] == 0.0 and abs(result[60]["wape"] - 0.5) < 1e-9


def _station_frame(days: int = 16, stations: tuple[str, ...] = ("05000", "05100")) -> pd.DataFrame:
    timestamps = pd.date_range("2026-09-01", periods=days * 96, freq="15min", tz="UTC")
    rng = np.random.default_rng(3)
    return pd.concat([
        pd.DataFrame({"station_id": s, "observed_at": timestamps, "demand": 300 + rng.normal(0, 15, len(timestamps))})
        for s in stations
    ], ignore_index=True)


def test_station_drift_fires_only_for_the_station_that_changed() -> None:
    from run_forecast_cycle import compute_station_drift

    data = _station_frame()
    end = data["observed_at"].max()
    trained = end - timedelta(days=2)
    data.loc[(data["station_id"] == "05100") & (data["observed_at"] > trained), "demand"] *= 0.4
    rows = {row["details"]["station_id"]: row for row in compute_station_drift(data, {"training_data_end": trained.isoformat()})}
    assert rows["05100"]["triggered"] is True
    assert 0.35 < rows["05100"]["details"]["relative_change"] < 0.45
    assert rows["05000"]["triggered"] is False


def test_station_drift_does_not_refire_after_retraining_on_the_new_level() -> None:
    from run_forecast_cycle import compute_station_drift

    data = _station_frame()
    end = data["observed_at"].max()
    data.loc[(data["station_id"] == "05100") & (data["observed_at"] > end - timedelta(days=2)), "demand"] *= 0.4
    # a model retrained at the current cutoff already saw the new level: nothing new to react to
    rows = compute_station_drift(data, {"training_data_end": end.isoformat()})
    assert rows and not any(row["triggered"] for row in rows)


def test_decide_action_names_the_station_on_station_drift() -> None:
    now = pd.Timestamp.now(tz="UTC")
    drift_rows = [{
        "feature_name": "station_level:05100", "value": 0.9, "threshold": float(np.log(1.5)), "triggered": True,
        "details": {"relative_change": 0.41},
    }]
    action, reason = decide_action({"trained_at": now.isoformat()}, {}, {}, drift_rows, now)
    assert action == "retrain"
    assert reason == "data_drift: station_level:05100 x0.41 vs training > x1.50"
