import numpy as np
import pandas as pd
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_forecast_cycle import BIAS_MIN_SAMPLES, bias_factor, bias_scale, compute_data_drift, decide_action, population_stability_index


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
