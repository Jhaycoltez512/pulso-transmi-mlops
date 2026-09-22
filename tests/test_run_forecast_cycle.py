import numpy as np
import pandas as pd
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_forecast_cycle import decide_action, population_stability_index


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
