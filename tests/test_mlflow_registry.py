import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")  # the mlops extra is optional; CI installs only dev,ml

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from catboost import CatBoostRegressor
from mlflow_tracking import register_bundle


def test_bundle_is_registered_and_loadable_by_champion_alias(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("registry-test")
    features = ["station_id", "x"]
    frame = pd.DataFrame({"station_id": ["03000", "05000"] * 20, "x": np.arange(40, dtype=float)})
    target = frame["x"] * 2 + 5
    models = {}
    for horizon in (15, 30):
        model = CatBoostRegressor(iterations=20, verbose=False, allow_writing_files=False)
        model.fit(frame[features], target, cat_features=["station_id"])
        models[horizon] = model
    bundle = {"models": models, "features": features, "horizons": [15, 30], "ensemble_weight": 0.75}

    with mlflow.start_run():
        register_bundle(mlflow, bundle, "pulso-test")

    loaded = mlflow.pyfunc.load_model("models:/pulso-test@champion")
    request = frame.head(4).assign(horizon_minutes=[15, 30, 15, 30])
    raw = loaded.predict(request)
    assert raw.shape == (4,) and not np.isnan(raw).any()

    # with the weekly-naive column it blends 0.75 model + 0.25 naive, like production
    blended = loaded.predict(request.assign(weekly_naive=0.0))
    assert np.allclose(blended, 0.75 * raw)
