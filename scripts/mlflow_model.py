"""MLflow pyfunc wrapper for the production bundle, so it can live in the Model Registry.

The bundle is the same dict persisted to Supabase Storage: one CatBoost model per horizon,
the feature list, and the weekly-naive blend weight. Kept in its own module (bundled into
the logged model via code_paths) so mlflow is only imported when a model is registered.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
import mlflow.pyfunc


class PulsoBundleModel(mlflow.pyfunc.PythonModel):
    """Input: the production FEATURES plus `horizon_minutes`, and optionally `weekly_naive`
    (demand 7 days before the target). With `weekly_naive`, returns the blended prediction
    that production submits (before its online bias correction); without it, raw CatBoost."""

    def load_context(self, context: Any) -> None:
        self.bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> np.ndarray:
        features = self.bundle["features"]
        weight = self.bundle["ensemble_weight"]
        output = np.full(len(model_input), np.nan)
        for horizon, model in self.bundle["models"].items():
            mask = (model_input["horizon_minutes"] == int(horizon)).to_numpy()
            if not mask.any():
                continue
            prediction = model.predict(model_input.loc[mask, features])
            if "weekly_naive" in model_input:
                naive = model_input.loc[mask, "weekly_naive"].to_numpy(dtype=float)
                prediction = weight * prediction + (1 - weight) * np.where(np.isnan(naive), prediction, naive)
            output[mask] = np.clip(prediction, 0, None)
        return output
