"""Optional MLflow logging that never blocks the durable Supabase lineage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def log_run(
    *,
    run_name: str,
    tags: dict[str, str],
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, Any],
    artifact_paths: list[Path] | None = None,
    model_bundle: dict[str, Any] | None = None,
    registered_model_name: str | None = None,
    dataset: "pd.DataFrame | None" = None,
    dataset_name: str | None = None,
) -> str | None:
    """Log an experiment when a persistent MLflow tracking URI is configured.

    Returning ``None`` means MLflow is intentionally disabled; Supabase remains
    the system of record for the run lineage.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("Install the mlops extra: python -m pip install -e '.[mlops]'") from error

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "pulso-transmi"))
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tags(tags)
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(metrics)
        for name, content in artifacts.items():
            mlflow.log_dict(content, name)
        for path in artifact_paths or []:
            if Path(path).exists():
                mlflow.log_artifact(str(path))
        if model_bundle is not None:
            try:
                register_bundle(mlflow, model_bundle, registered_model_name)
            except Exception as error:  # the registry is a convenience copy; never fail the run over it
                print(f"MLflow model registry skipped: {error}")
        if dataset is not None:
            try:
                log_dataset(mlflow, dataset, dataset_name or run_name)
            except Exception as error:  # same: a convenience copy, never blocks the durable run
                print(f"MLflow dataset logging skipped: {error}")
        return active_run.info.run_id


def log_dataset(mlflow: Any, dataset: "pd.DataFrame", name: str) -> None:
    """Log the exact training snapshot: as an MLflow Dataset (lineage, shows in DagsHub's
    Datasets tab) and as a downloadable parquet artifact, so a past model version can be
    reproduced or rolled back to with the data it was actually trained on, not just a hash."""
    import tempfile

    import mlflow.data
    from mlflow.data.http_dataset_source import HTTPDatasetSource

    supabase_url = os.getenv("SUPABASE_URL")
    source = HTTPDatasetSource(url=f"{supabase_url}/rest/v1/observations") if supabase_url else None
    tracked = mlflow.data.from_pandas(dataset, name=name, source=source)
    mlflow.log_input(tracked, context="training")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.parquet"
        dataset.to_parquet(path)
        mlflow.log_artifact(str(path), artifact_path="dataset")


def register_bundle(mlflow: Any, bundle: dict[str, Any], registered_model_name: str | None) -> None:
    """Log the bundle as a pyfunc model in the active run, register it, and point `champion` at it."""
    import tempfile

    import joblib
    from mlflow_model import PulsoBundleModel

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bundle.joblib"
        joblib.dump(bundle, path)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=PulsoBundleModel(),
            artifacts={"bundle": str(path)},
            code_paths=[str(Path(__file__).with_name("mlflow_model.py"))],
            registered_model_name=registered_model_name,
            pip_requirements=["catboost", "joblib", "numpy", "pandas"],
        )
    version = getattr(info, "registered_model_version", None)
    if registered_model_name and version:
        mlflow.MlflowClient().set_registered_model_alias(registered_model_name, "champion", version)
        print(f"Registered {registered_model_name} v{version} as champion.")
