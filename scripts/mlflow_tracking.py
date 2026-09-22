"""Optional MLflow logging that never blocks the durable Supabase lineage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def log_run(
    *,
    run_name: str,
    tags: dict[str, str],
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, Any],
    artifact_paths: list[Path] | None = None,
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
            mlflow.log_artifact(str(path))
        return active_run.info.run_id
