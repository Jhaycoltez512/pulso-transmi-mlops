# Collector incremental y trazabilidad

`scripts/sync_stream_observations.py` consulta `/v1/stream/observations` y
agrega exclusivamente las filas cuyo `released_at` es posterior al último corte
persistido en `sync_state`. Esto también captura datos liberados tarde aunque su
`observed_at` sea anterior. Por tanto es seguro ejecutarlo repetidamente.

Cada ejecución queda registrada en `ingestion_runs` con:

- estado, timestamps, filas leídas y nuevas;
- `data_version`, un identificador reproducible construido con el último
  timestamp y el hash del bloque recibido;
- commit del código;
- `mlflow_run_id`, si MLflow está habilitado.

El campo `released_at` de la API se guarda en `observations`, separado de
`observed_at`: el primero indica cuándo la API liberó el dato y el segundo a qué
momento corresponde la demanda.

## Ejecución local

```bash
python scripts/sync_stream_observations.py
```

Requiere `PULSO_API_URL`, `PULSO_API_KEY`, `SUPABASE_URL` y
`SUPABASE_SECRET_KEY` en `.env`.

## MLflow

Supabase es el registro durable mínimo. Para centralizar experimentos y
artefactos en MLflow, configura un servidor persistente y agrega:

```dotenv
MLFLOW_TRACKING_URI=https://<tu-servidor-mlflow>
MLFLOW_EXPERIMENT_NAME=pulso-transmi
```

Después instala el extra:

```bash
python -m pip install -e '.[mlops]'
```

Con esa URI, cada recolección registra parámetros, métricas y un manifiesto de
datos en MLflow; el identificador del run queda enlazado con `ingestion_runs`.
Si no existe una URI, el collector sigue funcionando y registra la trazabilidad
en Supabase, pero no intenta usar un almacenamiento local efímero.

`scripts/train_catboost_direct.py` sigue el mismo patrón para los modelos:
registra `data_version`, commit, métricas por horizonte, artefacto Joblib y el
`mlflow_run_id` en `model_versions` y `training_runs`. Si no hay ciclo abierto,
entrena y versiona el modelo, pero no genera una submission hasta que la API
publique objetivos nuevos.

## Automatización en GitHub Actions

El workflow `.github/workflows/collector.yml` programa la ejecución a los
minutos 03, 18, 33 y 48 de cada hora. Configura estos secrets en GitHub:

- `PULSO_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`
- `MLFLOW_TRACKING_URI` (opcional; omítelo si se usa solo Supabase)

Configura también la variable `PULSO_API_URL`. GitHub ejecuta workflows
programados desde la rama predeterminada: incorpora este archivo en esa rama
antes de esperar ejecuciones automáticas.
