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

## Ciclo operativo: evaluar, decidir, predecir, enviar, registrar

`scripts/run_forecast_cycle.py` corre después del collector, en el mismo job.
Si no hay un ciclo abierto (`GET /v1/forecast-cycles/current`), termina sin
tocar nada. Si lo hay:

1. **Evaluar**: llena `actual_demand` en `predictions` para las filas cuyo
   `target_at` ya tiene una observación real, comparando contra la predicción
   guardada en su momento.
2. **Medir**: WAPE reciente (últimos 3 días de predicciones ya evaluadas, con
   al menos 20 muestras por horizonte) y drift de datos — PSI (Population
   Stability Index, 10 bins) de `demand`, `rain_forecast`,
   `temperature_forecast` y `event_intensity`, comparando la ventana de
   entrenamiento del modelo activo (14 días hasta su `training_data_end`)
   contra los últimos 3 días. Ambos se guardan en `drift_measurements`.
3. **Decidir** — regla explícita, en este orden (la primera que aplique gana
   y su texto queda en `forecast_runs.trigger_reason`):
   1. `bootstrap: no active model` — no hay modelo activo todavía.
   2. `stale: last trained N days ago` — el modelo activo tiene más de 7 días.
   3. `performance_drift: h{H} recent WAPE {x} > threshold {y}` — el WAPE
      reciente de algún horizonte supera en 15% el WAPE de validación *del
      propio modelo activo* (ese umbral se calculó una vez al entrenarlo, no
      se recalcula cada hora).
   4. `data_drift: {feature} PSI={x} > 0.25` — PSI de alguna variable supera
      0.25 (umbral estándar de industria para "cambio significativo").
   5. Si nada aplica: `stable: no trigger met` → se conserva el modelo.

   `event_intensity` es una variable con eventos poco frecuentes; PSI puede
   marcar drift con más frecuencia que las otras variables porque cualquier
   ventana de 3 días sin eventos reales difiere bastante de una ventana de 14
   días que sí tuvo alguno. No es un error — es la naturaleza de una señal
   dispersa — pero conviene tenerlo presente al leer `trigger_reason`.

4. **Conservar o reentrenar**: el modelo entrenado (`.joblib`) se sube a un
   bucket privado de Supabase Storage (`models`) al reentrenar, y se descarga
   de ahí al conservar — así "conservar" evita reentrenar de verdad entre
   corridas de GitHub Actions (cada corrida es una VM nueva, sin disco
   persistente). `model_versions.is_active` marca cuál es el vigente; solo
   puede haber uno (índice único parcial). Si falla la descarga, cae a
   reentrenar como salvavidas y lo dice en la razón registrada.
5. **Predecir**: las 12 estaciones × horizontes que pida el ciclo (48 valores
   si pide los 4 horizontes), reusando `prediction_rows` de
   `train_catboost_direct.py`.
6. **Enviar**: valida el payload localmente (mismo `validate()` que usa la
   vista previa) y hace `POST /v1/submissions` con `Idempotency-Key:
   {cycle_id}:{model_version}` y la versión del modelo + commit de Git. El
   resultado (aceptada, rechazada o fallida) se guarda en `submissions`.

   El envío real está **activado por defecto**. Para apagarlo sin tocar
   código — por ejemplo mientras se depura algo — configura la variable de
   repositorio `PULSO_SUBMIT_ENABLED=false` en GitHub; el pipeline sigue
   corriendo completo (evalúa, decide, predice, valida) pero no hace el POST,
   y la fila en `submissions` queda en `pending` con la razón.
7. **Registrar**: `forecast_runs` se crea apenas se confirma que hay ciclo
   abierto (antes de decidir o predecir), así que cualquier falla en
   cualquier paso posterior queda con `status='failed'` y `error_message`, no
   solo las fallas de la submission.

```bash
python scripts/run_forecast_cycle.py
```

## Automatización en GitHub Actions

El workflow `.github/workflows/collector.yml` corre el collector y luego
`run_forecast_cycle.py`, en ese orden, en el mismo job, al minuto 03 de cada
hora. GitHub no garantiza disparos puntuales de `schedule` con frecuencia
menor a una hora: en pruebas, un cron cada 15 minutos disparó una sola vez en
más de dos horas, tanto en repositorio privado como público. Como ambos
scripts son idempotentes (el collector no duplica filas; el orquestador crea
un `forecast_run` nuevo por corrida y su decisión no depende de cuántas
corridas hubo antes), una corrida horaria no pierde nada, solo acumula.
Configura estos secrets en GitHub:

- `PULSO_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`
- `MLFLOW_TRACKING_URI` (opcional; omítelo si se usa solo Supabase)

Y estas variables (`vars`, no secrets):

- `PULSO_API_URL`
- `PULSO_SUBMIT_ENABLED` (opcional; `false` para desactivar el envío real)

GitHub ejecuta workflows programados desde la rama predeterminada: incorpora
este archivo en esa rama antes de esperar ejecuciones automáticas.
