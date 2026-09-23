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
- `mlflow_run_id`, solo si ese dato terminó usándose en un reentreno (ver
  "MLflow" más abajo).

El campo `released_at` de la API se guarda en `observations`, separado de
`observed_at`: el primero indica cuándo la API liberó el dato y el segundo a qué
momento corresponde la demanda.

### Manejo de fallas de red

- **Registro antes de llamar a la API.** La fila en `ingestion_runs` se crea
  *antes* de consultar la API. Si la llamada falla (timeout, error de
  conexión, error HTTP), queda con `status='failed'` y el error en
  `error_message`. Antes se creaba después, y un `ConnectTimeout` real
  (23-sep) tumbó la corrida sin dejar ningún registro.
- **Reintentos.** Tanto esta llamada como la consulta de ciclo abierto
  (`active_cycle()`) reintentan hasta 3 veces con espera exponencial, solo
  cuando falla la *conexión*. En ese caso la petición nunca llegó al servidor,
  así que reintentar es seguro. El límite para conectar es de 15 s, así que el
  peor caso son ~1 minuto de reintentos. Se agregó tras dos `ConnectTimeout`
  transitorios el 23-sep. Cuando el collector falla, GitHub Actions no corre
  el paso de predicción de esa corrida. Un fallo pasajero ya no hace perder
  el intento de envío del ciclo, y si la API está caída de verdad, la
  siguiente corrida (10 min después) vuelve a intentar dentro de la misma
  ventana de 25 minutos.

## Contexto (clima y eventos)

`scripts/sync_context.py` es el equivalente de arriba para `/v1/context`
(lluvia, temperatura, intensidad de eventos). No es el mismo tipo de script
por una razón real: `/v1/context` no tiene una variante `/v1/stream/` ni
semántica de `released_at` como observaciones — es el endpoint de lectura
normal (`start`/`end`/`cursor`), el mismo que usa la carga inicial
(`load_supabase.py`). La sincronización es incremental por `start=
<último observed_at sincronizado>` en vez de por `released_at`.

**Por qué se agregó (23-sep):** el drift de `temperature_forecast` empezó a
dar lecturas erráticas (ver "Ciclo operativo" más abajo, punto 3 de la regla
de decisión). Se encontró que `sync_state` de `context` no se
tocaba desde el 18-sep — la carga inicial nunca tuvo seguimiento automático,
mientras que `observations` se actualizaba solo cada 10 minutos. El contexto
llegó a estar más de 2 días atrás de las observaciones.

**Esto no afectaba al modelo.** `train_catboost_direct.py` (el modelo en
producción) no usa clima/eventos como feature — se probó y se descartó (ver
`docs/ml-baselines.md`). Lo único afectado era la detección de drift: con
ventanas "recientes" de 3 días llenas en su mayoría de valores nulos (por el
atraso), el PSI de clima fluctuaba de forma artificial y podía disparar
reentrenos innecesarios — no dañinos, pero de más.

**Con el fix en producción se descubrió algo más:** al correrlo, solo trajo
1 fila nueva. Consultando la API en vivo directamente (sin pasar por
Supabase) se confirmó que **no hay contexto más nuevo disponible todavía** —
la propia API libera el contexto más lento que las observaciones, algo que
ningún script de nuestro lado puede adelantar. El fix sigue siendo necesario:
sin él, el contexto se habría quedado congelado para siempre incluso cuando
la API sí libere datos nuevos. Con él, se pone al día automáticamente cada
10 minutos en cuanto haya algo que sincronizar.

## Ejecución local

```bash
python scripts/sync_stream_observations.py
python scripts/sync_context.py
```

Requiere `PULSO_API_URL`, `PULSO_API_KEY`, `SUPABASE_URL` y
`SUPABASE_SECRET_KEY` en `.env`.

## MLflow

Supabase es el registro durable mínimo. Para centralizar experimentos y
artefactos en MLflow, configura un servidor persistente y agrega:

```dotenv
MLFLOW_TRACKING_URI=https://<tu-servidor-mlflow>
MLFLOW_TRACKING_USERNAME=<usuario>       # si el servidor pide autenticación
MLFLOW_TRACKING_PASSWORD=<token>         # si el servidor pide autenticación
MLFLOW_EXPERIMENT_NAME=pulso-transmi
```

`MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` los lee el cliente de
MLflow directamente (autenticación HTTP básica estándar); no hace falta
tocar `scripts/mlflow_tracking.py` para usarlos.

**Opción gratuita sin infraestructura propia: [DagsHub](https://dagshub.com).**
Crea una cuenta (puede ser con GitHub) y un repositorio ahí — puede estar
vacío, solo se usa como backend de MLflow. En la pestaña "Remote" →
"Experiments" del repo aparece la URL (`https://dagshub.com/<usuario>/
<repo>.mlflow`); en Settings → Tokens generas el valor para
`MLFLOW_TRACKING_PASSWORD` (usa un token, no la contraseña de la cuenta).

En GitHub Actions, agrega los tres como secrets del repositorio
(`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`)
— el workflow ya los pasa a ambos pasos del pipeline
(`.github/workflows/collector.yml`).

Después instala el extra:

```bash
python -m pip install -e '.[mlops]'
```

`scripts/sync_stream_observations.py` (el collector) **no** loguea a MLflow —
corre cada 10 minutos, y crear un run ahí en cada corrida ensuciaría el
experimento con entradas casi siempre iguales. Supabase sigue siendo la
trazabilidad durable de cada sincronización, con o sin MLflow configurado.

Los datos y el modelo se versionan juntos en MLflow **solo cuando el modelo
se reentrena** — `record_lineage()` en `scripts/train_catboost_direct.py`,
que `run_forecast_cycle.py` solo invoca cuando la regla de decisión (drift o
desempeño) dice `retrain`. Ese único run de MLflow incluye: `data_version`
como tag, un `data_manifest.json` (filas leídas, `last_observed_at`, y el
`ingestion_run_id` exacto que produjo ese dato), métricas por horizonte,
`metrics.json` y el modelo en sí. El mismo `mlflow_run_id`
queda enlazado en Supabase en tres tablas a la vez: `model_versions`,
`training_runs`, e `ingestion_runs` — así una sola corrida de MLflow
representa "este dato + este modelo", trazable desde cualquiera de las tres.
Si no hay ciclo abierto o la regla decide `keep`, no se genera ningún run
nuevo de MLflow — el modelo activo simplemente se reutiliza.

**Model Registry.** Cada modelo reentrenado se registra además en el Model
Registry de MLflow (pestaña **Models** en DagsHub) como una versión nueva de
`pulso-catboost`, y el alias `champion` pasa a apuntar a ella. Así queda
separado el historial de entrenamientos (Experiments) del modelo versionado
y el que está en producción (Models). El modelo registrado es un *pyfunc*
(`scripts/mlflow_model.py`) que envuelve el bundle completo: los 4 CatBoost
por horizonte y el peso de mezcla con el naive semanal. Se puede cargar desde
cualquier lado con:

```python
import mlflow
model = mlflow.pyfunc.load_model("models:/pulso-catboost@champion")
# entrada: FEATURES + horizon_minutes (+ weekly_naive para obtener la mezcla)
```

Devuelve la predicción *antes* de la corrección de sesgo en línea, que
depende de las predicciones recientes y se aplica en `run_forecast_cycle.py`.
El pipeline sigue cargando el modelo desde Supabase Storage para predecir
(es lo que ya estaba probado): el Registry es el registro formal y visible,
y si falla nunca bloquea el envío.

**Versionado del dataset.** El mismo run que registra el modelo también
versiona el snapshot exacto de datos con el que se entrenó, vía
`scripts/mlflow_tracking.py:log_dataset()`. Queda de dos formas dentro del
run, no en corridas separadas:

- Como **MLflow Dataset** (`mlflow.data.from_pandas` + `mlflow.log_input`),
  visible en la pestaña "Datasets" de DagsHub, enlazado al run que lo generó.
  El `source` apunta al endpoint REST de Supabase del que salen las
  observaciones (`SUPABASE_URL`); si esa variable no está configurada en el
  entorno donde corre el reentrenamiento, MLflow usa como source la
  ubicación del código que llamó a `from_pandas` — el Dataset igual queda
  versionado, solo cambia la procedencia declarada.
- Como **artifact parquet descargable** (`dataset/<data_version>.parquet`),
  para poder recuperar el DataFrame completo tal cual, no solo su hash o
  esquema — necesario para poder reentrenar o auditar una versión anterior
  con los datos originales, no una aproximación.

Igual que el Model Registry, esto es una copia de conveniencia: si el
logging del dataset falla por cualquier razón, se imprime un aviso
("MLflow dataset logging skipped: ...") pero el run y el reentrenamiento en
sí nunca se bloquean por eso.

## Mecánica de los ciclos

No está documentada en este SDK ni en la plantilla original — se confirmó a
partir de la configuración del backend de la API (`.env.example` del servicio,
valores oficiales):

- `RELEASE_INTERVAL_MINUTES=30`: se libera un ciclo nuevo cada 30 minutos.
- `SUBMISSION_WINDOW_MINUTES=25`: cada ciclo acepta submissions durante 25
  minutos desde que abre (coincide con lo medido en el único ciclo real que
  observamos: abrió y cerró con 25 minutos de diferencia).
- `SUBMISSION_MAX_ATTEMPTS=3`: máximo 3 intentos válidos por ciclo.

Con un ciclo cada 30 minutos y una ventana de 25, el schedule de GitHub
Actions corre cada 10 minutos (`*/10 * * * *`, según la guía oficial del
curso). No es porque GitHub dispare puntual cada 10 minutos — no lo hace,
ver más abajo —, sino lo contrario: como los disparos de `schedule` son poco
confiables, más intentos por ventana (2-3 en 25 minutos) suben la probabilidad
de que *alguno* caiga dentro. Espaciar el cron para "ser más confiable" (lo
que se intentó primero: horario, luego cada 30 min) resultó ser el enfoque
equivocado para ciclos tan angostos y frecuentes.

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
      se recalcula en cada corrida).
   4. `data_drift: {feature} PSI={x} > {umbral}` — PSI de alguna variable
      supera su umbral: 0.25 (estándar de industria para "cambio
      significativo") para `demand`, `rain_forecast` y
      `temperature_forecast`, y 2.0 para `event_intensity`.
   5. Si nada aplica: `stable: no trigger met` → se conserva el modelo.

   `event_intensity` tiene umbral propio porque es una variable dispersa
   (eventos poco frecuentes): cualquier ventana de 3 días sin eventos
   difiere mucho de una de 14 días que sí tuvo alguno. Con el umbral general
   de 0.25, su PSI (entre 1.16 y 1.18 en las 18 mediciones registradas) disparaba un
   reentreno en **todos** los ciclos: un modelo nuevo por ciclo, la opción
   de "conservar" sin uso real, y ~10 MB más en Supabase Storage por ciclo.
   A ese ritmo, el 1 GB del plan gratuito se llenaba en unos dos días. Con
   2.0 deja de dispararse por ese valor de fondo, pero sigue avisando si hay
   un salto real.

   **Muestra mínima para confiar en el PSI de clima/eventos.** Como el
   contexto va atrasado respecto a las observaciones (ver "Contexto" más
   abajo), la ventana "reciente" de 3 días puede tener casi todo vacío. PSI
   descarta los vacíos por diseño, así que en la práctica termina comparando
   una muestra chica que se corre de un lado a otro en cada corrida —
   confirmado en vivo el 23-sep: `temperature_forecast` pasó de 0.07 a 0.29
   en un par de horas sin que el clima cambiara. Por eso `rain_forecast`,
   `temperature_forecast` y `event_intensity` (una sola fila por instante,
   repetida en las 12 estaciones por el merge, a diferencia de `demand` que sí
   es por estación) necesitan al menos 150 timestamps únicos con dato real en
   la ventana reciente — más o menos la mitad de una ventana completa de 3
   días. Por debajo de eso, `triggered` queda forzado a `false` sin importar
   qué tan alto salga el PSI, aunque el valor se sigue guardando (con
   `details.insufficient_samples=true`) para que quede visible, no oculto.

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

   **Corrección de sesgo en línea**: antes de enviar, se mide cuánto se
   equivocó el modelo en las últimas 4 horas, usando las predicciones ya
   evaluadas contra demanda real de todas las estaciones y horizontes
   (factor = demanda real / predicción). Todas las predicciones del ciclo se
   multiplican por la mitad de esa corrección, limitada entre ×0.8 y ×1.25.
   Por ejemplo, si el modelo viene subestimando un 6%, las predicciones
   suben un 3%. Hacen falta al menos 48 predicciones evaluadas (un ciclo
   completo); si no las hay, no se corrige. El factor se calcula siempre
   sobre la **salida pura del modelo** (`predictions.raw_predicted_demand`),
   no sobre el valor corregido que se envió (`predicted_demand`): así la
   corrección no se acumula sobre sí misma, y la regla de `performance_drift`
   sigue midiendo la calidad real del modelo sin que la corrección la
   enmascare. Cada corrección queda registrada en `drift_measurements`
   (`feature_name='prediction_bias'`, marcada como alerta si el sesgo supera
   el 5%). Se validó con backtest walk-forward antes de adoptarla (ver
   `docs/ml-baselines.md`). Para desactivarla sin tocar código:
   `PULSO_BIAS_CORRECTION=false`.
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
`run_forecast_cycle.py`, en ese orden, en el mismo job, cada 10 minutos
(`*/10 * * * *`, siguiendo la guía oficial del curso). GitHub no garantiza
disparos puntuales de `schedule` — en pruebas, un cron cada 15 minutos
disparó una sola vez en más de dos horas, tanto en repositorio privado como
público — pero a diferencia de espaciar el cron (lo que se intentó primero),
disparar cada 10 minutos da varios intentos por cada ventana de submission de
25 minutos, así que el jitter de GitHub importa menos: alcanza con que *uno*
de esos intentos caiga dentro de la ventana. Como ambos scripts son
idempotentes (el collector no duplica filas; el orquestador crea un
`forecast_run` nuevo por corrida y su decisión no depende de cuántas
corridas hubo antes), correr de más tampoco tiene costo más allá del cómputo.
Configura estos secrets en GitHub:

- `PULSO_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`
- `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`
  (opcionales; omítelos si se usa solo Supabase — ver sección MLflow arriba)

Y estas variables (`vars`, no secrets):

- `PULSO_API_URL`
- `PULSO_SUBMIT_ENABLED` (opcional; `false` para desactivar el envío real)
- `PULSO_BIAS_CORRECTION` (opcional; `false` para enviar la salida pura del
  modelo, sin corrección de sesgo)

GitHub ejecuta workflows programados desde la rama predeterminada: incorpora
este archivo en esa rama antes de esperar ejecuciones automáticas.

## Disparador externo (cron-job.org)

Incluso con `*/10 * * * *`, el `schedule` nativo de GitHub resultó no
disparar solo: en pruebas reales de esta sesión, más de 6 ventanas de 10
minutos pasaron en más de una hora sin ninguna corrida automática, con la
configuración ya verificada como correcta (workflow `active`, repo público,
YAML sin errores). Por eso el disparo real del pipeline hoy depende de un
cron externo, no del `schedule:` del workflow (que se deja igual, como
respaldo gratuito que no estorba si algún día empieza a disparar).

**Mecanismo**: [cron-job.org](https://cron-job.org) (cuenta gratuita) hace un
`POST` cada 10 minutos a la API de GitHub para invocar `workflow_dispatch`
directamente, sin pasar por el scheduler interno de GitHub:

| Campo | Valor |
|---|---|
| URL | `https://api.github.com/repos/Jhaycoltez512/pulso-transmi-mlops/actions/workflows/collector.yml/dispatches` |
| Método | `POST` |
| Headers | `Authorization: Bearer <token>`, `Accept: application/vnd.github+json`, `Content-Type: application/json` |
| Body | `{"ref":"main"}` |
| Frecuencia | cada 10 minutos |

El `<token>` es un *fine-grained personal access token* de GitHub, con acceso
limitado únicamente a este repositorio y permiso "Actions: Read and write"
(mínimo privilegio) — se genera en
`https://github.com/settings/personal-access-tokens/new`, tiene fecha de
expiración obligatoria, y vive únicamente en la configuración del job de
cron-job.org, no en este repositorio. **Hay que renovarlo antes de que
expire**, o el disparador deja de funcionar sin ningún aviso — conviene
poner un recordatorio en la misma fecha de expiración elegida al crearlo.
Si el token se filtra, se revoca desde la misma página de GitHub sin afectar
nada más del proyecto.
