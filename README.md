# Pulso TransMi — SDK para estudiantes

Starter kit oficial del reto MLOps **Pulso TransMi**. Incluye un cliente Python,
ejemplos reproducibles y una plantilla de GitHub Actions para construir un
pipeline que descargue datos, entrene, monitoree y posteriormente envíe
predicciones.

> **Disponible públicamente:** la API de lectura está en
> `https://pulso-transmi.72-60-245-2.sslip.io` y su documentación interactiva en
> [`/docs`](https://pulso-transmi.72-60-245-2.sslip.io/docs).

## El reto

Se pronostica demanda sintética cada 15 minutos para 12 estaciones reales de
TransMilenio. El sistema liberará observaciones con el tiempo y cambiará algunos
patrones durante la competencia. Un modelo entrenado una sola vez puede perder
desempeño: el objetivo es operar un pipeline capaz de medir, decidir y
reentrenar.

La demanda, clima y eventos son sintéticos. Los nombres y coordenadas de las
estaciones provienen de datos oficiales de TransMilenio.

## Inicio rápido

Requiere Python 3.11 o superior.

```bash
git clone https://github.com/uexternadojz/pulso-transmi-sdk.git
cd pulso-transmi-sdk
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml]'
cp .env.example .env
python examples/01_download.py
python examples/02_naive_baseline.py
```

En Windows PowerShell, la activación es `.venv\Scripts\Activate.ps1`.

## Implementación del equipo

La rama `main` contiene el pipeline en operación.

**Datos y trazabilidad**

- Esquema y carga inicial en Supabase.
- EDA reproducible en `analysis/eda.py`, con resultados en `eda_results/`.
- Collector incremental cada 10 minutos en GitHub Actions. Lo dispara un cron
  externo porque el `schedule` nativo de GitHub no disparaba solo; ver
  [docs/collector-and-lineage.md](docs/collector-and-lineage.md).
- Cada reentreno versiona juntos datos y modelo en Supabase y en MLflow
  (Model Registry con alias `champion` y dataset versionado).

**Modelo en producción**

- CatBoost multi-horizonte directo: un modelo por horizonte (15, 30, 45 y 60 min).
- Demanda y objetivo **normalizados por el nivel reciente de cada estación**
  (media de las últimas 4 h). Si una estación sube o baja de golpe, el modelo
  sigue funcionando sin salirse de su rango.
- **Solo memoria corta**: demanda actual, rezagos de hasta 2 h y medias
  móviles de 1 h y 3 h, más calendario y estación.
- Se entrena con toda la historia disponible. Las métricas y umbrales de
  alerta salen de las ventanas de validación y prueba.

Reemplazó al ensamble CatBoost + naive semanal cuando la competencia empezó a
mover demanda entre estaciones (desde el 13-sep simulado: 05100 cayó a ×0.18,
05000 y 02300 subieron ×2.5–3.2). En un backtest con esos datos reales pasa
de 74–78% a 84–88% de accuracy según el horizonte. El detalle y los
experimentos descartados están en [docs/ml-baselines.md](docs/ml-baselines.md).

Como referencia se conservan dos baselines: `HistGradientBoostingRegressor`
con retardos diario y semanal, y el naive semanal (demanda de la misma
estación siete días antes).

**Ciclo operativo** (`scripts/run_forecast_cycle.py`)

En cada ciclo abierto:

1. Evalúa las predicciones pasadas del modelo activo contra la demanda real.
2. Mide drift:
   - PSI de demanda y contexto;
   - **drift por estación**: cambio de la relación "últimas 24 h ÷ mismas 24 h
     de la semana anterior" desde que se entrenó el modelo.
3. Decide con una regla explícita si conserva o reentrena. Reentrena por
   antigüedad, por error sobre el umbral o por drift; el motivo queda
   registrado.
4. Predice los 4 horizontes de las 12 estaciones y envía la submission.
5. Registra éxito o error de cada paso.

Los modelos se guardan en Supabase Storage con poda automática.

**Dashboard y herramientas**

- Dashboard en `dashboard/` (Vite + React, en Vercel): estado del pipeline,
  modelo activo, errores, drift por estación y leaderboard; ver
  [docs/dashboard.md](docs/dashboard.md).
- Backtests walk-forward, entre ellos `scripts/backtest_production_shift.py`
  sobre el periodo real de cambios.
- Vista previa y validación local de la submission, sin enviarla.

Consulta [la guía de Supabase](docs/supabase.md), [la guía de baselines](docs/ml-baselines.md)
y [la guía del collector y lineage](docs/collector-and-lineage.md) para los
comandos, el particionado temporal y las métricas.

## Uso del SDK

```python
from pulso_transmi import PulsoTransmiClient

client = PulsoTransmiClient()

print(client.meta())
stations = client.stations()
observations = client.observations_dataframe(station_id="07107")
context = client.context_dataframe()

print(stations.head())
print(observations.tail())
```

El SDK recorre automáticamente todas las páginas. Si prefieres controlar cada
página, usa `client.observations_page(...)` y conserva `next_cursor` exactamente
como lo entrega la API.

## Datos iniciales

| Recurso | Tamaño |
|---|---:|
| Estaciones | 12 |
| Frecuencia | 15 minutos |
| Historia | 45 días |
| Periodos por estación | 4.320 |
| Observaciones | 51.840 |

Para evaluación local, usa una división temporal: por ejemplo, primeros 38 días
para entrenamiento y últimos 7 para validación. Una partición aleatoria mezcla
futuro y pasado y genera métricas engañosas.

## API de lectura `0.2.0`

| Método | Ruta | Uso |
|---|---|---|
| `GET` | `/health` | Estado básico |
| `GET` | `/v1/meta` | Versión, rango, hashes y enlaces |
| `GET` | `/v1/stations` | Catálogo geográfico |
| `GET` | `/v1/observations` | Demanda paginada |
| `GET` | `/v1/context` | Clima y eventos |
| `GET` | `/v1/downloads/{filename}` | Descarga completa |

Swagger está disponible en `/docs`. Consulta [docs/api.md](docs/api.md) para
filtros, paginación y errores.

## Estructura esperada del proyecto estudiantil

```text
mi-pulso-transmi/
├── src/
│   ├── ingest.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── monitor.py
├── tests/
├── artifacts/
├── requirements.txt o pyproject.toml
└── .github/workflows/pipeline.yml
```

El repositorio de cada equipo debe dejar trazabilidad de:

- cutoff de datos usado;
- versión o commit del código;
- features y modelo entrenado;
- métricas de validación temporal;
- momento y razón de cada reentrenamiento;
- errores de ingesta o inferencia.

## GitHub Actions

[`templates/pipeline.yml`](templates/pipeline.yml) es una plantilla manual. Cópiala
a `.github/workflows/pipeline.yml` dentro del repositorio de tu equipo. Cuando se
habilite la competencia, agrega el API key como secret y luego activa el horario
indicado por el profesor.

Nunca escribas API keys, contraseñas de Supabase ni tokens dentro del código.

## Supabase y Vercel

Supabase es opcional para persistir ejecuciones, métricas, predicciones y estado
del modelo. Vercel es opcional y corresponde al bono de visualización. Ninguna de
las dos plataformas reemplaza el repositorio ni GitHub Actions.

Consulta [docs/student-project.md](docs/student-project.md) para el flujo completo
y los entregables.

## Métrica

La referencia actual es:

```text
WAPE = sum(abs(real - predicción)) / sum(real)
Accuracy = 100 × max(0, 1 - WAPE)
```

La métrica se calcula por estación y luego se promedia. El contrato definitivo
de submissions y leaderboard se publicará antes de iniciar la ventana competitiva.

## Desarrollo del SDK

```bash
python -m pip install -e '.[dev,ml]'
pytest -q
```

Este repositorio es público para estudiantes. No debe contener ground truth
futuro, semillas, configuración privada del escenario ni parámetros de drift.
