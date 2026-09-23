# Baselines de demanda

Los scripts en `scripts/` consultan las tablas `observations` y `context` de
Supabase. Configura las credenciales locales antes de ejecutarlos:

```dotenv
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SECRET_KEY=<clave-de-servidor>
```

Instala las dependencias y ejecuta las pruebas:

```bash
python -m pip install -e '.[dev,ml]'
python -m pytest -q
```

## Partición temporal

Los modelos ordenan los datos por tiempo y usan tres segmentos sin mezcla de
futuro y pasado:

| Segmento | Tamaño del corte inicial | Uso |
|---|---:|---|
| Entrenamiento | 27.648 filas | Ajustar el modelo |
| Validación | 8.064 filas (7 días) | Comparar decisiones de modelo |
| Prueba | 8.064 filas (7 días) | Estimación final no usada para ajuste |

La métrica se calcula por estación y luego se promedia:

```text
WAPE = sum(abs(real - predicción)) / sum(real)
Accuracy = 100 × max(0, 1 - WAPE)
```

## Modelo con variables y retardos

`scripts/train_baseline.py` entrena un `HistGradientBoostingRegressor` con:

- estación, hora y día de la semana;
- demanda con retardo de 24 horas (`lag_96`) y siete días (`lag_672`);
- pronóstico de lluvia, temperatura y eventos.

```bash
python scripts/train_baseline.py
```

En el corte inicial obtuvo `85.72%` de accuracy en validación y `86.27%` en
prueba. Guarda el modelo y sus métricas en `artifacts/` mediante Joblib.

## Baseline naive semanal

`scripts/train_weekly_naive.py` no ajusta parámetros: para cada estación y
momento pronostica su demanda de 672 períodos antes, equivalentes a siete días.

```bash
python scripts/train_weekly_naive.py
```

En el mismo corte obtuvo `83.02%` en validación y `83.11%` en prueba. Sus
resultados se guardan en `artifacts/weekly_naive.joblib` y
`artifacts/weekly_naive_metrics.json`.

## CatBoost multi-horizonte directo + ensamble con naive semanal

`scripts/train_catboost_direct.py` entrena un `CatBoostRegressor` independiente
por horizonte (15, 30, 45 y 60 minutos, o los horizontes del ciclo activo), en
vez de encadenar predicciones recursivas. Usa `loss_function="MAE"`, alineado
con WAPE porque minimiza la mediana condicional en lugar de la media.

Variables: calendario cíclico del momento objetivo, demanda actual, rezagos
(`lag_1` a `lag_672`) y medias móviles (`rolling_mean_4` a `rolling_mean_672`).

La predicción final no es la salida cruda de CatBoost: se mezcla con el
baseline naive semanal, `0.75 * catboost + 0.25 * naive` (`ENSEMBLE_WEIGHT` en
el script), con retroceso a CatBoost solo si falta la demanda de hace siete
días. El peso se eligió con `scripts/backtest_ensemble.py`, que barre pesos de
0.5 a 1.0 sobre los mismos folds walk-forward: 0.70–0.85 es cercano al óptimo
en los cuatro horizontes, con ganancia sobre CatBoost solo muy por encima del
ruido entre folds (0.0011–0.0047 de WAPE, contra folds con desviación
estándar de 0.0003–0.0014). Afinar el peso por horizonte en vez de usar uno
global ganaría menos de 0.0006 de WAPE adicional — no vale el riesgo de
sobreajustar a solo 3 folds.

```bash
python scripts/train_catboost_direct.py
```

En el mismo corte temporal que los otros baselines:

| Horizonte | WAPE validación | Accuracy validación | WAPE prueba | Accuracy prueba |
|---|---:|---:|---:|---:|
| 15 min | 0.1289 | 87.11% | 0.1289 | 87.11% |
| 30 min | 0.1308 | 86.92% | 0.1308 | 86.92% |
| 45 min | 0.1339 | 86.61% | 0.1324 | 86.76% |
| 60 min | 0.1345 | 86.55% | 0.1367 | 86.33% |

### Experimentos descartados

Un solo split de validación/prueba (7 días, 8.064 filas) tiene ruido del mismo
orden que las mejoras que se buscan, así que dos cambios se evaluaron con
`scripts/backtest_catboost_direct.py` (backtest *walk-forward*: reentrena en
una ventana creciente y evalúa en varias ventanas de prueba de 7 días
sucesivas, no solapadas) antes de decidir:

- **Ponderar el loss por el inverso de la demanda media de cada estación**,
  buscando alinear el entrenamiento con WAPE promediado por estación (en vez
  de MAE global). Empeoró el WAPE en los cuatro horizontes de un solo split y
  se descartó sin necesidad de backtest.
- **Variables de contexto (lluvia, temperatura, eventos) unidas por el
  momento objetivo `target_at`**. En un solo split parecía mejorar el WAPE de
  prueba (~0.0005–0.0014), pero un backtest pareado de 3 folds mostró que el
  signo de la diferencia cambiaba entre folds en los cuatro horizontes, con un
  promedio (±0.0001–0.0002) muy por debajo de la desviación estándar entre
  folds (0.0005–0.0010): la mejora observada era ruido de esa ventana, no
  señal real. Se revirtió.

En ambos casos las estaciones con peor desempeño (`02300`, `10009`) lo son de
forma consistente en los tres modelos del repositorio (naive, GBM y
CatBoost), lo que apunta a menor predictibilidad de su demanda y no a un
problema de escala o de variables faltantes.

Un tercer experimento, más adelante en producción: el modelo activo mostró
una caída lenta y sostenida de accuracy (~87.1% → ~86.9% a lo largo de
~19 reentrenos y ~20 horas). Se investigó antes de tocar nada — demanda total
en la ventana de prueba estable, PSI de las 4 variables monitoreadas sin
cambios que coincidan con el inicio de la caída, y la comparación estación
por estación entre el modelo temprano y el más reciente mostró empeoramiento
repartido en 10 de 12 estaciones (ninguna dominante). Todo apunta a un
corrimiento genuino y gradual en la relación entre variables y demanda
(*concept drift*), consistente con lo que el README del reto advierte sobre
cambios de patrón durante la competencia — no un bug del pipeline.

Con eso confirmado, se probó **ponderar el entrenamiento por qué tan
reciente es cada fila** (`scripts/backtest_recency_weight.py`, decaimiento
exponencial con distintos half-life) para ver si ayudaba a adaptarse más
rápido al corrimiento. No ayudó: con half-life de 30 días el resultado fue
indistinguible de no ponderar, y con half-lives más cortos (14, 7, 3 días)
empeoró de forma consistente en los cuatro horizontes — reducir el historial
efectivo le quita al modelo la comparación semana-a-semana (`lag_672`) que
necesita, sin ganar nada a cambio porque el drift es demasiado lento para que
"lo más reciente" sea mejor maestro que el historial completo. Se descartó;
el modelo sigue como está, confiando en que `performance_drift` siga
disparando reentrenos automáticos para compensar el corrimiento de a poco.

Cuarto experimento, cambiando **lo que el modelo ve** en vez de cómo se
pondera (`scripts/backtest_seasonal_features.py`): rezagos alineados al
momento objetivo de 1, 2 y 3 semanas, su mediana y tendencia, y un ratio de
"nivel" (demanda actual sobre su propio perfil multi-semana, promediado en
la última hora y las últimas 4 horas). Resultado mixto pero concluyente:

- Con **CatBoost solo**, las features ayudan, sobre todo en horizontes
  largos (h60: 0.1310 vs 0.1368 en un fold), y eso con menos datos de
  entrenamiento.
- Pero **sustituyen** a la mezcla con el naive semanal en vez de sumarse: el
  peso de mezcla óptimo de la variante resultó 1.0, es decir, sin naive.
  Comparada con justicia (cada modelo con su propio peso óptimo, mismos
  folds), la variante de 3 semanas quedó peor que producción en h15-h45
  (+0.0011 a +0.0013 de WAPE, mismo signo en ambos folds) y empatada en h60
  (el signo cambia entre folds). Lo más probable es que el warmup de 21 días
  (~40% menos filas de entrenamiento con la historia actual) se coma la
  ganancia.

No se adoptó. Puede valer la pena repetirlo cuando haya más historia
acumulada: el costo del warmup es fijo y pesa menos a medida que crece el
dataset. Tras cuatro intentos (peso por estación, contexto climático, peso
por recencia, features estacionales), el ensamble CatBoost + naive actual
sigue siendo el mejor que tenemos con ~47 días de datos.

Resultados y modelo se guardan en `artifacts/catboost_direct_metrics.json` y
`artifacts/catboost_direct.joblib`. Cada entrenamiento también registra
lineage en Supabase (`model_versions`, `training_runs`, `model_metrics`) con
el `data_version` de la corrida de datos usada y el commit de Git activo al
momento de entrenar.

## Backtest walk-forward de CatBoost

`scripts/backtest_catboost_direct.py` entrena y evalúa en varias ventanas de
prueba sucesivas de 7 días (ventana de entrenamiento creciente) en lugar de un
solo split, para poder distinguir una mejora real de ruido entre ventanas.
Reutiliza `FEATURES`, `model` y las funciones de features de
`train_catboost_direct.py`; no duplica lógica.

```bash
python scripts/backtest_catboost_direct.py
```

Guarda el detalle por fold en `artifacts/catboost_direct_backtest.json` e
imprime el WAPE promedio y su desviación estándar entre folds por horizonte.
Con la historia disponible (~46 días) obtiene 3 folds por horizonte; ese
número crece a medida que el collector acumula más datos. Úsalo para juzgar
cualquier cambio de features o hiperparámetros antes de adoptarlo: si la
diferencia observada es menor que la desviación estándar entre folds, no hay
evidencia suficiente de que el cambio ayude.

`scripts/backtest_ensemble.py` hace lo mismo pero barre un peso de mezcla
CatBoost/naive sobre los mismos folds, para elegir el peso con evidencia en
vez de un solo split. Guarda el detalle en `artifacts/ensemble_backtest.json`.

```bash
python scripts/backtest_ensemble.py
```

## Vista previa y validación de submission

El script siguiente consulta el ciclo activo, construye las 12 predicciones del
baseline semanal y valida localmente esquema, IDs de estación, objetivos,
timestamps, valores y cantidad de predicciones. **No realiza ningún POST**.

```bash
python scripts/generate_weekly_submission_preview.py
```

Los archivos `artifacts/submission_weekly_naive_preview.json` y
`artifacts/submission_weekly_naive_validation.json` quedan fuera de Git junto
con los modelos, métricas y recibos de submissions.

La primera submission de práctica se aceptó para el ciclo
`cyc_practice_20260918`: 12 predicciones, estado `accepted`, intento oficial
`1`. El recibo permanece únicamente como artefacto local para no versionar datos
de ejecución ni identificadores operativos.
