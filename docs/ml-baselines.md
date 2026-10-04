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
(`lag_1` a `lag_672`) y medias móviles (`rolling_mean_4` a `rolling_mean_672`). Desde el
30-sep todas las variables de demanda y el objetivo van normalizados por el nivel reciente
de la estación (ver "Adoptado: modelo normalizado por nivel" más abajo).

Hasta el 30-sep la predicción final no era la salida cruda de CatBoost: se mezclaba con el
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

### Adoptado: corrección de sesgo en línea

Quinto experimento, y el primero que se adoptó
(`scripts/backtest_bias_correction.py`). En vez de cambiar el modelo, se
corrige su salida: en cada momento de predicción se mide el cociente demanda
real / predicción de las predicciones de las últimas horas (solo las que ya
tienen demanda real conocida en ese momento) y se escala la predicción
siguiente. Se probaron 12 configuraciones sobre las mismas predicciones de
producción: alcance por estación o global, ventana de 4/12/24 horas, y
corrección al 50% o al 100%.

| Horizonte | Producción | Global, 4h, 50% | Delta por fold |
|---|---:|---:|---|
| 15 min | 0.1294 | 0.1282 | −0.0021, −0.0003, −0.0012 |
| 30 min | 0.1318 | 0.1305 | −0.0025, 0.0000, −0.0014 |
| 45 min | 0.1341 | 0.1329 | −0.0022, +0.0001, −0.0013 |
| 60 min | 0.1361 | 0.1348 | −0.0023, 0.0000, −0.0014 |

Por qué se considera confiable, a pesar de ser una ganancia chica (~0.0012
de WAPE, ~+0.12 puntos de accuracy):

- **La misma configuración gana en los 4 horizontes**, cada uno evaluado por
  separado. Con 12 configuraciones probadas, eso descarta que la ganadora
  sea suerte de la búsqueda.
- **Nunca empeora**: 9 de 12 combinaciones fold-horizonte mejoran y 3 quedan
  empatadas (máximo +0.0001). En los experimentos anteriores la diferencia
  cambiaba de signo entre folds.
- **Las configuraciones vecinas también mejoran** (por estación a 4h, por
  estación a 12h): es una zona estable, no un pico aislado.
- **Es coherente con el drift encontrado**: ventana corta y corrección
  parcial le ganan a las largas o agresivas, y la versión global le gana a
  la por estación porque sumar las 12 estaciones reduce el ruido.

Cómo funciona en producción: ver "Corrección de sesgo en línea" en
`docs/collector-and-lineage.md`. Al adoptarla, el sesgo real medido sobre
las últimas 4 horas era +6.7% (el modelo venía subestimando), lo que llevó
a subir las predicciones un 3.4%.

Resultados y modelo se guardan en `artifacts/catboost_direct_metrics.json` y
`artifacts/catboost_direct.joblib`. Cada entrenamiento también registra
lineage en Supabase (`model_versions`, `training_runs`, `model_metrics`) con
el `data_version` de la corrida de datos usada y el commit de Git activo al
momento de entrenar.

### Adoptado: modelo normalizado por nivel, memoria corta y reentreno con toda la historia

**Qué pasó.** Desde el 13-sep (tiempo simulado) la competencia empezó a mover demanda entre
estaciones, de forma progresiva: frente a la semana anterior, 05100 cayó a 0.72 → 0.43 →
0.18, 05000 subió ×2.5 → ×3.2, 02300 ×2.0 → ×2.5, 03000 bajó a 0.41 y 07111 quedó ×1.4.
Las otras 7 estaciones siguieron estables. La accuracy de producción pasó de ~85% a 77–80%
(13–15 sep) y a 65–70% (16–17 sep); 05100 llegó a −50%. El PSI de `demand` no lo vio
(0.01–0.04, porque mezcla todas las estaciones y los cambios se compensan) y reentrenar cada
ciclo no ayudaba.

**Por qué no se recuperaba:**

1. **El modelo desplegado nunca veía los últimos 14 días.** `train_models` ajustaba con el
   split de entrenamiento y usaba ese modelo en producción. Ahora las métricas y los
   umbrales de `performance_drift` salen de validación/prueba como antes, pero el modelo
   desplegado se reajusta con toda la historia.
2. **Un árbol no extrapola** fuera del rango de demanda que vio al entrenar.
3. **Los rezagos de 1 día / 1 semana y el naive semanal anclan cada estación a un patrón
   que ya no existe.**

**Qué se adoptó** (`scripts/train_catboost_direct.py`):

- Features y objetivo normalizados por el nivel de cada estación (media de las últimas 4 h,
  `LEVEL_WINDOW=16`); la predicción se multiplica de vuelta. Ajuste con
  `sample_weight=level`, así el MAE sobre el cociente equivale al MAE en unidades de
  demanda.
- Solo memoria corta (`MODEL_DEMAND_FEATURES`: demanda actual, `lag_1/2/4/8`,
  `rolling_mean_4/12`) más calendario y estación.
- Sin mezcla con el naive semanal (`ENSEMBLE_WEIGHT=1.0`).
- Corrección de sesgo en línea desactivada por defecto (se sigue midiendo y registrando).

**Validación con datos reales** (`scripts/backtest_production_shift.py`): repite el periodo
12–18 sep con los datos de Supabase. Cada 12 h se entrena con todo lo conocido en ese
momento y se predice las 12 h siguientes, igual que en producción. Se compararon 52
combinaciones (modelo crudo / por nivel / por nivel con memoria corta × sin naive / naive /
naive ajustado con cotas 0.5–2 o 0.1–5 × sin corrección / corrección global / por
estación al 50% o 100%). La misma combinación ganó en los cuatro horizontes:

| Horizonte | Producción anterior | Nivel + naive ajustado + corrección global | **Adoptado** |
|---|---:|---:|---:|
| 15 min | 77.86% | 85.60% | **87.55%** |
| 30 min | 76.74% | 84.70% | **86.60%** |
| 45 min | 75.57% | 83.57% | **85.33%** |
| 60 min | 74.14% | 82.64% | **84.08%** |

Por estación (h15), el modelo adoptado queda entre 86.4% y 88.5% en las 12, incluidas las
que cambiaron (05100: 44% → 88%, 05000: 65% → 88%, 02300: 70% → 89%). En los días
16–17 sep llega a ~91% en h15 y ~87.5% en h60.

**Costo en régimen estable.** Con el dataset inicial (sin cambios de patrón) el modelo
adoptado da 86.1/85.4/84.5/83.8% en validación (h15–h60), contra 87.1/86.9/86.6/86.6% del
modelo anterior: pierde 1–3 puntos cuando nada cambia, a cambio de 10 puntos o más cuando
cambia. Dado que el reto anuncia cambios de patrón, se prioriza la robustez. Los umbrales
de `performance_drift` salen de esa validación (WAPE ×1.15 ≈ 0.16–0.19).

**Advertencia.** Se eligió la mejor de 52 combinaciones sobre el mismo periodo, así que la
cifra exacta es algo optimista. La ventaja sobre la producción anterior (8–10 puntos) es
mucho mayor que ese sesgo de selección, y la misma combinación ganó en los cuatro
horizontes por separado.

**Compatibilidad.** Los bundles llevan `model_format="level-ratio-v2"`. Si el modelo activo
en Storage tiene otro formato, `run_forecast_cycle.py` no lo reutiliza y reentrena. Los
backtests de pesos, sesgo y walk-forward usan el modelo actual (`fit_model` /
`predict_demand`); los de recencia y features estacionales conservan `RAW_FEATURES` porque
documentan experimentos sobre el modelo anterior.

### Descartado: ventana de entrenamiento, recencia y memoria de 1 día (30-sep)

Con el drift inyectado a mano por el profesor, se probó si convenía entrenar con menos
historia, dar más peso a lo reciente o recuperar la memoria de 1 día
(`scripts/backtest_training_window.py`, mismo walk-forward cada 12 h sobre 12–18 sep con
datos reales, mismo modelo normalizado por nivel). Accuracy media por estación, todo el
periodo:

| Variante | h15 | h30 | h45 | h60 |
|---|---:|---:|---:|---:|
| **Producción (toda la historia)** | **87.19** | **86.00** | 84.68 | 83.36 |
| + `lag_96` y `rolling_mean_96` | 87.14 | 85.91 | **84.77** | **83.65** |
| Recencia, vida media 7 días | 87.06 | 85.80 | 84.30 | 82.93 |
| Últimos 14 días | 86.94 | 85.54 | 84.03 | 82.45 |
| Recencia, vida media 3 días | 86.98 | 85.39 | 83.97 | 82.41 |
| Últimos 7 días | 86.66 | 84.98 | 83.22 | 81.35 |

- **Recortar la historia o ponderar por recencia empeora en los 4 horizontes**, más cuanto
  más corta la ventana. El modelo normalizado por nivel aprovecha la historia vieja (la
  forma de la demanda) sin anclarse a su nivel, así que conviene seguir entrenando con todo.
- **La memoria de 1 día queda empatada** (±0.3 puntos). Ayuda en las estaciones estables y
  en las últimas 24 h (+0.4 a +0.7), pero empeora 1.5–2.5 puntos el día de los cambios de
  nivel (16-sep) y en las estaciones que cambiaron (02300, 05000, 05100). No se adoptó.
- **Ninguna variante resuelve el evento del 18-sep 05:00–06:00 UTC** (picos ×3–5 en 09122,
  03000, 06111 y 07105, caídas en 10009 y 07107): todas quedan en 36–64% en esa hora. Un
  evento nuevo no se puede anticipar desde la demanda pasada; haría falta el contexto de
  eventos (`event_intensity`), que la API no publica desde el 9-sep.

### Adoptado: expertos periódicos para el drift de 4 h (1-oct)

**El patrón.** Desde el 18-sep a las 05:00 UTC, la demanda inyectada no es un pulso aislado:
es una oscilación con período de exactamente 4 h. Frente al día comparable, los picos llegan a
×5–11 y los valles a ×0.1–0.2. Las estaciones forman cuatro grupos, desfasados 1 h entre sí:

| Grupo | Estaciones | Valles (UTC) |
|---|---|---|
| A | 02300, 06000, 07111 | 07, 11, 15 h |
| B | 05000, 07105, 09122 | 05, 09, 13 h |
| C | 05100, 07107, 10009 | 06, 10, 14 h |
| D | 03000, 06111, 09000 | 08, 12 h |

**El modelo no puede aprenderlo:**

- su memoria va de 15 min a 2 h;
- normaliza por el nivel de las últimas 4 h;
- el evento son ~10 h de datos frente a 50 días de historia normal;
- la API no publica contexto después del 09-sep, así que no hay `event_intensity` ni ninguna
  otra señal del evento.

**El remedio.** "Copiar la serie de hace 4 h" acierta ~90% durante este régimen y ~36% en un
día normal. Son justo los dos casos en que el modelo hace lo contrario. Por eso
`forecast_adjustments.py` suma al ensamble los expertos `lag_2h` … `lag_6h` (la demanda P horas
antes del objetivo), y los pesos por error reciente eligen el adecuado.

**Backtest** (walk-forward, 12-sep → 18-sep 15:00). Accuracy media por estación (%), promedio
de h15–h60:

| Variante | Todo | Sin evento | Evento | Evento 05–09h | Evento 09h+ |
|---|---|---|---|---|---|
| Modelo solo | 80.67 | 85.82 | 26.2 | 38.5 | 22.5 |
| Ensamble viejo (6 h, p3) | 82.87 | 86.06 | 48.2 | 48.7 | 47.7 |
| Modelo + periódicos (4 h, p6) | 84.34 | 85.77 | 68.5 | 43.7 | 83.3 |
| **Todos los expertos, global, 4 h, p6** | **84.77** | **86.04** | **70.9** | **49.3** | **83.7** |
| Copia de hace 4 h (sola) | 15.8 | 13.5 | 64.6 | 36.2 | 89.0 |

**Repetición sobre las predicciones reales del modelo en producción**
(`production_adjust` con el historial real). El ensamble viejo reproduce exactamente lo que se
envió en vivo (55.5 y 58.6 a las 13:00 y 14:00), lo que valida la repetición.

| Corte | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | Media 09–14 |
|---|---|---|---|---|---|---|---|---|---|---|
| Modelo | 40.0 | 50.8 | 44.2 | 45.0 | 41.7 | 65.9 | 58.1 | 60.8 | 60.5 | 55.3 |
| Ensamble viejo | 42.7 | 51.9 | 53.1 | 50.1 | 47.8 | 60.0 | 56.6 | 55.5 | 58.6 | 54.8 |
| **Nuevo** | 42.3 | 57.9 | 58.4 | 65.7 | 70.7 | 88.0 | 86.8 | 90.2 | 90.5 | **82.0** |

**Adoptado:** `DEFAULT_ADJUSTMENT="ensemble"` con todos los expertos, pesos globales por
horizonte, ventana de 4 h y potencia 6.

- En horas normales el modelo conserva ~todo el peso.
- Tras un período completo del régimen (09h+), `lag_4h` pesa ~0.8.

**Límites:**

- Las primeras ~4 h de un régimen nuevo no se pueden copiar: aún no hay un período completo
  observado.
- Un drift sin periodicidad no se beneficia de los expertos periódicos.
- Solo cubre períodos de 2 a 6 h.

### Adoptado: promedios de período además de las copias (2-oct)

**Por qué hacía falta.** Con el ensamble de copias, el 19-sep obtuvimos ~91% por ciclo,
mientras los mejores rivales sacaban 92–93% (tabla `leaderboard_snapshots`). Copiar un solo
período arrastra su ruido. `scripts/backtest_periodic.py` compara refinamientos de la copia de
4 h en los 19 ciclos reales del régimen completo (19-sep, 05–23 h), con la métrica del
leaderboard: 1 − WAPE por ciclo.

| Variante | Por ciclo |
|---|---|
| **Media de los últimos 6 períodos** | **92.98** |
| Media ponderada por recencia (d=0.7) | 92.93 |
| Media de 4 períodos | 92.85 |
| Ridge sobre 6 períodos | 92.75 |
| Ajustes por nivel o por desvío actual | 89.0–92.8 |
| Copia de hace 4 h / lo enviado | 91.25 / 91.30 |

**Cambio en el ensamble.** Cada experto periódico `per_{P}h` promedia su período sobre las
últimas 24 h (6 copias para P = 4 h). Se suma a las copias simples `lag_{P}h`; no las
reemplaza.

**Repetición sobre las predicciones reales** (43 ciclos del régimen, con el modelo en
producción como experto "modelo"):

| Conjunto de expertos | Primeras 24 h del régimen | Desde 24 h de régimen | Todo |
|---|---|---|---|
| Producción (copias) | 82.44 | 91.30 | 86.35 |
| Solo promedios | 66.94 | 92.91 | 78.41 |
| **Copias + promedios** | 81.57 | **92.80** | **86.54** |

**Walk-forward 12–20 sep** (accuracy media por estación, promedio de h15–h60):

| Conjunto de expertos | Días normales | 19-sep |
|---|---|---|
| Copias | 86.04 | 90.15 |
| Copias + promedios | 86.03 | 92.03 |

**Lectura de los resultados:**

- Los promedios solos fallan al inicio de un régimen, porque mezclan copias de antes de que
  empezara. Por eso van junto a las copias.
- El ensamble elige la copia al principio y el promedio cuando ya hay historia.
- En días normales no cambia nada.

### Descartado: ponderaciones del promedio de período (2-oct)

`scripts/backtest_periodic_weighted.py` compara refinamientos del promedio de 6 períodos en 21
ciclos reales del régimen (19-sep 05:00 → 20-sep 01:00 UTC), con la métrica 1 − WAPE por ciclo.
Todas las variantes usan solo copias de al menos 4 h de antigüedad, así que no usan
información futura.

| Variante | Por ciclo | Gana a media 6 |
|---|---|---|
| Promedio ponderado por recencia (0.85) | 93.03 | 11/21 |
| **Media 6 (producción)** | **93.01** | — |
| Ponderación elegida por estación según su error reciente | 92.94 | 8/21 |
| Media 6 suavizada en el tiempo | 92.95 | 10/21 |
| Media recortada / mediana | 92.81–92.89 | 4–6/21 |
| Corrección de sesgo por estación | 92.85 | 6/21 |
| Pesos por estación (ridge) | 89.4–91.1 | 0–1/21 |
| Media 8 / media 12 | 90.64 / 78.48 | 5 / 0 |

**Resultado:** ninguna supera a la media de 6 más allá del ruido.

- Con 8 o 12 períodos el promedio mezcla datos de antes del régimen.
- Los pesos por estación sobreajustan con 24 h de datos.

~93% por ciclo parece el techo del enfoque periódico, y coincide con el promedio de 7 ciclos de
los líderes de la competencia (93.0–93.2).

### Probado y revertido: ensamble adaptativo de expertos después del modelo (1-oct)

Desde el 18-sep a las 05:00 UTC la competencia inyecta eventos: todo el sistema ×2.6–3.4
frente al día anterior, y grupos de estaciones que pulsan juntos. El modelo normalizado por
nivel sigue el nivel reciente de cada estación. Eso funciona bien ante cambios sostenidos,
pero en un evento aplica la rampa habitual de la mañana sobre un nivel ya inflado
(07111: nivel 3759, predijo 4690 → 6718, real 2775 → 1751). Ningún predictor gana en todas
partes:

- el día comparable acierta en horas normales y falla en el evento;
- la persistencia hace lo contrario;
- el modelo gana ante cambios de nivel sostenidos.

`scripts/forecast_adjustments.py` reúne los ajustes posteriores al modelo:

- **Día comparable**: el día más reciente del mismo tipo. Martes a viernes usan el día
  anterior, el lunes usa el viernes y los fines de semana usan el mismo día de la semana
  anterior.
- **Mezcla suave**: lleva la predicción hacia el día comparable con peso
  0.25 × horizonte / 60.
- **Tope de crecimiento**: si la estación va más de N veces por encima de su día comparable,
  no se predice por encima de máx(nivel actual, día comparable en el objetivo).
- **Ensamble adaptativo**: combina modelo, persistencia y día comparable con pesos
  proporcionales a MAE^-p. El MAE de cada experto se calcula sobre los objetivos ya
  observados en una ventana reciente, sin mirar el futuro.

`scripts/backtest_adjustments.py` los compara en walk-forward sobre datos reales: reentrena
cada 12 h y evalúa orígenes horarios, como los ciclos. El periodo va del 12-sep al
18-sep 12:00 UTC. El tramo "evento" son las ~6 h desde las 05:00 UTC del 18-sep.

Precisión media por estación (%), promedio de h15/h30/h45/h60:

| Variante | Todo | Sin evento | Evento | h15 | h30 | h45 | h60 |
|---|---|---|---|---|---|---|---|
| Producción (modelo solo) | 81.90 | 85.82 | 26.9 | 85.28 | 83.03 | 80.84 | 78.43 |
| Mezcla 0.25 | 82.66 | 86.13 | 32.2 | 85.61 | 83.59 | 81.71 | 79.71 |
| Mezcla + tope ×1.5 | 82.90 | 85.63 | 42.2 | 85.48 | 83.63 | 82.18 | 80.29 |
| Mezcla + tope ×2.5 | 83.40 | 86.10 | 42.9 | 85.85 | 84.06 | 82.77 | 80.93 |
| Ensamble por estación+h, 24 h, p2 | 83.06 | 85.91 | 40.5 | 85.91 | 83.84 | 82.08 | 80.42 |
| Ensamble global por h, 6 h, p2 | 83.31 | 85.80 | 46.3 | 86.31 | 83.96 | 82.43 | 80.55 |
| **Ensamble global por h, 6 h, p3** | **83.60** | **86.06** | **47.0** | 86.43 | 84.20 | 82.78 | 80.98 |
| Ensamble sobre mezcla+tope ×2, global 6 h | 83.00 | 85.29 | 48.8 | 86.18 | 83.57 | 82.02 | 80.22 |
| Persistencia | 73.9 | 75.6 | 48.4 | 83.55 | 77.78 | 70.75 | 63.59 |
| Día comparable | 72.7 | 76.1 | 26.5 | 72.58 | 72.27 | 72.66 | 73.38 |

**Adoptado**: el ensamble global por horizonte con ventana de 6 h y potencia 3
(`DEFAULT_ADJUSTMENT="ensemble"` en `run_forecast_cycle.py`). En horas normales mejora al
modelo en los cuatro horizontes, y en el evento lo mejora mucho (+20 puntos de media).

- Los pesos se agrupan por horizonte entre todas las estaciones. Así cada peso tiene unas
  72 muestras en 6 h, y una estación sin historial propio aprende de las demás.
- Una ventana más corta reacciona antes al evento. La potencia 3 deja que lidere el experto
  que viene acertando.
- En la variante global de 6 h cuyos pesos se registraron (p2 sobre mezcla+tope), el modelo
  pesa ~0.55 en horas normales, y el resto se reparte entre persistencia y día comparable.
  Durante el evento el modelo baja a ~0.35 y suben la persistencia (a h15) o el día
  comparable (a h60).

Sin historial (primer ciclo, o sin predicciones evaluadas en 6 h) el ensamble envía el modelo
solo. `PULSO_ADJUSTMENT=blend_cap` usa la mezcla + tope ×2.5 (segunda mejor y sin estado), y
`PULSO_ADJUSTMENT=none` desactiva cualquier ajuste. `raw_predicted_demand` sigue guardando la
salida pura del modelo: es lo que miden `performance_drift` y el factor de sesgo, y es el
historial del experto "modelo".

Caveat: el tramo de evento es un solo episodio de ~6 h, así que la ganancia en evento tiene
mucha incertidumbre. Lo robusto es que en ~6 días de horas normales el ensamble no empeora a
ningún horizonte.

**Revertido el mismo 1-oct** (`DEFAULT_ADJUSTMENT="none"`). En vivo, el ensamble perdió
contra el modelo solo en sus dos primeros ciclos evaluados:

| Corte (simulado) | Ensamble | Modelo solo | Persistencia | Día comparable |
|---|---|---|---|---|
| 18-sep 13:00 | 55.5 | 60.8 | 45.1 | 37.8 |
| 18-sep 14:00 | 58.6 | 60.5 | 47.4 | 38.7 |

En los 8 cruces de horizonte y ciclo quedó igual o peor que el modelo. Después de las 13:00
las estaciones oscilan con fuerza (06000 cae de 1496 a 263 en una hora, 05000 sube de 786 a
1475), un régimen que el periodo del backtest no contenía. Los pesos de las 6 h previas
empujaban hacia la persistencia justo cuando el modelo tenía la dirección correcta. El código
sigue disponible con `PULSO_ADJUSTMENT=ensemble` o `blend_cap`.

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

## Fin del régimen 4h y guarda de ruptura (2026-10-04)

Desde 2026-09-20 12:15 (virtual), a la vez que el stream pasó a schema v2, la oscilación de 4 h
dejó de repetirse: copiar cualquier periodo (2–6 h) pasó de ~7 % a >25 % de WAPE. El ensamble,
con pesos de las últimas 4 h (casi todo régimen viejo), siguió enviando las copias periódicas:
34,0 % y 10,8 % en los ciclos 12:00 y 13:00, cuando la persistencia daba 74,5 % y 68,9 %.
Acortar la ventana no basta (1 h: 38,5 % de media en esos dos ciclos).

`forecast_adjustments.periodic_break` mide en las observaciones el WAPE de la mejor copia
periódica en la última hora y en las 24 h anteriores; si la reciente es > 0,25 y > 2× la de
referencia, ese ciclo sólo compiten modelo, persistencia y día comparable, pesados sobre la
última hora. Repetición sobre 29 ciclos reales (`scripts/backtest_regime_switch.py`):

| variante | régimen 4h (27) | tras el corte (2) | todo (29) |
|---|---|---|---|
| producción 4h p6 | 93,05 | 22,39 | 88,18 |
| guarda 4h p6 | 93,05 (nunca se activa) | 51,41 (13:00: 68,8) | 90,18 |
| ventana 1h p6 | 92,72 | 38,51 | 88,98 |

El ciclo 12:00 no se podía salvar (la ruptura empezó justo después de su corte). Activada en
producción (`ENSEMBLE_BREAK_GUARD = True`).

## Experto de tendencia amortiguada (2026-10-04)

Tras la ruptura cada estación deriva suavemente durante horas (03000: 42→~1000; 07111: 2108→211
y vuelve a subir) y ninguna copia periódica (1,5–8 h), ni el día o la semana anteriores, sirve.
Nuevo experto `trend` = último valor + 0,5 × pendiente de la última hora × horizonte (≥ 0).
En 814 pronósticos densos post-ruptura: persistencia 76,1 %, tendencia ×0,5 78,0 %, ×0,3 77,9 %,
×1,0 74,2 %. Repetición del ensamble sobre 40 ciclos reales (`scripts/backtest_regime_switch.py`):

| ciclo | producción (guarda) | + trend 0,5 | trend solo |
|---|---|---|---|
| 13:00 | 68,8 | 69,7 | 79,0 |
| 14:00 | 81,4 | 81,1 | 80,3 |
| 15:00 | 75,3 | 75,8 | 75,9 |
| 16:00 | 78,9 | 81,0 | 81,8 |
| 17:00 (24 reales) | 82,2 | 84,8 | 86,2 |
| régimen 4h (31 ciclos) | 92,98 | 92,97 | — |

Se añade al ensamble (+1,2 puntos de media en los 5 ciclos con la guarda activa, neutral en el
régimen 4h). La tendencia sola fue aún mejor después de la ruptura (80,6 vs 78,5), pero son 5
ciclos; se revisará con más datos antes de darle más peso.

## Régimen de 8 h: periodos largos y copia ajustada (2026-10-04)

Con ~14 h del régimen nuevo, el error de copiar el valor P horas antes muestra un periodo exacto de
**8 h** (copiar 7/8/9 h: 63,2 / 79,0 / 60,2 %; 4–6 h: 23–42 %). Nuestros expertos periódicos sólo
cubrían 2–6 h y la guarda de ruptura tampoco lo veía. Los rivales que lo detectaron hacían 81–88 %
por ciclo mientras nosotros 73–82 %.

Cambios: periodos 7, 8, 9, 10 y 12 h (`lag_*`, `per_*`), un experto `shift_Ph` = copia de hace P
+ 0,5 × (ahora − valor P antes de ahora), la guarda evalúa los periodos que el ensamble puede copiar,
y ventana de pesos de 2 h. Pronósticos densos (1015): persistencia 77,9; tendencia 79,6; copia 8 h
85,4; copia 8 h ajustada 86,7. Repetición sobre 44 ciclos reales:

| variante | régimen 4h (29) | ciclos con copia de 8 h (21:00–02:00, 6) |
|---|---|---|
| producción anterior | 93,06 | 78,4 |
| + 7–12 h + shift, ventana 4 h | 92,90 | 84,5 |
| **+ 7–12 h + shift, ventana 2 h** | 92,80 | **87,4** (últimos 3: 91,0 / 90,5 / 88,3) |
| copia 8 h sola | 90,98 | 88,5 |
