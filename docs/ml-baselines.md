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
