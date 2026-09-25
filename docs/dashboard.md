# Dashboard (bono)

Bono opcional de `docs/student-project.md`. Sitio estático en `dashboard/`
(Vite + React + TypeScript), pensado para desplegarse en Vercel.

## Por qué esta arquitectura

La guía advierte explícitamente: nunca expongas claves privadas de Supabase
en el navegador. Antes de tocar código, verifiqué el estado real de RLS en
la base y **no había ninguna política definida** — es decir, con la clave
pública el navegador no veía absolutamente nada. Eso, junto con la
advertencia, apunta a la arquitectura esperada: un sitio estático que use la
`SUPABASE_PUBLISHABLE_KEY` (segura de exponer, es su propósito) más
políticas RLS nuevas de **solo lectura** — no un backend con la clave
secreta.

La única excepción es el leaderboard: `GET /v1/leaderboard` exige
`Authorization: Bearer <PULSO_API_KEY>`, y esa clave no tiene versión
pública. Por eso hay una sola función serverless (`dashboard/api/
leaderboard.ts`, convención nativa de Vercel, funciona junto a un sitio Vite
sin necesitar un framework con backend) que hace de proxy: la clave vive
solo ahí, como variable de entorno de servidor, y nunca llega al bundle del
navegador. Devuelve `/v1/me` y ambas ventanas del leaderboard
(`cumulative`, `rolling_24h`) combinadas en una sola respuesta.

## Políticas RLS agregadas

`supabase/migrations/20260923020000_add_public_read_policies.sql` agrega
`for select to anon using (true)` en `stations`, `observations`,
`predictions`, `forecast_runs`, `model_versions`, `training_runs`,
`model_metrics`, `drift_measurements`, `ingestion_runs` — nada sensible,
son datos sintéticos del reto y las métricas propias del equipo. No se
agregó ninguna política de escritura: siguen bloqueadas para todo lo que no
sea el `service_role` que usan los scripts de Python, exactamente igual que
antes.

## Qué muestra cada sección

| Sección | Fuente |
|---|---|
| Última ejecución del pipeline | `forecast_runs` + `ingestion_runs`, más recientes |
| Versión activa del modelo | `model_versions` donde `is_active = true`, con sus métricas de `model_metrics`, el `data_version` del dataset con el que entrenó, y un link al run de MLflow (modelo + dataset versionados juntos, ver `docs/collector-and-lineage.md`) |
| Mapa y serie por estación | `stations` (mapa fijo, 12 marcadores) + `observations` (últimos 7 días de la estación seleccionada) |
| Distribución de errores | `predictions` con `actual_demand` ya evaluado |
| Señales de drift | `drift_measurements`, la más reciente por variable. La corrección de sesgo en línea (`prediction_bias`) se muestra aparte, ya que no dispara reentrenamiento como el resto — solo corrige la predicción antes de enviarla |
| Leaderboard (acumulado y rolling 24h) | `/api/leaderboard` → API de Pulso, resalta tu propia fila |

Todo se refresca cada 60 segundos por polling simple (no hay suscripciones
en tiempo real de Supabase todavía — se puede agregar después sin rehacer
nada).

## Desarrollo local

```bash
cd dashboard
cp .env.example .env   # completa con tus credenciales
npm install
npm run dev
```

## Desplegar en Vercel

1. En [vercel.com](https://vercel.com), **Add New → Project**, importa este
   repositorio de GitHub.
2. En la configuración del proyecto, **Root Directory**: `dashboard`.
   Vercel detecta Vite automáticamente (build command y output quedan
   por defecto).
3. En **Environment Variables**, agrega:

   | Variable | Valor | Alcance |
   |---|---|---|
   | `VITE_SUPABASE_URL` | URL del proyecto de Supabase | pública (va al bundle) |
   | `VITE_SUPABASE_PUBLISHABLE_KEY` | clave publicable de Supabase | pública (va al bundle) |
   | `VITE_MLFLOW_UI_URL` (opcional) | `https://dagshub.com/<usuario>/<repo>.mlflow/#/experiments/1` | pública (va al bundle) — sin token, es solo la URL del experimento en DagsHub. Si se omite, la tarjeta del modelo activo simplemente no muestra el link |
   | `PULSO_API_URL` | `https://pulso-transmi.72-60-245-2.sslip.io` | servidor únicamente |
   | `PULSO_API_KEY` | tu clave del portal de Pulso | servidor únicamente — **nunca** le pongas el prefijo `VITE_` |

4. **Deploy**. Cada push a `main` vuelve a desplegar automáticamente.

## Verificado antes de entregar esto

- Migración RLS aplicada a Supabase real; confirmé con la clave publicable
  (no la secreta) que cada tabla listada ya devuelve filas.
- `npm run build` local compila sin errores (TypeScript + Vite).
- Probé la lógica de `api/leaderboard.ts` contra la API real de Pulso:
  responde 200 en los tres endpoints combinados, y confirmé explícitamente
  que `PULSO_API_KEY` no aparece en ninguna respuesta.
- No pude desplegar a Vercel yo mismo (no tengo cuenta) ni correr `vercel
  dev` (requiere su CLI) — los pasos de arriba son para que lo hagas tú.
