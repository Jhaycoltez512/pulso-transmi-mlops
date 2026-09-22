# Supabase: migración e ingesta inicial

La migración [initial_schema.sql](../supabase/migrations/20260918193000_initial_schema.sql)
crea las tablas de datos fuente y trazabilidad MLOps acordadas. Todas tienen RLS
activado; las escrituras se realizan exclusivamente desde un proceso backend con
una clave `service_role`.

## Aplicar la migración

En Supabase CLI, autentica y vincula el proyecto una vez:

```bash
npx supabase login
npx supabase link --project-ref <project-ref>
npx supabase db push
```

Alternativamente, con una URL de conexión de Postgres con permisos de migración:

```bash
psql "$SUPABASE_DB_URL" -f supabase/migrations/20260918193000_initial_schema.sql
```

## Carga inicial

Configura la URL base del proyecto y una clave de servidor en `.env`:

```dotenv
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SECRET_KEY=<service-role-key>
```

Después de aplicar la migración, instala el SDK y ejecuta:

```bash
python -m pip install -e .
python scripts/load_supabase.py
```

El cargador consulta la API original y hace *upsert* de `stations`, `context` y
`observations`; además registra la ejecución en `ingestion_runs` y deja el
progreso en `sync_state`. Puede ejecutarse de nuevo sin duplicar las filas.
