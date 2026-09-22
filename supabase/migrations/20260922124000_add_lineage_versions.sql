-- Persistent links between Supabase operational records and optional MLflow runs.
alter table public.ingestion_runs
  add column if not exists data_version text,
  add column if not exists mlflow_run_id text;

alter table public.model_versions
  add column if not exists data_version text,
  add column if not exists mlflow_run_id text;

alter table public.training_runs
  add column if not exists mlflow_run_id text;

alter table public.sync_state
  add column if not exists last_released_at timestamptz;

create index if not exists ingestion_runs_data_version_idx
  on public.ingestion_runs(data_version);
