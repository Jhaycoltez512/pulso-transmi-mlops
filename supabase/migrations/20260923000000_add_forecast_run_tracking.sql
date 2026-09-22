-- Track pipeline-level failures on forecast_runs (mirrors ingestion_runs.error_message),
-- and provision a private bucket for durable model artifacts between GitHub Actions runs.
alter table public.forecast_runs
  add column if not exists error_message text;

insert into storage.buckets (id, name, public)
values ('models', 'models', false)
on conflict (id) do nothing;
