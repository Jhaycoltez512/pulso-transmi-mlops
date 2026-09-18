-- Pulso TransMi: schema for source data and MLOps traceability.
-- Apply with `supabase db push` or `psql "$SUPABASE_DB_URL" -f <this-file>`.

create extension if not exists pgcrypto;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table public.stations (
  station_id varchar(5) primary key check (station_id ~ '^[0-9]{5}$'),
  station_name text not null,
  corridor text not null,
  latitude double precision not null check (latitude between -90 and 90),
  longitude double precision not null check (longitude between -180 and 180),
  source_updated_at timestamptz not null default now()
);

create table public.ingestion_runs (
  id uuid primary key default gen_random_uuid(),
  source_name text not null default 'pulso-transmi-api',
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  status text not null default 'running' check (status in ('running', 'succeeded', 'failed')),
  cursor_before text,
  cursor_after text,
  last_observed_at timestamptz,
  observation_rows_read integer not null default 0 check (observation_rows_read >= 0),
  context_rows_read integer not null default 0 check (context_rows_read >= 0),
  error_message text,
  git_commit varchar(40)
);

create table public.sync_state (
  stream_name text primary key check (stream_name in ('observations', 'context')),
  last_cursor text,
  last_observed_at timestamptz,
  last_ingestion_run_id uuid references public.ingestion_runs(id) on delete set null,
  updated_at timestamptz not null default now()
);

create table public.context (
  observed_at timestamptz primary key,
  rain_mm double precision not null check (rain_mm >= 0),
  rain_forecast double precision not null check (rain_forecast >= 0),
  temperature_c double precision not null,
  temperature_forecast double precision not null,
  event_intensity double precision not null check (event_intensity >= 0),
  ingestion_run_id uuid references public.ingestion_runs(id) on delete set null,
  received_at timestamptz not null default now()
);

create table public.observations (
  station_id varchar(5) not null references public.stations(station_id),
  observed_at timestamptz not null,
  demand integer not null check (demand >= 0),
  ingestion_run_id uuid references public.ingestion_runs(id) on delete set null,
  received_at timestamptz not null default now(),
  primary key (station_id, observed_at)
);
create index observations_observed_at_idx on public.observations(observed_at);

create table public.data_quality_results (
  id uuid primary key default gen_random_uuid(),
  ingestion_run_id uuid not null references public.ingestion_runs(id) on delete cascade,
  rule_name text not null,
  status text not null check (status in ('passed', 'warning', 'failed')),
  affected_rows integer not null default 0 check (affected_rows >= 0),
  details jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table public.drift_measurements (
  id uuid primary key default gen_random_uuid(),
  ingestion_run_id uuid not null references public.ingestion_runs(id) on delete cascade,
  feature_name text not null,
  drift_type text not null check (drift_type in ('data', 'concept', 'performance')),
  method text not null,
  value double precision not null,
  threshold double precision,
  triggered boolean not null default false,
  details jsonb not null default '{}'::jsonb,
  calculated_at timestamptz not null default now()
);

create table public.feature_sets (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  version text not null,
  definition jsonb not null,
  code_commit varchar(40),
  created_at timestamptz not null default now(),
  unique (name, version)
);

create table public.model_versions (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  version varchar(64) not null unique,
  algorithm text not null,
  artifact_uri text,
  feature_set_id uuid references public.feature_sets(id) on delete set null,
  trained_at timestamptz,
  training_data_end timestamptz,
  git_commit varchar(40),
  is_active boolean not null default false,
  created_at timestamptz not null default now()
);
create unique index model_versions_one_active_idx on public.model_versions(is_active) where is_active;

create table public.training_runs (
  id uuid primary key default gen_random_uuid(),
  model_version_id uuid references public.model_versions(id) on delete set null,
  feature_set_id uuid references public.feature_sets(id) on delete set null,
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  training_start timestamptz,
  training_end timestamptz,
  validation_start timestamptz,
  validation_end timestamptz,
  trigger_reason text not null,
  status text not null check (status in ('running', 'succeeded', 'failed')),
  parameters jsonb not null default '{}'::jsonb
);

create table public.model_metrics (
  id uuid primary key default gen_random_uuid(),
  training_run_id uuid not null references public.training_runs(id) on delete cascade,
  station_id varchar(5) references public.stations(station_id) on delete set null,
  split_name text not null,
  metric_name text not null,
  metric_value double precision not null,
  calculated_at timestamptz not null default now(),
  unique (training_run_id, station_id, split_name, metric_name)
);

create table public.forecast_cycles (
  cycle_id text primary key,
  state text not null,
  origin_at timestamptz not null,
  data_cutoff timestamptz not null,
  opens_at timestamptz,
  closes_at timestamptz,
  expected_predictions integer not null check (expected_predictions > 0),
  fetched_at timestamptz not null default now()
);

create table public.forecast_targets (
  cycle_id text not null references public.forecast_cycles(cycle_id) on delete cascade,
  station_id varchar(5) not null references public.stations(station_id),
  target_at timestamptz not null,
  horizon_minutes integer not null check (horizon_minutes > 0),
  primary key (cycle_id, station_id, target_at)
);

create table public.forecast_runs (
  id uuid primary key default gen_random_uuid(),
  cycle_id text references public.forecast_cycles(cycle_id) on delete set null,
  model_version_id uuid references public.model_versions(id) on delete set null,
  feature_set_id uuid references public.feature_sets(id) on delete set null,
  data_cutoff timestamptz not null,
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  status text not null check (status in ('running', 'succeeded', 'failed')),
  git_commit varchar(40),
  trigger_reason text
);

create table public.submissions (
  id uuid primary key default gen_random_uuid(),
  external_submission_id text unique,
  forecast_run_id uuid not null unique references public.forecast_runs(id) on delete cascade,
  cycle_id text not null references public.forecast_cycles(cycle_id),
  client_run_id varchar(128) not null unique,
  idempotency_key varchar(128) not null unique,
  submitted_at timestamptz,
  status text not null check (status in ('pending', 'accepted', 'rejected', 'failed')),
  response_payload jsonb,
  error_message text
);

create table public.predictions (
  id uuid primary key default gen_random_uuid(),
  forecast_run_id uuid not null references public.forecast_runs(id) on delete cascade,
  station_id varchar(5) not null references public.stations(station_id),
  target_at timestamptz not null,
  horizon_minutes integer not null check (horizon_minutes > 0),
  predicted_demand double precision not null check (predicted_demand >= 0),
  actual_demand integer check (actual_demand >= 0),
  evaluated_at timestamptz,
  unique (forecast_run_id, station_id, target_at)
);
create index predictions_station_target_idx on public.predictions(station_id, target_at);

create trigger sync_state_set_updated_at before update on public.sync_state
for each row execute function public.set_updated_at();

-- The database is backend-only. The ingestion script must use a service-role key,
-- which bypasses RLS; no anonymous write policy is created.
alter table public.stations enable row level security;
alter table public.ingestion_runs enable row level security;
alter table public.sync_state enable row level security;
alter table public.context enable row level security;
alter table public.observations enable row level security;
alter table public.data_quality_results enable row level security;
alter table public.drift_measurements enable row level security;
alter table public.feature_sets enable row level security;
alter table public.model_versions enable row level security;
alter table public.training_runs enable row level security;
alter table public.model_metrics enable row level security;
alter table public.forecast_cycles enable row level security;
alter table public.forecast_targets enable row level security;
alter table public.forecast_runs enable row level security;
alter table public.submissions enable row level security;
alter table public.predictions enable row level security;
