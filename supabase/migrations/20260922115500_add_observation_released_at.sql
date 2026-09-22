-- The competition stream exposes when an observation became available.
-- Keep it separately from observed_at for ingestion auditability.
alter table public.observations
  add column if not exists released_at timestamptz;

create index if not exists observations_released_at_idx
  on public.observations(released_at);
