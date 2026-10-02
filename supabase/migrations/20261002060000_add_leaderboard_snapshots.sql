-- Snapshots of the competition leaderboard (GET /v1/leaderboard), one row per participant and
-- snapshot, written by scripts/sync_leaderboard.py on every collector run. The API only serves
-- the cumulative and rolling-24h windows; consecutive cumulative snapshots let us derive each
-- participant's result on every resolved cycle (cycle_wape / cycle_accuracy).
create table public.leaderboard_snapshots (
  id bigserial primary key,
  captured_at timestamptz not null default now(),
  board_window text not null check (board_window in ('cumulative', 'rolling_24h')),
  -- cumulative: resolved_cycles; rolling_24h: calculated_at. One snapshot per key and window.
  snapshot_key text not null,
  resolved_cycles integer,
  starts_at timestamptz,
  -- actual demand summed over every target of the first resolved_cycles cycles (same for all)
  demand_total double precision,
  display_name text not null,
  kind text,
  eligible boolean,
  accuracy double precision,
  raw_wape double precision,
  accuracy_at_20 double precision,
  coverage double precision,
  rank integer,
  calculated_at timestamptz,
  is_me boolean not null default false,
  -- derived from this and the previous cumulative snapshot: 1 - WAPE on the newest resolved cycle
  cycle_wape double precision,
  cycle_accuracy double precision,
  unique (board_window, snapshot_key, display_name)
);

create index leaderboard_snapshots_cycles_idx on public.leaderboard_snapshots (board_window, resolved_cycles);

alter table public.leaderboard_snapshots enable row level security;
-- Read-only for the dashboard (same reasoning as 20260923020000_add_public_read_policies.sql);
-- writes stay with the service-role key.
create policy "public read" on public.leaderboard_snapshots for select to anon using (true);
