-- submissions had no creation timestamp, so "most recent submission" couldn't be queried
-- reliably (id is a random UUID, not chronological; submitted_at is null until accepted).
alter table public.submissions
  add column if not exists created_at timestamptz not null default now();

create index if not exists submissions_created_at_idx on public.submissions(created_at);
