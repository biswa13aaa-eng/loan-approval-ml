create table if not exists public.prediction_history (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  model_name text not null,
  prediction text not null check (prediction in ('Approved','Rejected')),
  approval_probability double precision,
  risk_level text
);
alter table public.prediction_history enable row level security;
