-- predicted_demand is what gets submitted (after the online bias correction in
-- run_forecast_cycle.py); raw_predicted_demand is the model's own output before it.
-- The correction factor and the performance-drift rule both read the raw value, so the
-- correction never feeds back into itself and never masks model degradation.
-- Null on rows written before the correction existed (they were uncorrected anyway).
alter table public.predictions
  add column if not exists raw_predicted_demand double precision check (raw_predicted_demand >= 0);
