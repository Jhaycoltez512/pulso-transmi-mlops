-- Read-only public access for the optional dashboard bonus (docs/student-project.md).
-- The dashboard uses the anon/publishable key from the browser; nothing here is sensitive
-- (synthetic challenge data and this team's own model/pipeline metrics). Writes stay
-- restricted to the service-role key used by the backend scripts -- no write policy is added.
create policy "public read" on public.stations for select to anon using (true);
create policy "public read" on public.observations for select to anon using (true);
create policy "public read" on public.predictions for select to anon using (true);
create policy "public read" on public.forecast_runs for select to anon using (true);
create policy "public read" on public.model_versions for select to anon using (true);
create policy "public read" on public.training_runs for select to anon using (true);
create policy "public read" on public.model_metrics for select to anon using (true);
create policy "public read" on public.drift_measurements for select to anon using (true);
create policy "public read" on public.ingestion_runs for select to anon using (true);
