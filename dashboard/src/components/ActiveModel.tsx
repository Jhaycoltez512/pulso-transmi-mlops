import { useEffect, useState } from "react";
import { supabase } from "../lib/supabase";
import type { ModelMetric, ModelVersion } from "../lib/types";

const REFRESH_MS = 60_000;
const HORIZONS = [15, 30, 45, 60];

export function ActiveModel() {
  const [model, setModel] = useState<ModelVersion | null>(null);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const { data: models } = await supabase.from("model_versions").select("*").eq("is_active", true).limit(1);
      const active = models?.[0] ?? null;
      if (cancelled) return;
      setModel(active);
      if (active) {
        // model_metrics is keyed by training_run_id, not model_version_id -- look it up first.
        const { data: trainingRuns } = await supabase.from("training_runs").select("id").eq("model_version_id", active.id).order("started_at", { ascending: false }).limit(1);
        const trainingRunId = trainingRuns?.[0]?.id;
        if (trainingRunId) {
          const { data: metricRows } = await supabase.from("model_metrics").select("*").eq("training_run_id", trainingRunId).eq("split_name", "test").is("station_id", null);
          if (!cancelled) setMetrics(metricRows ?? []);
        }
      }
      setLoading(false);
    }
    load();
    const interval = setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Versión activa del modelo</h2>
      {loading ? (
        <p className="text-slate-500">Cargando…</p>
      ) : model ? (
        <div className="space-y-2 text-sm">
          <p className="font-mono text-base text-slate-100">{model.version}</p>
          <p className="text-slate-400">{model.algorithm}</p>
          <p className="text-slate-500">
            entrenado {model.trained_at ? new Date(model.trained_at).toLocaleString() : "—"} · commit{" "}
            <span className="font-mono">{model.git_commit?.slice(0, 8) ?? "—"}</span>
          </p>
          {metrics.length > 0 && (
            <table className="mt-3 w-full text-xs">
              <thead className="text-slate-500">
                <tr>
                  <th className="pb-1 text-left">horizonte</th>
                  <th className="pb-1 text-right">WAPE test</th>
                  <th className="pb-1 text-right">accuracy test</th>
                </tr>
              </thead>
              <tbody>
                {HORIZONS.map((horizon) => {
                  const label = `h${horizon}_test`;
                  const wape = metrics.find((m) => m.split_name === label && m.metric_name === "mean_station_wape");
                  const accuracy = metrics.find((m) => m.split_name === label && m.metric_name === "mean_station_accuracy");
                  return (
                    <tr key={horizon} className="border-t border-slate-800 text-slate-300">
                      <td className="py-1">{horizon} min</td>
                      <td className="py-1 text-right">{wape ? wape.metric_value.toFixed(4) : "—"}</td>
                      <td className="py-1 text-right">{accuracy ? `${accuracy.metric_value.toFixed(2)}%` : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      ) : (
        <p className="text-slate-500">Sin modelo activo todavía.</p>
      )}
    </section>
  );
}
