import { useEffect, useState } from "react";
import { supabase } from "../lib/supabase";
import type { ModelMetric, ModelVersion } from "../lib/types";
import { Card } from "./Card";
import { ExternalLinkIcon, ModelIcon } from "./icons";
import { LiveBadge } from "./LiveBadge";
import { Skeleton } from "./Skeleton";

const REFRESH_MS = 60_000;
const HORIZONS = [15, 30, 45, 60];
const MLFLOW_UI_URL = import.meta.env.VITE_MLFLOW_UI_URL as string | undefined;

function AccuracyBar({ value }: { value: number }) {
  const color = value >= 85 ? "bg-emerald-500" : value >= 70 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
      <div className={`h-full rounded-full ${color} transition-all duration-700 ease-out`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}

export function ActiveModel() {
  const [model, setModel] = useState<ModelVersion | null>(null);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);

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
      setLastFetched(new Date());
    }
    load();
    const interval = setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <Card title="Versión activa del modelo" icon={<ModelIcon />} right={<LiveBadge lastUpdated={lastFetched} />}>
      {loading ? (
        <Skeleton lines={4} />
      ) : model ? (
        <div className="space-y-2 text-sm">
          <p className="font-mono text-base text-slate-100">{model.version}</p>
          <p className="text-slate-400">{model.algorithm}</p>
          <p className="text-slate-500">
            entrenado {model.trained_at ? new Date(model.trained_at).toLocaleString() : "—"} · commit{" "}
            <span className="font-mono">{model.git_commit?.slice(0, 8) ?? "—"}</span>
          </p>
          {model.data_version && (
            <p className="truncate text-slate-500" title={model.data_version}>
              dataset <span className="font-mono text-slate-400">{model.data_version}</span>
            </p>
          )}
          {model.mlflow_run_id && MLFLOW_UI_URL && (
            <a
              href={`${MLFLOW_UI_URL}/runs/${model.mlflow_run_id}`}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-sky-400 transition-colors hover:text-sky-300"
            >
              Ver modelo y dataset en MLflow <ExternalLinkIcon className="h-3 w-3" />
            </a>
          )}
          {metrics.length > 0 && (
            <div className="mt-3 space-y-2">
              {HORIZONS.map((horizon) => {
                const label = `h${horizon}_test`;
                const wape = metrics.find((m) => m.split_name === label && m.metric_name === "mean_station_wape");
                const accuracy = metrics.find((m) => m.split_name === label && m.metric_name === "mean_station_accuracy");
                return (
                  <div key={horizon} className="grid grid-cols-[3rem_1fr_3.5rem] items-center gap-2 text-xs">
                    <span className="text-slate-400">{horizon} min</span>
                    <AccuracyBar value={accuracy?.metric_value ?? 0} />
                    <span className="text-right text-slate-300">{accuracy ? `${accuracy.metric_value.toFixed(1)}%` : "—"}</span>
                    <span className="col-span-3 -mt-1 text-right text-[10px] text-slate-600">WAPE {wape ? wape.metric_value.toFixed(4) : "—"}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      ) : (
        <p className="text-slate-500">Sin modelo activo todavía.</p>
      )}
    </Card>
  );
}
