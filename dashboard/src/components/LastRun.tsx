import { useEffect, useState } from "react";
import { supabase } from "../lib/supabase";
import type { ForecastRun, IngestionRun } from "../lib/types";
import { Card } from "./Card";
import { PulseIcon } from "./icons";
import { LiveBadge } from "./LiveBadge";
import { Skeleton } from "./Skeleton";

const REFRESH_MS = 60_000;

function StatusBadge({ status }: { status: string }) {
  const color =
    status === "succeeded" ? "bg-emerald-500/20 text-emerald-300" : status === "failed" ? "bg-red-500/20 text-red-300" : "bg-amber-500/20 text-amber-300";
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${color}`}>{status}</span>;
}

export function LastRun() {
  const [forecastRun, setForecastRun] = useState<ForecastRun | null>(null);
  const [ingestionRun, setIngestionRun] = useState<IngestionRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const [{ data: forecastRuns }, { data: ingestionRuns }] = await Promise.all([
        supabase.from("forecast_runs").select("*").order("started_at", { ascending: false }).limit(1),
        supabase.from("ingestion_runs").select("*").order("started_at", { ascending: false }).limit(1),
      ]);
      if (cancelled) return;
      setForecastRun(forecastRuns?.[0] ?? null);
      setIngestionRun(ingestionRuns?.[0] ?? null);
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
    <Card title="Última ejecución del pipeline" icon={<PulseIcon />} right={<LiveBadge lastUpdated={lastFetched} />}>
      {loading ? (
        <Skeleton lines={4} />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <p className="mb-1 text-xs text-slate-500">Collector (ingestion_runs)</p>
            {ingestionRun ? (
              <div className="space-y-1 text-sm">
                <StatusBadge status={ingestionRun.status} />
                <p className="text-slate-300">{new Date(ingestionRun.started_at).toLocaleString()}</p>
                <p className="text-slate-500">{ingestionRun.observation_rows_read} filas nuevas</p>
                {ingestionRun.error_message && <p className="break-words text-red-400">{ingestionRun.error_message}</p>}
              </div>
            ) : (
              <p className="text-slate-500">Sin datos todavía.</p>
            )}
          </div>
          <div>
            <p className="mb-1 text-xs text-slate-500">Orquestador (forecast_runs)</p>
            {forecastRun ? (
              <div className="space-y-1 text-sm">
                <StatusBadge status={forecastRun.status} />
                <p className="text-slate-300">{new Date(forecastRun.started_at).toLocaleString()}</p>
                <p className="text-slate-500">{forecastRun.trigger_reason ?? "sin decisión registrada"}</p>
                {forecastRun.error_message && <p className="break-words text-red-400">{forecastRun.error_message}</p>}
              </div>
            ) : (
              <p className="text-slate-500">Sin datos todavía.</p>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}
