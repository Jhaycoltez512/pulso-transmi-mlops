import { useEffect, useState } from "react";
import { supabase } from "../lib/supabase";
import type { DriftMeasurement } from "../lib/types";
import { Card } from "./Card";
import { DriftIcon } from "./icons";
import { LiveBadge } from "./LiveBadge";
import { Skeleton } from "./Skeleton";

const REFRESH_MS = 60_000;

export function DriftSignals() {
  const [rows, setRows] = useState<DriftMeasurement[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const { data } = await supabase.from("drift_measurements").select("*").order("calculated_at", { ascending: false }).limit(20);
      if (!cancelled) {
        setRows(data ?? []);
        setLoading(false);
        setLastFetched(new Date());
      }
    }
    load();
    const interval = setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  // Only the most recent measurement per feature -- older ones are history, not "current state".
  const latestByFeature = new Map<string, DriftMeasurement>();
  for (const row of rows) {
    if (!latestByFeature.has(row.feature_name)) latestByFeature.set(row.feature_name, row);
  }
  // prediction_bias isn't a retrain trigger like the others -- it's the online correction
  // applied to every prediction before submitting, so it's shown separately, not as an alarm.
  const biasRow = latestByFeature.get("prediction_bias") ?? null;
  const driftRows = [...latestByFeature.values()].filter((row) => row.feature_name !== "prediction_bias");
  const triggeredCount = driftRows.filter((row) => row.triggered).length;

  return (
    <Card
      title="Señales de drift"
      icon={<DriftIcon />}
      right={
        <div className="flex items-center gap-3">
          {triggeredCount > 0 && (
            <span className="rounded-full bg-red-500/20 px-2 py-0.5 text-xs font-medium text-red-300">{triggeredCount} activa{triggeredCount > 1 ? "s" : ""}</span>
          )}
          <LiveBadge lastUpdated={lastFetched} />
        </div>
      }
    >
      {loading ? (
        <Skeleton lines={4} />
      ) : driftRows.length === 0 && !biasRow ? (
        <p className="text-slate-500">Todavía no se ha medido drift (requiere un reentreno).</p>
      ) : (
        <div className="space-y-3">
          {biasRow && (
            <div
              className={`rounded-lg border p-3 text-sm ${
                biasRow.triggered ? "border-amber-500/40 bg-amber-500/10" : "border-slate-800 bg-slate-950/40"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-medium text-slate-200">Corrección de sesgo en línea</span>
                <span className={`rounded-full px-2 py-0.5 text-xs ${biasRow.triggered ? "bg-amber-500/20 text-amber-300" : "bg-slate-800 text-slate-400"}`}>
                  {biasRow.details?.enabled === false ? "deshabilitada" : biasRow.triggered ? "corrigiendo" : "dentro de rango"}
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                el modelo viene {biasRow.value >= 0 ? "sub-prediciendo" : "sobre-prediciendo"} {Math.abs(biasRow.value * 100).toFixed(1)}% en las últimas 4h
                {biasRow.details?.applied_scale != null && <> · escala aplicada ×{biasRow.details.applied_scale.toFixed(4)}</>}
                {biasRow.details?.samples != null && <> · {biasRow.details.samples} muestras</>}
              </p>
              <p className="text-xs text-slate-600">{new Date(biasRow.calculated_at).toLocaleString()}</p>
            </div>
          )}
          {driftRows.length > 0 && (
            <div className="grid gap-2 sm:grid-cols-2">
              {driftRows.map((row, i) => (
                <div
                  key={row.feature_name}
                  style={{ animationDelay: `${i * 60}ms` }}
                  className={`animate-[fadeIn_0.4s_ease-out_both] rounded-lg border p-3 text-sm transition-colors ${
                    row.triggered ? "border-red-500/40 bg-red-500/10 hover:bg-red-500/15" : "border-slate-800 bg-slate-950/40 hover:bg-slate-950/70"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-slate-200">{row.feature_name}</span>
                    {row.triggered && <span className="rounded-full bg-red-500/20 px-2 py-0.5 text-xs text-red-300">drift</span>}
                  </div>
                  <p className="mt-1 text-xs text-slate-500">
                    {row.method.toUpperCase()} = {row.value.toFixed(4)} {row.threshold !== null && `(umbral ${row.threshold})`}
                  </p>
                  <p className="text-xs text-slate-600">{new Date(row.calculated_at).toLocaleString()}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
