import { useEffect, useState } from "react";
import { supabase } from "../lib/supabase";
import type { DriftMeasurement } from "../lib/types";

const REFRESH_MS = 60_000;

export function DriftSignals() {
  const [rows, setRows] = useState<DriftMeasurement[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const { data } = await supabase.from("drift_measurements").select("*").order("calculated_at", { ascending: false }).limit(20);
      if (!cancelled) {
        setRows(data ?? []);
        setLoading(false);
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

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Señales de drift</h2>
      {loading ? (
        <p className="text-slate-500">Cargando…</p>
      ) : latestByFeature.size === 0 ? (
        <p className="text-slate-500">Todavía no se ha medido drift (requiere un reentreno).</p>
      ) : (
        <div className="grid gap-2 sm:grid-cols-2">
          {[...latestByFeature.values()].map((row) => (
            <div
              key={row.feature_name}
              className={`rounded-lg border p-3 text-sm ${row.triggered ? "border-red-500/40 bg-red-500/10" : "border-slate-800 bg-slate-950/40"}`}
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
    </section>
  );
}
