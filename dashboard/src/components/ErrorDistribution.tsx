import { useEffect, useState } from "react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { supabase } from "../lib/supabase";
import type { Prediction } from "../lib/types";

const REFRESH_MS = 60_000;
const BIN_COUNT = 12;

function buildHistogram(errors: number[]): { bucket: string; count: number }[] {
  if (errors.length === 0) return [];
  const min = Math.min(...errors);
  const max = Math.max(...errors);
  const width = (max - min || 1) / BIN_COUNT;
  const bins = Array.from({ length: BIN_COUNT }, (_, i) => ({ start: min + i * width, count: 0 }));
  for (const error of errors) {
    const index = Math.min(BIN_COUNT - 1, Math.floor((error - min) / width));
    bins[index].count += 1;
  }
  return bins.map((bin) => ({ bucket: bin.start.toFixed(0), count: bin.count }));
}

export function ErrorDistribution() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const { data } = await supabase
        .from("predictions")
        .select("station_id,target_at,horizon_minutes,predicted_demand,actual_demand,evaluated_at")
        .not("actual_demand", "is", null)
        .order("evaluated_at", { ascending: false })
        .limit(2000);
      if (!cancelled) {
        setPredictions(data ?? []);
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

  const errors = predictions.filter((p) => p.actual_demand !== null).map((p) => p.predicted_demand - (p.actual_demand as number));
  const histogram = buildHistogram(errors);
  const meanAbsError = errors.length ? errors.reduce((sum, e) => sum + Math.abs(e), 0) / errors.length : null;

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide text-slate-400">Distribución de errores</h2>
      <p className="mb-3 text-xs text-slate-500">
        predicción − demanda real, sobre las {errors.length} predicciones ya evaluadas {meanAbsError !== null && `· error absoluto medio: ${meanAbsError.toFixed(1)}`}
      </p>
      {loading ? (
        <p className="text-slate-500">Cargando…</p>
      ) : histogram.length === 0 ? (
        <p className="text-slate-500">Todavía no hay predicciones evaluadas contra demanda real.</p>
      ) : (
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogram}>
              <XAxis dataKey="bucket" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={12} allowDecimals={false} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b" }} />
              <Bar dataKey="count" fill="#f59e0b" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}
