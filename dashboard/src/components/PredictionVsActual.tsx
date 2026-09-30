import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { supabase } from "../lib/supabase";
import { LiveBadge } from "./LiveBadge";
import { Skeleton } from "./Skeleton";

const REFRESH_MS = 60_000;
const WINDOW_HOURS = 48;
const HORIZONS = [15, 30, 45, 60] as const;
// Validated pair on the dark chart surface (lightness band, CVD and normal-vision separation, contrast).
const ACTUAL_COLOR = "#1d9bd6";
const PREDICTED_COLOR = "#d97706";

interface Point {
  t: number;
  actual: number | null;
  predicted: number | null;
}

const timeFormat = new Intl.DateTimeFormat("es-CO", { timeZone: "America/Bogota", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
const hourFormat = new Intl.DateTimeFormat("es-CO", { timeZone: "America/Bogota", hour: "2-digit", minute: "2-digit" });

export function PredictionVsActual({ stationId, stationName }: { stationId: string | null; stationName: string | null }) {
  const [horizon, setHorizon] = useState<(typeof HORIZONS)[number]>(15);
  const [points, setPoints] = useState<Point[]>([]);
  // Which station/horizon the current points belong to: loading is derived from it instead of
  // being set synchronously inside the effect.
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);
  const [showTable, setShowTable] = useState(false);

  useEffect(() => {
    if (!stationId) return;
    const key = `${stationId}:${horizon}`;
    let cancelled = false;
    async function load() {
      // Anchor on the latest observation, not the wall clock: the competition runs on simulated time.
      const { data: latest } = await supabase
        .from("observations")
        .select("observed_at")
        .eq("station_id", stationId)
        .order("observed_at", { ascending: false })
        .limit(1);
      const end = latest?.[0]?.observed_at ? new Date(latest[0].observed_at).getTime() : Date.now();
      const since = new Date(end - WINDOW_HOURS * 3_600_000).toISOString();
      const [observations, predictions] = await Promise.all([
        supabase.from("observations").select("observed_at,demand").eq("station_id", stationId).gte("observed_at", since).order("observed_at"),
        supabase
          .from("predictions")
          .select("target_at,predicted_demand")
          .eq("station_id", stationId)
          .eq("horizon_minutes", horizon)
          .gte("target_at", since)
          .order("target_at"),
      ]);
      const byTime = new Map<number, Point>();
      for (const row of observations.data ?? []) {
        const t = new Date(row.observed_at).getTime();
        byTime.set(t, { t, actual: row.demand, predicted: null });
      }
      // A target can be predicted more than once (a re-run within the same cycle): keep the last one.
      for (const row of predictions.data ?? []) {
        const t = new Date(row.target_at).getTime();
        const point = byTime.get(t) ?? { t, actual: null, predicted: null };
        point.predicted = Math.round(row.predicted_demand);
        byTime.set(t, point);
      }
      if (!cancelled) {
        setPoints([...byTime.values()].sort((a, b) => a.t - b.t));
        setLoadedKey(key);
        setLastFetched(new Date());
      }
    }
    load();
    const interval = setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [stationId, horizon]);

  const loading = stationId != null && loadedKey !== `${stationId}:${horizon}`;
  const scored = points.filter((p) => p.actual != null && p.predicted != null);
  const actualSum = scored.reduce((sum, p) => sum + (p.actual ?? 0), 0);
  const errorSum = scored.reduce((sum, p) => sum + Math.abs((p.actual ?? 0) - (p.predicted ?? 0)), 0);
  const accuracy = actualSum > 0 ? Math.max(0, 1 - errorSum / actualSum) * 100 : null;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-sm font-medium text-slate-200">Predicción vs demanda real</p>
          <p className="text-xs text-slate-500">
            últimas {WINDOW_HOURS} h de datos {stationName && <span className="text-slate-400">· {stationName}</span>}
            {accuracy != null && (
              <>
                {" "}· accuracy <span className="tabular-nums text-slate-300">{accuracy.toFixed(1)}%</span> en {scored.length} puntos
              </>
            )}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex rounded-full bg-slate-800/60 p-0.5" role="group" aria-label="Horizonte de predicción">
            {HORIZONS.map((h) => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                aria-pressed={h === horizon}
                className={`rounded-full px-2.5 py-0.5 text-xs transition-colors ${
                  h === horizon ? "bg-slate-700 text-slate-100" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {h} min
              </button>
            ))}
          </div>
          {stationId && <LiveBadge lastUpdated={lastFetched} />}
        </div>
      </div>
      {!stationId ? (
        <p className="text-slate-500">Elige una estación arriba.</p>
      ) : loading ? (
        <Skeleton lines={5} />
      ) : points.length === 0 ? (
        <p className="text-slate-500">Sin datos recientes para esta estación.</p>
      ) : (
        <>
          <div className="mb-2 flex gap-4 text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 rounded" style={{ background: ACTUAL_COLOR }} /> Demanda real
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: PREDICTED_COLOR }} /> Predicción enviada (h={horizon})
            </span>
          </div>
          <div className="h-64 animate-[fadeIn_0.4s_ease-out_both]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={points} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="#1e293b" vertical={false} />
                <XAxis
                  dataKey="t"
                  type="number"
                  scale="time"
                  domain={["dataMin", "dataMax"]}
                  tickFormatter={(t: number) => hourFormat.format(t)}
                  stroke="#64748b"
                  fontSize={11}
                  minTickGap={40}
                />
                <YAxis stroke="#64748b" fontSize={11} width={44} />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 12 }}
                  labelStyle={{ color: "#cbd5e1" }}
                  itemStyle={{ color: "#e2e8f0" }}
                  labelFormatter={(t) => timeFormat.format(Number(t))}
                  formatter={(value, name) => [value, name === "actual" ? "Demanda real" : "Predicción"]}
                  cursor={{ stroke: "#475569", strokeWidth: 1 }}
                />
                <Line type="monotone" dataKey="actual" stroke={ACTUAL_COLOR} strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke={PREDICTED_COLOR}
                  strokeWidth={2}
                  strokeDasharray="4 3"
                  dot={{ r: 3, fill: PREDICTED_COLOR, stroke: "#0f172a", strokeWidth: 2 }}
                  activeDot={{ r: 5 }}
                  connectNulls
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <button onClick={() => setShowTable((v) => !v)} className="mt-2 text-xs text-slate-500 underline-offset-2 hover:text-slate-300 hover:underline">
            {showTable ? "Ocultar tabla" : "Ver tabla"}
          </button>
          {showTable && (
            <div className="mt-2 max-h-56 overflow-auto rounded-lg border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-slate-900 text-slate-400">
                  <tr>
                    <th className="px-3 py-1.5 font-medium">Momento</th>
                    <th className="px-3 py-1.5 text-right font-medium">Real</th>
                    <th className="px-3 py-1.5 text-right font-medium">Predicción</th>
                    <th className="px-3 py-1.5 text-right font-medium">Error</th>
                  </tr>
                </thead>
                <tbody className="text-slate-300">
                  {points
                    .filter((p) => p.predicted != null)
                    .reverse()
                    .map((p) => (
                      <tr key={p.t} className="border-t border-slate-800/70">
                        <td className="px-3 py-1">{timeFormat.format(p.t)}</td>
                        <td className="px-3 py-1 text-right tabular-nums">{p.actual ?? "—"}</td>
                        <td className="px-3 py-1 text-right tabular-nums">{p.predicted}</td>
                        <td className="px-3 py-1 text-right tabular-nums">{p.actual != null ? (p.predicted! - p.actual).toLocaleString("es-CO") : "pendiente"}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
