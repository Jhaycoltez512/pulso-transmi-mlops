import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { supabase } from "../lib/supabase";
import type { LeaderboardSnapshotRow } from "../lib/types";
import { Card } from "./Card";
import { TrophyIcon } from "./icons";
import { LiveBadge } from "./LiveBadge";
import { Skeleton } from "./Skeleton";

const REFRESH_MS = 60_000;
const CHART_CYCLES = 24;
const TABLE_CYCLES = 7;
// Categorical slots 1-3 (dark steps), validated all-pairs on the card surface #0f172a:
// blue = this team, orange / aqua = the two best rivals over the last TABLE_CYCLES cycles.
const SERIES_COLORS = ["#3987e5", "#d95926", "#199e70"] as const;

const hourFormat = new Intl.DateTimeFormat("es-CO", { timeZone: "America/Bogota", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });

interface Participant {
  name: string;
  isMe: boolean;
  rank: number;
  byCycle: Map<number, number | null>;
  mean: number | null;
}

function mean(values: (number | null | undefined)[]): number | null {
  const valid = values.filter((v): v is number => v != null);
  return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null;
}

export function CycleLeaderboard() {
  const [rows, setRows] = useState<LeaderboardSnapshotRow[] | null>(null);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const { data: latest } = await supabase
        .from("leaderboard_snapshots")
        .select("resolved_cycles")
        .eq("board_window", "cumulative")
        .order("resolved_cycles", { ascending: false })
        .limit(1);
      const newest = latest?.[0]?.resolved_cycles ?? 0;
      const { data } = await supabase
        .from("leaderboard_snapshots")
        .select("display_name,resolved_cycles,captured_at,rank,accuracy,cycle_accuracy,is_me")
        .eq("board_window", "cumulative")
        .gt("resolved_cycles", newest - CHART_CYCLES)
        .order("resolved_cycles");
      if (!cancelled) {
        setRows((data as LeaderboardSnapshotRow[]) ?? []);
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

  const view = useMemo(() => {
    if (!rows || rows.length === 0) return null;
    const cycles = [...new Set(rows.map((r) => r.resolved_cycles))].sort((a, b) => a - b);
    const capturedAt = new Map(rows.map((r) => [r.resolved_cycles, r.captured_at]));
    const newest = cycles[cycles.length - 1];
    const tableCycles = cycles.filter((c) => c > newest - TABLE_CYCLES);
    const people = new Map<string, Participant>();
    for (const r of rows) {
      const p = people.get(r.display_name) ?? { name: r.display_name, isMe: r.is_me, rank: r.rank, byCycle: new Map(), mean: null };
      p.byCycle.set(r.resolved_cycles, r.cycle_accuracy);
      if (r.resolved_cycles === newest) p.rank = r.rank;
      people.set(r.display_name, p);
    }
    for (const p of people.values()) p.mean = mean(tableCycles.map((c) => p.byCycle.get(c)));
    const ranked = [...people.values()].sort((a, b) => (b.mean ?? -1) - (a.mean ?? -1));
    const me = ranked.find((p) => p.isMe);
    const rivals = ranked.filter((p) => !p.isMe && p.mean != null).slice(0, 2);
    const series = [me, ...rivals].filter((p): p is Participant => p != null);
    const points = cycles.map((c) => {
      const point: Record<string, number | string | null> = { cycle: c, at: capturedAt.get(c) ?? "" };
      for (const p of series) point[p.name] = p.byCycle.get(c) ?? null;
      return point;
    });
    const hasCycleData = rows.some((r) => r.cycle_accuracy != null);
    const latest = (p: Participant) => [...cycles].reverse().map((c) => p.byCycle.get(c)).find((v) => v != null) ?? null;
    const shown = series.flatMap((p) => cycles.map((c) => p.byCycle.get(c))).filter((v): v is number => v != null);
    // Zoom to the band the shown series live in (never below 0): a 0-100 axis flattens the race.
    const yMin = shown.length ? Math.max(0, Math.floor((Math.min(...shown) - 5) / 10) * 10) : 0;
    return { cycles, tableCycles, capturedAt, ranked, series, points, hasCycleData, latest, yMin };
  }, [rows]);

  return (
    <Card title="Competencia ciclo a ciclo" icon={<TrophyIcon />} right={<LiveBadge lastUpdated={lastFetched} />}>
      {rows === null ? (
        <Skeleton lines={5} />
      ) : !view || !view.hasCycleData ? (
        <p className="text-sm text-slate-500">
          Recolectando fotos de la tabla de posiciones en cada ciclo. El resultado por ciclo aparece desde el segundo ciclo resuelto después de activar
          esta vista.
        </p>
      ) : (
        <>
          <p className="mb-3 text-xs text-slate-500">
            Accuracy de cada ciclo = 1 − WAPE global del ciclo, derivada del acumulado oficial de la API (puede diferir levemente de la accuracy por
            estación). Gráfica: tu equipo y los dos mejores rivales de los últimos {TABLE_CYCLES} ciclos.
          </p>
          <div className="mb-2 flex flex-wrap gap-4 text-xs text-slate-400">
            {view.series.map((p, i) => (
              <span key={p.name} className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 rounded" style={{ background: SERIES_COLORS[i] }} />
                {p.isMe ? `${p.name} (tú)` : p.name}
                {view.latest(p) != null && <span className="tabular-nums text-slate-300">· {view.latest(p)!.toFixed(1)}%</span>}
              </span>
            ))}
          </div>
          <div className="h-64 animate-[fadeIn_0.4s_ease-out_both]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={view.points} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="#1e293b" vertical={false} />
                <XAxis
                  dataKey="cycle"
                  stroke="#64748b"
                  fontSize={11}
                  minTickGap={60}
                  padding={{ right: 8 }}
                  tickFormatter={(c: number) => {
                    const at = view.capturedAt.get(c);
                    return at ? hourFormat.format(new Date(at)) : String(c);
                  }}
                />
                <YAxis stroke="#64748b" fontSize={11} width={44} domain={[view.yMin, 100]} unit="%" allowDataOverflow />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 12 }}
                  labelStyle={{ color: "#cbd5e1" }}
                  itemStyle={{ color: "#e2e8f0" }}
                  labelFormatter={(c) => {
                    const at = view.capturedAt.get(Number(c));
                    return `Ciclo ${c}${at ? ` · ${hourFormat.format(new Date(at))}` : ""}`;
                  }}
                  formatter={(value) => [value == null ? "no envió" : `${Number(value).toFixed(1)}%`]}
                  cursor={{ stroke: "#475569", strokeWidth: 1 }}
                />
                {view.series.map((p, i) => (
                  <Line
                    key={p.name}
                    type="monotone"
                    dataKey={p.name}
                    name={p.name}
                    stroke={SERIES_COLORS[i]}
                    strokeWidth={2}
                    dot={{ r: 3, fill: SERIES_COLORS[i], stroke: "#0f172a", strokeWidth: 2 }}
                    activeDot={{ r: 5 }}
                    connectNulls
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <p className="mb-2 mt-5 text-sm font-medium text-slate-200">Últimos {view.tableCycles.length} ciclos</p>
          <div className="overflow-x-auto rounded-lg border border-slate-800">
            <table className="w-full text-xs">
              <thead className="bg-slate-900 text-slate-400">
                <tr>
                  <th className="px-2 py-1.5 text-left font-medium">#</th>
                  <th className="px-2 py-1.5 text-left font-medium">participante</th>
                  {view.tableCycles.map((c) => (
                    <th key={c} className="px-2 py-1.5 text-right font-medium" title={`Ciclo ${c}`}>
                      {(() => {
                        const at = view.capturedAt.get(c);
                        return at ? new Intl.DateTimeFormat("es-CO", { timeZone: "America/Bogota", hour: "2-digit", minute: "2-digit" }).format(new Date(at)) : c;
                      })()}
                    </th>
                  ))}
                  <th className="px-2 py-1.5 text-right font-medium">promedio</th>
                </tr>
              </thead>
              <tbody>
                {view.ranked.map((p) => (
                  <tr
                    key={p.name}
                    className={`border-t border-slate-800 ${p.isMe ? "bg-sky-500/10 font-semibold text-slate-100" : "text-slate-300 hover:bg-slate-800/40"}`}
                  >
                    <td className="px-2 py-1 tabular-nums text-slate-500" title="Puesto en el acumulado oficial">
                      {p.rank}
                    </td>
                    <td className="px-2 py-1">{p.isMe ? `${p.name} (tú)` : p.name}</td>
                    {view.tableCycles.map((c) => {
                      const v = p.byCycle.get(c);
                      return (
                        <td key={c} className="px-2 py-1 text-right tabular-nums">
                          {v == null ? <span className="text-slate-600">{p.byCycle.has(c) ? "no envió" : "—"}</span> : v.toFixed(1)}
                        </td>
                      );
                    })}
                    <td className="px-2 py-1 text-right tabular-nums">{p.mean == null ? "—" : `${p.mean.toFixed(1)}%`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  );
}
