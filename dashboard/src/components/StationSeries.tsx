import { useEffect, useState } from "react";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { supabase } from "../lib/supabase";
import type { Observation } from "../lib/types";
import { LiveBadge } from "./LiveBadge";
import { Skeleton } from "./Skeleton";

const HISTORY_DAYS = 7;

export function StationSeries({ stationId, stationName }: { stationId: string | null; stationName: string | null }) {
  const [rows, setRows] = useState<Observation[]>([]);
  const [loading, setLoading] = useState(false);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);

  useEffect(() => {
    if (!stationId) {
      setRows([]);
      return;
    }
    let cancelled = false;
    setLoading(true);
    async function load() {
      const since = new Date(Date.now() - HISTORY_DAYS * 24 * 60 * 60 * 1000).toISOString();
      const { data } = await supabase
        .from("observations")
        .select("station_id,observed_at,demand")
        .eq("station_id", stationId)
        .gte("observed_at", since)
        .order("observed_at", { ascending: true });
      if (!cancelled) {
        setRows(data ?? []);
        setLoading(false);
        setLastFetched(new Date());
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [stationId]);

  const chartData = rows.map((row) => ({ time: new Date(row.observed_at).toLocaleString(), demand: row.demand }));

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <p className="text-xs text-slate-500">
          últimos {HISTORY_DAYS} días {stationName && <span className="text-slate-400">· {stationName}</span>}
        </p>
        {stationId && <LiveBadge lastUpdated={lastFetched} />}
      </div>
      {!stationId ? (
        <p className="text-slate-500">Elige una estación arriba o haz clic en un marcador del mapa.</p>
      ) : loading ? (
        <Skeleton lines={5} />
      ) : chartData.length === 0 ? (
        <p className="text-slate-500">Sin observaciones recientes para esta estación.</p>
      ) : (
        <div className="h-64 animate-[fadeIn_0.4s_ease-out_both]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="demandFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#38bdf8" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="time" hide />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b" }} />
              <Area type="monotone" dataKey="demand" stroke="#38bdf8" strokeWidth={2} fill="url(#demandFill)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
