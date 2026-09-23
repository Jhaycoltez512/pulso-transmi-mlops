import { useEffect, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { supabase } from "../lib/supabase";
import type { Observation } from "../lib/types";

const HISTORY_DAYS = 7;

export function StationSeries({ stationId }: { stationId: string | null }) {
  const [rows, setRows] = useState<Observation[]>([]);
  const [loading, setLoading] = useState(false);

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
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [stationId]);

  const chartData = rows.map((row) => ({ time: new Date(row.observed_at).toLocaleString(), demand: row.demand }));

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
        Demanda — {stationId ?? "selecciona una estación en el mapa"}
      </h2>
      {!stationId ? (
        <p className="text-slate-500">Haz clic en un marcador del mapa.</p>
      ) : loading ? (
        <p className="text-slate-500">Cargando…</p>
      ) : chartData.length === 0 ? (
        <p className="text-slate-500">Sin observaciones recientes para esta estación.</p>
      ) : (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <XAxis dataKey="time" hide />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b" }} />
              <Line type="monotone" dataKey="demand" stroke="#38bdf8" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}
