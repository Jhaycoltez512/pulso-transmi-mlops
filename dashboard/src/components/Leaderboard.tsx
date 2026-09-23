import { useEffect, useState } from "react";
import type { LeaderboardResponse, LeaderboardRow } from "../lib/types";

const REFRESH_MS = 60_000;

function Table({ rows, meName }: { rows: LeaderboardRow[]; meName: string | undefined }) {
  return (
    <table className="w-full text-xs">
      <thead className="text-slate-500">
        <tr>
          <th className="pb-1 text-left">#</th>
          <th className="pb-1 text-left">participante</th>
          <th className="pb-1 text-right">accuracy</th>
          <th className="pb-1 text-right">WAPE</th>
          <th className="pb-1 text-right">cobertura</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.rank} className={`border-t border-slate-800 ${row.display_name === meName ? "bg-sky-500/10 font-semibold text-sky-300" : "text-slate-300"}`}>
            <td className="py-1">{row.rank}</td>
            <td className="py-1">{row.display_name}</td>
            <td className="py-1 text-right">{row.accuracy.toFixed(1)}%</td>
            <td className="py-1 text-right">{row.raw_wape.toFixed(3)}</td>
            <td className="py-1 text-right">{(row.coverage * 100).toFixed(0)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function Leaderboard() {
  const [data, setData] = useState<LeaderboardResponse | null>(null);
  const [tab, setTab] = useState<"cumulative" | "rolling_24h">("cumulative");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const response = await fetch("/api/leaderboard");
        const json = (await response.json()) as LeaderboardResponse;
        if (!cancelled) setData(json);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    const interval = setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const meRow = data?.[tab]?.find((row) => row.display_name === data.me?.display_name);

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">Leaderboard</h2>
        <div className="flex gap-1 text-xs">
          <button onClick={() => setTab("cumulative")} className={`rounded px-2 py-1 ${tab === "cumulative" ? "bg-sky-500/20 text-sky-300" : "text-slate-500"}`}>
            acumulado
          </button>
          <button onClick={() => setTab("rolling_24h")} className={`rounded px-2 py-1 ${tab === "rolling_24h" ? "bg-sky-500/20 text-sky-300" : "text-slate-500"}`}>
            rolling 24h
          </button>
        </div>
      </div>
      {loading ? (
        <p className="text-slate-500">Cargando…</p>
      ) : data?.error ? (
        <p className="text-red-400">{data.error}</p>
      ) : !data || data[tab].length === 0 ? (
        <p className="text-slate-500">Sin datos de leaderboard todavía.</p>
      ) : (
        <>
          {meRow && (
            <p className="mb-2 text-xs text-sky-300">
              Tu posición: #{meRow.rank} · {meRow.accuracy.toFixed(1)}% accuracy
            </p>
          )}
          <div className="max-h-64 overflow-y-auto">
            <Table rows={data[tab]} meName={data.me?.display_name} />
          </div>
        </>
      )}
    </section>
  );
}
