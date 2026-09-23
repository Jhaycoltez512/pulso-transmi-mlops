import { useEffect, useState } from "react";
import type { LeaderboardResponse, LeaderboardRow } from "../lib/types";
import { Card } from "./Card";
import { TrophyIcon } from "./icons";
import { LiveBadge } from "./LiveBadge";
import { Skeleton } from "./Skeleton";

const REFRESH_MS = 60_000;
const MEDALS: Record<number, string> = { 1: "🥇", 2: "🥈", 3: "🥉" };

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
          <tr
            key={row.rank}
            className={`border-t border-slate-800 transition-colors ${
              row.display_name === meName ? "bg-sky-500/10 font-semibold text-sky-300" : "text-slate-300 hover:bg-slate-800/40"
            }`}
          >
            <td className="py-1.5">{MEDALS[row.rank] ?? row.rank}</td>
            <td className="py-1.5">{row.display_name}</td>
            <td className="py-1.5 text-right">{row.accuracy.toFixed(1)}%</td>
            <td className="py-1.5 text-right">{row.raw_wape.toFixed(3)}</td>
            <td className="py-1.5 text-right">{(row.coverage * 100).toFixed(0)}%</td>
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
  const [lastFetched, setLastFetched] = useState<Date | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const response = await fetch("/api/leaderboard");
        const json = (await response.json()) as LeaderboardResponse;
        if (!cancelled) {
          setData(json);
          setLastFetched(new Date());
        }
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
    <Card
      title="Leaderboard"
      icon={<TrophyIcon />}
      right={
        <div className="flex items-center gap-3">
          <div className="flex gap-1 rounded-lg bg-slate-950/60 p-0.5 text-xs">
            {(["cumulative", "rolling_24h"] as const).map((key) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={`rounded-md px-2 py-1 transition-colors duration-200 ${tab === key ? "bg-sky-500/20 text-sky-300" : "text-slate-500 hover:text-slate-300"}`}
              >
                {key === "cumulative" ? "acumulado" : "rolling 24h"}
              </button>
            ))}
          </div>
          <LiveBadge lastUpdated={lastFetched} />
        </div>
      }
    >
      {loading ? (
        <Skeleton lines={5} />
      ) : data?.error ? (
        <p className="text-red-400">{data.error}</p>
      ) : !data || data[tab].length === 0 ? (
        <p className="text-slate-500">Sin datos de leaderboard todavía.</p>
      ) : (
        <>
          {meRow && (
            <p className="mb-2 animate-[fadeIn_0.3s_ease-out_both] text-xs text-sky-300">
              Tu posición: #{meRow.rank} · {meRow.accuracy.toFixed(1)}% accuracy
            </p>
          )}
          <div className="max-h-64 overflow-y-auto">
            <Table rows={data[tab]} meName={data.me?.display_name} />
          </div>
        </>
      )}
    </Card>
  );
}
