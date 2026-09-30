import { useEffect, useState } from "react";
import { supabase } from "../lib/supabase";
import type { DriftMeasurement, Station } from "../lib/types";
import { Card } from "./Card";
import { DriftIcon } from "./icons";
import { LiveBadge } from "./LiveBadge";
import { Skeleton } from "./Skeleton";

const REFRESH_MS = 60_000;
export const STATION_DRIFT_PREFIX = "station_level:";
// Bars are drawn on a log scale so x0.5 and x2 sit the same distance from x1.
const SCALE_LIMIT = Math.log(3);

function position(ratio: number): number {
  const clamped = Math.max(-SCALE_LIMIT, Math.min(SCALE_LIMIT, Math.log(ratio)));
  return 50 + (clamped / SCALE_LIMIT) * 50;
}

function formatRatio(ratio: number | undefined): string {
  return ratio == null ? "—" : `×${ratio.toFixed(2)}`;
}

export function StationDrift({ stations }: { stations: Station[] }) {
  const [rows, setRows] = useState<DriftMeasurement[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const { data } = await supabase
        .from("drift_measurements")
        .select("*")
        .like("feature_name", `${STATION_DRIFT_PREFIX}%`)
        .order("calculated_at", { ascending: false })
        .limit(48);
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

  const latest = new Map<string, DriftMeasurement>();
  for (const row of rows) {
    if (!latest.has(row.feature_name)) latest.set(row.feature_name, row);
  }
  const names = new Map(stations.map((s) => [s.station_id, s.station_name]));
  const items = [...latest.values()].sort((a, b) => b.value - a.value);
  const triggered = items.filter((row) => row.triggered).length;
  const threshold = items[0]?.threshold != null ? Math.exp(items[0].threshold) : 1.5;

  return (
    <Card
      title="Drift por estación"
      icon={<DriftIcon />}
      right={
        <div className="flex items-center gap-3">
          {triggered > 0 && (
            <span className="rounded-full bg-red-500/20 px-2 py-0.5 text-xs font-medium text-red-300">
              {triggered} estación{triggered > 1 ? "es" : ""} con drift
            </span>
          )}
          <LiveBadge lastUpdated={lastFetched} />
        </div>
      }
    >
      {loading ? (
        <Skeleton lines={6} />
      ) : items.length === 0 ? (
        <p className="text-slate-500">Todavía no hay mediciones por estación (se calculan en cada ciclo con modelo activo).</p>
      ) : (
        <>
          <p className="mb-3 text-xs text-slate-500">
            Demanda de las últimas 24 h frente a las mismas 24 h de la semana anterior, comparada con esa misma relación cuando se
            entrenó el modelo activo. Por encima de ×{threshold.toFixed(1)} (o por debajo de ×{(1 / threshold).toFixed(2)}) se
            reentrena.
          </p>
          <div className="space-y-2">
            {items.map((row, i) => {
              const details = row.details ?? {};
              const change = details.relative_change ?? 1;
              const left = Math.min(50, position(change));
              const width = Math.abs(position(change) - 50);
              return (
                <div
                  key={row.feature_name}
                  style={{ animationDelay: `${i * 40}ms` }}
                  className={`animate-[fadeIn_0.4s_ease-out_both] rounded-lg border px-3 py-2 text-sm ${
                    row.triggered ? "border-red-500/40 bg-red-500/10" : "border-slate-800 bg-slate-950/40"
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="truncate font-medium text-slate-200">
                      {names.get(details.station_id ?? "") ?? details.station_id}
                      <span className="ml-1.5 text-xs text-slate-500">{details.station_id}</span>
                    </span>
                    <span className={`shrink-0 tabular-nums ${row.triggered ? "text-red-300" : "text-slate-300"}`}>
                      {formatRatio(change)} vs entrenamiento
                    </span>
                  </div>
                  <div className="relative mt-1.5 h-1.5 rounded-full bg-slate-800">
                    <div className="absolute inset-y-0 w-px bg-slate-500" style={{ left: `${position(1 / threshold)}%` }} />
                    <div className="absolute inset-y-0 w-px bg-slate-500" style={{ left: `${position(threshold)}%` }} />
                    <div
                      className={`absolute inset-y-0 rounded-full ${row.triggered ? "bg-red-400" : change >= 1 ? "bg-sky-400" : "bg-amber-400"}`}
                      style={{ left: `${left}%`, width: `${Math.max(width, 0.5)}%` }}
                    />
                  </div>
                  <p className="mt-1 text-xs text-slate-500">
                    semana a semana: hoy {formatRatio(details.ratio_now)} · al entrenar {formatRatio(details.ratio_at_training)}
                  </p>
                </div>
              );
            })}
          </div>
          <p className="mt-3 text-xs text-slate-600">{new Date(items[0].calculated_at).toLocaleString()}</p>
        </>
      )}
    </Card>
  );
}
