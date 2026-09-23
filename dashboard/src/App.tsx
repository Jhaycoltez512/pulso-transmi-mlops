import { useEffect, useState } from "react";
import { ActiveModel } from "./components/ActiveModel";
import { Card } from "./components/Card";
import { DriftSignals } from "./components/DriftSignals";
import { ErrorDistribution } from "./components/ErrorDistribution";
import { MapIcon } from "./components/icons";
import { LastRun } from "./components/LastRun";
import { Leaderboard } from "./components/Leaderboard";
import { StationMap } from "./components/StationMap";
import { StationSeries } from "./components/StationSeries";
import { supabase } from "./lib/supabase";
import type { Station } from "./lib/types";

export default function App() {
  const [stations, setStations] = useState<Station[]>([]);
  const [selectedStation, setSelectedStation] = useState<string | null>(null);

  useEffect(() => {
    supabase
      .from("stations")
      .select("station_id,station_name,corridor,latitude,longitude")
      .order("station_id")
      .then(({ data }) => {
        setStations(data ?? []);
        if (data && data.length > 0) setSelectedStation((current) => current ?? data[0].station_id);
      });
  }, []);

  const selected = stations.find((s) => s.station_id === selectedStation) ?? null;

  return (
    <div className="min-h-screen bg-slate-950 px-4 py-8 sm:px-8">
      <header className="mx-auto mb-8 max-w-6xl">
        <h1 className="bg-gradient-to-r from-slate-100 to-slate-400 bg-clip-text text-2xl font-bold text-transparent">Pulso TransMi — Dashboard</h1>
        <p className="text-sm text-slate-500">Monitoreo en vivo del pipeline, el modelo activo y la posición en la competencia.</p>
      </header>

      <main className="mx-auto grid max-w-6xl gap-6">
        <div className="grid gap-6 lg:grid-cols-2">
          <LastRun />
          <ActiveModel />
        </div>

        <Card title="Estaciones" icon={<MapIcon />}>
          <div className="mb-3 flex flex-wrap gap-1.5">
            {stations.map((station) => (
              <button
                key={station.station_id}
                onClick={() => setSelectedStation(station.station_id)}
                className={`rounded-full px-2.5 py-1 text-xs transition-all duration-200 ${
                  station.station_id === selectedStation
                    ? "bg-sky-500/20 text-sky-300 ring-1 ring-sky-500/40"
                    : "bg-slate-800/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                }`}
              >
                {station.station_name}
              </button>
            ))}
          </div>
          <StationMap stations={stations} selected={selectedStation} onSelect={setSelectedStation} />
          <div className="mt-5 border-t border-slate-800 pt-5">
            <StationSeries stationId={selectedStation} stationName={selected?.station_name ?? null} />
          </div>
        </Card>

        <div className="grid gap-6 lg:grid-cols-2">
          <ErrorDistribution />
          <DriftSignals />
        </div>

        <Leaderboard />
      </main>

      <footer className="mx-auto mt-10 max-w-6xl text-xs text-slate-600">
        Datos leídos directo de Supabase con la clave pública (RLS de solo lectura). El leaderboard pasa por una función
        serverless propia para no exponer la clave de la API de Pulso en el navegador.
      </footer>
    </div>
  );
}
