import { useEffect, useState } from "react";
import { ActiveModel } from "./components/ActiveModel";
import { DriftSignals } from "./components/DriftSignals";
import { ErrorDistribution } from "./components/ErrorDistribution";
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
      .then(({ data }) => setStations(data ?? []));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 px-4 py-8 sm:px-8">
      <header className="mx-auto mb-8 max-w-6xl">
        <h1 className="text-2xl font-bold text-slate-100">Pulso TransMi — Dashboard</h1>
        <p className="text-sm text-slate-500">Monitoreo en vivo del pipeline, el modelo activo y la posición en la competencia.</p>
      </header>

      <main className="mx-auto grid max-w-6xl gap-6">
        <div className="grid gap-6 lg:grid-cols-2">
          <LastRun />
          <ActiveModel />
        </div>

        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Estaciones</h2>
          <StationMap stations={stations} selected={selectedStation} onSelect={setSelectedStation} />
        </section>

        <StationSeries stationId={selectedStation} />

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
