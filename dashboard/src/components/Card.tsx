import type { ReactNode } from "react";

export function Card({ title, icon, right, children }: { title: string; icon: ReactNode; right?: ReactNode; children: ReactNode }) {
  return (
    <section className="group rounded-xl border border-slate-800 bg-slate-900/60 p-5 transition-all duration-300 hover:-translate-y-0.5 hover:border-slate-700 hover:bg-slate-900/80 hover:shadow-lg hover:shadow-sky-500/5">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-400 transition-colors group-hover:text-slate-300">
          <span className="text-slate-500 transition-colors group-hover:text-sky-400">{icon}</span>
          {title}
        </h2>
        {right}
      </div>
      {children}
    </section>
  );
}
