import { useEffect, useState } from "react";

function relativeTime(from: Date, now: Date): string {
  const seconds = Math.max(0, Math.round((now.getTime() - from.getTime()) / 1000));
  if (seconds < 5) return "justo ahora";
  if (seconds < 60) return `hace ${seconds}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `hace ${minutes}m`;
  const hours = Math.round(minutes / 60);
  return `hace ${hours}h`;
}

/** Ticking "actualizado hace Xs" with a pulsing dot -- gives the dashboard a sense of being alive. */
export function LiveBadge({ lastUpdated }: { lastUpdated: Date | null }) {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const interval = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  if (!lastUpdated) return null;

  return (
    <span className="flex items-center gap-1.5 text-[11px] text-slate-600">
      <span className="relative flex h-1.5 w-1.5">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
        <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
      </span>
      actualizado {relativeTime(lastUpdated, now)}
    </span>
  );
}
