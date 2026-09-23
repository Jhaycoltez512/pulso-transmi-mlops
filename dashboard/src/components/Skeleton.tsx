export function Skeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="animate-pulse space-y-2">
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className="h-3 rounded bg-slate-800" style={{ width: `${85 - i * 15}%` }} />
      ))}
    </div>
  );
}
