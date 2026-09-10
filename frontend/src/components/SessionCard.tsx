import { bandColor } from './RiskGauge'

/** Session card — quiet list row, Apple style. */
export function SessionCard({ s, onClick, active }: { s: any; onClick?: () => void; active?: boolean }) {
  const band =
    s.max_score >= 0.85 ? 'critical' : s.max_score >= 0.68 ? 'high' : s.max_score >= 0.35 ? 'medium' : 'low'
  const color = bandColor(band)
  return (
    <button
      onClick={onClick}
      className="w-full rounded-2xl border p-4 text-left transition-colors"
      style={{
        background: active ? '#f5f5f7' : '#fff',
        borderColor: active ? '#1d1d1f' : '#d2d2d7',
      }}
    >
      <div className="flex items-center justify-between">
        <div className="font-mono text-[13px]" style={{ color: '#1d1d1f' }}>
          {s.call_id}
        </div>
        <div className="text-[15px] font-semibold" style={{ color }}>
          {(s.max_score * 100).toFixed(0)}%
        </div>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[12px]" style={{ color: '#6e6e73' }}>
        <span className="capitalize">{s.role}</span>
        <span className="uppercase">{s.language}</span>
        <span>{s.window_count} windows</span>
        {s.enterprise_verified && <span style={{ color: '#34c759' }}>Enterprise verified</span>}
        {s.status === 'active' && (
          <span className="live-dot inline-flex items-center gap-1" style={{ color: '#0071e3' }}>
            <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: '#0071e3' }} />
            Live
          </span>
        )}
      </div>
    </button>
  )
}
