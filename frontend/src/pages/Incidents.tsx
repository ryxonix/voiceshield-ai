import { useEffect, useState } from 'react'
import { Breadcrumbs, PageHeader, Divider, apiBase, btn } from '../components/ui'
import { bandColor } from '../components/RiskGauge'

export default function Incidents() {
  const [incidents, setIncidents] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${apiBase()}/api/incidents`)
      .then((r) => r.json())
      .then(setIncidents)
      .catch(() => setIncidents([]))
      .finally(() => setLoading(false))
  }, [])

  const ack = async (id: string) => {
    await fetch(`${apiBase()}/api/incidents/${id}/ack`, { method: 'POST' })
    setIncidents((xs) => xs.map((x) => (x.id === id ? { ...x, acknowledged: true } : x)))
  }

  return (
    <article>
      <Breadcrumbs trail={['Risk & incidents', 'Incident log']} />
      <PageHeader
        title="Incident log"
        meta="Threshold crossings recorded during live monitoring. Each incident carries a downloadable I4C-format forensic report with IST timestamps."
        actions={
          <button className={btn()} onClick={() => location.reload()}>
            Refresh
          </button>
        }
      />
      <Divider />

      {loading ? (
        <p className="text-[14px] text-zinc-500">Loading…</p>
      ) : incidents.length === 0 ? (
        <div className="card p-8 text-center">
          <p className="font-serif text-[19px] font-semibold text-zinc-900">No incidents recorded</p>
          <p className="mx-auto mt-1.5 max-w-md text-[13.5px] leading-relaxed text-zinc-500">
            Incidents appear when a live session crosses its role threshold — 70% for child-safety calls, 85% for
            adults. Run the live monitor to see the full mitigation flow.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {incidents.map((i) => (
            <div key={i.id} className="card p-5 transition-colors hover:border-zinc-300">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="flex flex-wrap items-center gap-2.5">
                    <span
                      className="rounded-full px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wide"
                      style={{ background: `${bandColor(i.severity)}1a`, color: bandColor(i.severity) }}
                    >
                      {i.severity}
                    </span>
                    <span className="font-mono text-[12.5px] text-zinc-500">{i.id}</span>
                    <span className="text-[12.5px] text-zinc-500">
                      {new Date(i.created_at).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })}
                    </span>
                  </div>
                  <p className="mt-2 text-[14.5px] text-zinc-800">
                    Synthetic voice detected on call{' '}
                    <span className="font-mono text-[13px] text-zinc-600">{i.session_id}</span>
                    <span className="text-zinc-500"> · {i.role} · {i.language.toUpperCase()}</span>
                  </p>
                  <p className="mt-1 text-[13px] text-zinc-500">
                    Triggers: {(i.triggers || []).join(', ') || '—'}
                    {i.speaker_mismatch && ' · speaker mismatch'}
                  </p>
                </div>
                <div className="flex flex-col items-end gap-2">
                  <span className="font-serif text-[26px] font-bold leading-none" style={{ color: bandColor(i.severity) }}>
                    {(i.score * 100).toFixed(0)}%
                  </span>
                  <div className="flex items-center gap-3">
                    <a
                      href={`${apiBase()}/api/incidents/${i.id}/report`}
                      target="_blank"
                      rel="noreferrer"
                      className="text-[13px] font-medium text-zinc-900 underline decoration-[#E4E4E7] underline-offset-2 transition-colors hover:decoration-zinc-500"
                    >
                      I4C PDF
                    </a>
                    {!i.acknowledged ? (
                      <button
                        onClick={() => ack(i.id)}
                        className="text-[13px] font-medium text-zinc-900 underline decoration-[#E4E4E7] underline-offset-2 transition-colors hover:decoration-zinc-500"
                      >
                        Acknowledge
                      </button>
                    ) : (
                      <span className="text-[12.5px] text-zinc-400">Acknowledged ✓</span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </article>
  )
}
