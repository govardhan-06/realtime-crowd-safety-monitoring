import Link from 'next/link';
import { getHealth, getIncidents, getRuns, getSources } from '../../lib/api';

function formatTime(value) {
  return value == null ? '—' : `${Number(value).toFixed(1)}s`;
}

export default async function IncidentsPage() {
  let incidents, sources, runs, health;
  try {
    const [incidentData, sourceData, runData, healthData] = await Promise.all([getIncidents(), getSources(), getRuns(), getHealth()]);
    incidents = incidentData;
    sources = sourceData;
    runs = runData;
    health = healthData;
  } catch {
    return <main className="shell"><section className="panel"><p className="eyebrow">Connection issue</p><h1>Incident feed unavailable</h1><p className="muted">Start the local API and reload this page.</p></section></main>;
  }
  return <main className="shell">
    <header className="masthead"><div><p className="eyebrow">Human review console / M5</p><h1>Detected indicators</h1><p className="lede">One evolving incident per source and zone. Review the evidence before choosing an action.</p></div><div className="live-mark"><span aria-hidden="true" /> Review API {health.status === 'ok' ? 'online' : 'unavailable'}</div></header>
    <section className="status-strip" aria-label="System status"><div><span className="label">Sources</span><strong>{sources.length ? `${sources.length} available` : 'No sources'}</strong></div><div><span className="label">Latest run</span><strong>{runs.length ? runs[0].run_id : 'No imported run'}</strong></div><div><span className="label">Input mode</span><strong>Offline video</strong></div><div><span className="label">VLM explanation</span><strong>Disabled</strong></div></section>
    <section className="panel" aria-labelledby="incident-heading"><div className="section-heading"><div><p className="eyebrow">Authoritative records</p><h2 id="incident-heading">Recent incidents</h2></div><span className="count">{incidents.length} records</span></div>
      {incidents.length === 0 ? <div role="status" className="empty"><h3>No incidents to review</h3><p>Process an authorised local video to create a review record.</p></div> : <div className="incident-list" role="list">{incidents.map((item) => <Link className="incident-row" role="listitem" href={`/incidents/${item.record_id || item.incident.incident_id}`} key={item.record_id || item.incident.incident_id}><div className="severity-bar" data-severity={item.incident.severity} /><div className="incident-main"><div className="row-top"><strong>{item.incident.severity} indicator</strong><span className="state">{item.incident.state}</span></div><p>{item.incident.source_id} / {item.incident.region_id}</p><div className="reason-line">{item.reason_codes.join(' · ') || 'No reason codes recorded'}</div></div><div className="incident-meta"><span>Peak {Number(item.incident.peak_risk).toFixed(2)}</span><span>{formatTime(item.incident.last_updated_at_s)}</span><span>{item.evidence_available ? 'Evidence ready' : 'Evidence unavailable'}</span></div><span className="arrow" aria-hidden="true">→</span></Link>)}</div>}
    </section>
    <p className="footnote">System reasons and lifecycle state are deterministic. Generated text, when enabled, is supplementary and never changes incident decisions.</p>
  </main>;
}
