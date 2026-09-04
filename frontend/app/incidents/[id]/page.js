import Link from 'next/link';
import { getIncident } from '../../../lib/api';
import DispositionControls from '../../../components/DispositionControls';

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

function EvidenceArtifact({ incidentId, artifact }) {
  if (artifact.status !== 'available') {
    return <div className="evidence-row"><span>{artifact.kind}</span><strong data-status={artifact.status}>{artifact.status}</strong>{artifact.detail && <small>{artifact.detail}</small>}</div>;
  }
  const path = `${apiBase}/incidents/${incidentId}/evidence/${artifact.kind}`;
  return <div className="evidence-row"><span>{artifact.kind}</span><strong data-status={artifact.status}>{artifact.status}</strong>{artifact.kind === 'snapshot' ? <img className="evidence-media" src={path} alt="Captured incident frame" /> : <video className="evidence-media" controls preload="metadata"><source src={path} type="video/mp4" /></video>}</div>;
}

export default async function IncidentPage({ params }) {
  const { id } = await params;
  let record;
  try {
    record = await getIncident(id);
  } catch {
    return <main className="shell"><Link className="back" href="/incidents">← All incidents</Link><section className="panel"><p className="eyebrow">Unavailable</p><h1>Incident could not be loaded</h1><p className="muted">Check the incident ID and local API.</p></section></main>;
  }
  const incident = record.incident;
  const recordId = record.record_id || incident.incident_id;
  return <main className="shell">
    <Link className="back" href="/incidents">← All incidents</Link>
    <header className="detail-head"><div><p className="eyebrow">{incident.source_id} / {incident.region_id}</p><h1>{incident.severity} indicator</h1><p className="lede">{incident.state} · peak fused risk {Number(incident.peak_risk).toFixed(2)}</p></div><div className="decision-badge" data-severity={incident.severity}>{incident.state}</div></header>
    <div className="detail-grid">
      <section className="panel"><p className="eyebrow">Why the system alerted</p><h2>Deterministic evidence</h2><div className="reason-grid">{(record.deterministic.reason_codes || []).map((reason) => <span className="reason-chip" key={reason}>{reason}</span>)}</div><h3>Signal timeline</h3><ol className="timeline">{record.deterministic.timeline.length ? record.deterministic.timeline.map((point, index) => <li key={`${point.timestamp_s}-${index}`}><span className="timeline-dot" /><div><strong>{Number(point.timestamp_s).toFixed(1)}s</strong><p>Fused risk {Number(point.fused_risk).toFixed(2)}</p></div></li>) : <li className="muted">No signal timeline was stored.</li>}</ol></section>
      <aside className="panel review-panel"><p className="eyebrow">Human disposition</p><h2>Review and record</h2><p className="muted">Actions are appended to the audit history. Escalate records an internal human request only; it does not contact emergency services.</p><DispositionControls incidentId={recordId} /><div className="audit"><h3>Audit history</h3>{record.actions.length ? record.actions.map((action) => <div className="audit-row" key={action.action_id}><strong>{action.action}</strong><span>{action.actor}</span><time>{action.timestamp}</time>{action.note && <small>{action.note}</small>}</div>) : <p className="muted">No actions recorded.</p>}</div></aside>
    </div>
    <section className="panel evidence-panel"><div><p className="eyebrow">Captured review material</p><h2>Evidence references</h2></div>{record.deterministic.evidence.length ? record.deterministic.evidence.map((manifest) => <div className="evidence-manifest" key={manifest.incident_id}><p>{manifest.run_id ? `Run ${manifest.run_id} · bounded ${manifest.pre_event_s ?? '—'}s before / ${manifest.post_event_s ?? '—'}s after` : 'Evidence manifest imported without run metadata.'}</p>{manifest.artifacts?.length ? manifest.artifacts.map((artifact) => <EvidenceArtifact incidentId={recordId} artifact={artifact} key={artifact.kind} />) : <p role="status" className="muted">No captured media is available for this incident. Deterministic incident data remains reviewable.</p>}{Object.keys(manifest.stage_health || {}).length > 0 && <><h3>Stage health</h3>{Object.entries(manifest.stage_health).map(([stage, health]) => <div className="evidence-row" key={stage}><span>{stage}</span><strong data-status={health.status}>{health.status}</strong></div>)}</>}</div>) : <p className="muted">No evidence manifest was imported.</p>}</section>
    <section className="panel generated-panel"><p className="eyebrow">Supplementary layer</p><h2>AI-generated explanation</h2><p className="muted">This text is labelled separately and is not used to create, close, escalate, or change severity.</p><div className="generated-state" data-status={record.explanation.status}><strong>{record.explanation.status}</strong>{record.explanation.text && <p>{record.explanation.text}</p>}{record.explanation.detail && <p className="muted">{record.explanation.detail}</p>}</div></section>
  </main>;
}
