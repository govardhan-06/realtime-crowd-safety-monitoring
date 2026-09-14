'use client';

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
const stages = ['ingestion', 'detection/tracking', 'crowd signals', 'violence', 'fusion', 'incident'];

export default function ProcessPage() {
  const [file, setFile] = useState(null);
  const [runId, setRunId] = useState(null);
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!runId) return undefined;
    const socket = new WebSocket(`${apiBase.replace(/^http/, 'ws')}/runs/${runId}/stream`);
    setStatus('connecting');
    socket.onopen = () => setStatus('connected');
    socket.onmessage = ({ data }) => {
      const event = JSON.parse(data);
      setEvents((current) => [...current, event]);
      if (event.type === 'status') setStatus(event.state);
      if (event.type === 'complete') setStatus('completed');
      if (event.type === 'error') { setStatus('failed'); setError(event.message); }
      if (event.type === 'frame' && canvasRef.current) {
        const image = new Image();
        image.onload = () => {
          const canvas = canvasRef.current;
          canvas.width = image.width;
          canvas.height = image.height;
          canvas.getContext('2d').drawImage(image, 0, 0);
        };
        image.src = `data:image/jpeg;base64,${event.jpeg_base64}`;
      }
    };
    socket.onerror = () => { setStatus('connection error'); setError('The processing stream could not be reached.'); };
    return () => socket.close();
  }, [runId]);

  async function start() {
    if (!file) return;
    setError(''); setEvents([]); setStatus('uploading');
    try {
      const response = await fetch(`${apiBase}/runs`, { method: 'POST', body: file, headers: { 'content-type': file.type || 'video/mp4', 'x-filename': file.name } });
      if (!response.ok) throw new Error(`Upload failed (${response.status})`);
      setRunId((await response.json()).run_id);
    } catch (reason) { setStatus('failed'); setError(reason.message); }
  }

  const latest = (type) => [...events].reverse().find((event) => event.type === type);
  const incidents = events.filter((event) => event.type === 'incident');
  const frame = latest('frame');
  const stageStatus = (stage) => {
    const names = { ingestion: [], 'detection/tracking': ['detector', 'tracker'], 'crowd signals': ['crowd_features'], violence: ['violence'], fusion: ['fusion'], incident: ['incident'] };
    if (stage === 'ingestion') return status === 'idle' ? 'waiting' : status;
    const event = [...events].reverse().find((item) => item.type === 'stage' && names[stage].includes(item.stage));
    return event ? `${event.status}${event.detail ? ` — ${event.detail}` : ''}` : 'waiting';
  };

  return <main className="shell">
    <Link className="back" href="/incidents">← Review incidents</Link>
    <header className="masthead"><div><p className="eyebrow">Near-real-time processing</p><h1>Watch the evidence arrive.</h1><p className="lede">Upload an authorised video, then start processing. Annotated frames and alert signals appear while the final offline artifacts are produced.</p></div><div className="live-mark"><span aria-hidden="true" /> {status}</div></header>
    <section className="process-grid">
      <div className="panel"><p className="eyebrow">Input</p><h2>Upload video</h2><input aria-label="Video file" type="file" accept="video/*" onChange={(event) => setFile(event.target.files?.[0] || null)} /><button className="button primary process-button" disabled={!file || ['uploading', 'processing', 'connecting'].includes(status)} onClick={start}>Start processing</button>{file && <p className="muted">{file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB</p>}{error && <p role="alert" className="action-message">{error}</p>}</div>
      <div className="panel viewer-panel"><div className="section-heading"><div><p className="eyebrow">Annotated viewer</p><h2>Current frame</h2></div><span className="state">Near-real-time</span></div><div className="canvas-wrap">{frame ? <canvas ref={canvasRef} aria-label="Latest annotated video frame" /> : <p className="muted">Frames will appear here before processing completes.</p>}</div><div className="viewer-stats"><span>Source time <strong>{frame ? `${Number(frame.timestamp_s).toFixed(1)}s` : '—'}</strong></span><span>Frame <strong>{frame?.frame_index ?? '—'}</strong></span><span>Lag <strong>{frame ? `${Number(frame.lag_s).toFixed(2)}s` : '—'}</strong></span><span>Connection <strong>{status}</strong></span></div>{runId && status === 'completed' && <a className="button secondary" href={`${apiBase}/runs/${runId}/artifact`}>Download annotated.mp4</a>}</div>
    </section>
    <section className="panel process-panel"><p className="eyebrow">Alerting path</p><h2>Stage health</h2><div className="stage-list">{stages.map((stage) => <div className="stage-row" key={stage}><strong>{stage}</strong><span>{stageStatus(stage)}</span></div>)}</div><h2 className="timeline-title">Incident timeline</h2>{incidents.length ? <ol className="timeline">{incidents.map((event, index) => <li key={`${event.incident_id}-${event.timestamp_s}-${index}`}><div className="timeline-dot" /><div><strong>{event.state} · {event.severity}</strong><p>{Number(event.timestamp_s).toFixed(1)}s · risk {Number(event.risk).toFixed(2)} · {event.reason_codes.join(' · ') || 'No reason codes'}</p></div></li>)}</ol> : <p className="muted">No lifecycle transitions have been emitted. Human review remains required before escalation.</p>}</section>
    <p className="footnote">Model unavailable or degraded states are evidence-health states, not ordinary negative evidence. Escalation is always a human action.</p>
  </main>;
}
