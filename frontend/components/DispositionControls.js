'use client';

import { useState } from 'react';

const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

export default function DispositionControls({ incidentId }) {
  const [actor, setActor] = useState('operator');
  const [note, setNote] = useState('');
  const [pending, setPending] = useState('');
  const [message, setMessage] = useState('');
  async function submit(action) {
    setPending(action); setMessage('');
    try {
      const response = await fetch(`${baseUrl}/incidents/${encodeURIComponent(incidentId)}/${action}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ actor, timestamp: new Date().toISOString(), note: note || null }) });
      if (!response.ok) throw new Error(`Action failed (${response.status})`);
      setMessage(`${action} recorded.`);
      window.location.reload();
    } catch (error) { setMessage(error.message); }
    finally { setPending(''); }
  }
  return <div className="actions"><label htmlFor="actor">Operator</label><input id="actor" value={actor} onChange={(event) => setActor(event.target.value)} /><label htmlFor="note">Note <span className="muted">(optional)</span></label><textarea id="note" rows="2" value={note} onChange={(event) => setNote(event.target.value)} /><div className="action-buttons">{['acknowledge', 'dismiss', 'escalate'].map((action) => <button className={`button ${action === 'escalate' ? 'danger' : 'secondary'}`} disabled={Boolean(pending) || !actor.trim()} onClick={() => submit(action)} key={action}>{pending === action ? 'Saving…' : action}</button>)}</div>{message && <p role="status" className="action-message">{message}</p>}</div>;
}
