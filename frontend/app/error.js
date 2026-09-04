'use client';

export default function Error({ reset }) {
  return <main className="shell"><section className="panel"><p className="eyebrow">Connection issue</p><h1>Review data is unavailable</h1><p className="muted">The safety API did not return a usable response.</p><button className="button primary" onClick={reset}>Try again</button></section></main>;
}
