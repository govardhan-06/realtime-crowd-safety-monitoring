CREATE TABLE IF NOT EXISTS m5_sources (
    source_id TEXT PRIMARY KEY,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS m5_runs (
    run_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES m5_sources(source_id),
    config_hash TEXT NOT NULL,
    deterministic_hash TEXT NOT NULL,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS m5_incidents (
    incident_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES m5_runs(run_id),
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS m5_transitions (
    transition_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES m5_runs(run_id),
    incident_id TEXT NOT NULL REFERENCES m5_incidents(incident_id),
    sequence INTEGER NOT NULL,
    payload JSONB NOT NULL,
    UNIQUE (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS m5_timelines (
    timeline_id BIGSERIAL PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES m5_incidents(incident_id),
    sequence INTEGER NOT NULL,
    payload JSONB NOT NULL,
    UNIQUE (incident_id, sequence)
);

CREATE TABLE IF NOT EXISTS m5_evidence (
    evidence_id BIGSERIAL PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES m5_incidents(incident_id),
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS m5_explanations (
    incident_id TEXT PRIMARY KEY REFERENCES m5_incidents(incident_id),
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS m5_actions (
    sequence BIGSERIAL PRIMARY KEY,
    action_id TEXT UNIQUE NOT NULL,
    incident_id TEXT NOT NULL REFERENCES m5_incidents(incident_id),
    action TEXT NOT NULL CHECK (action IN ('acknowledge', 'dismiss', 'escalate')),
    actor TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    note TEXT
);
