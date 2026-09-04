import test from 'node:test';
import assert from 'node:assert/strict';
import { incidentListState, incidentDetailState } from './view-model.mjs';

test('incident list models loading, empty, error, and data states', () => {
  assert.equal(incidentListState({ loading: true }).status, 'loading');
  assert.equal(incidentListState({ data: [] }).status, 'empty');
  assert.equal(incidentListState({ error: 'API unavailable' }).status, 'error');
  assert.equal(incidentListState({ data: [{ incident: { incident_id: 'i1' } }] }).status, 'ready');
});

test('incident detail keeps deterministic evidence and explanation separate', () => {
  const state = incidentDetailState({
    incident: { incident: { incident_id: 'i1' }, deterministic: { reason_codes: ['violence_high'] }, explanation: { status: 'disabled' } },
  });
  assert.equal(state.status, 'ready');
  assert.deepEqual(state.deterministicReasons, ['violence_high']);
  assert.equal(state.explanationStatus, 'disabled');
});
