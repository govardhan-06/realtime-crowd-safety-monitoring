export function incidentListState({ loading = false, data, error = null }) {
  if (loading) return { status: 'loading' };
  if (error) return { status: 'error', message: error };
  if (!data || data.length === 0) return { status: 'empty' };
  return { status: 'ready', items: data };
}

export function incidentDetailState({ incident, error = null }) {
  if (error) return { status: 'error', message: error };
  if (!incident) return { status: 'loading' };
  return {
    status: 'ready',
    deterministicReasons: incident.deterministic?.reason_codes ?? [],
    explanationStatus: incident.explanation?.status ?? 'disabled',
  };
}
