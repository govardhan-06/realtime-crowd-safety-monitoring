const baseUrl = process.env.CROWD_API_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

async function request(path, options = {}) {
  const response = await fetch(`${baseUrl}${path}`, { ...options, cache: 'no-store' });
  if (!response.ok) throw new Error(`API request failed (${response.status})`);
  return response.json();
}

export const getIncidents = () => request('/incidents');
export const getHealth = () => request('/health');
export const getIncident = (id) => request(`/incidents/${encodeURIComponent(id)}`);
export const getSources = () => request('/sources');
export const getRuns = () => request('/runs');
