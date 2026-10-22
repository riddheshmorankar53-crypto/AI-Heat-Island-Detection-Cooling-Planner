/**
 * API Service for Urban Cooling Planner.
 * Communicates with FastAPI backend with robust error handling and fallbacks.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

async function handleResponse(res) {
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`API error (${res.status}): ${errorText || res.statusText}`);
  }
  return res.json();
}

export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return handleResponse(res);
}

export async function fetchAOI() {
  const res = await fetch(`${BASE_URL}/aoi`);
  return handleResponse(res);
}

export async function fetchHotspotsSummary() {
  const res = await fetch(`${BASE_URL}/hotspots/summary`);
  return handleResponse(res);
}

export async function fetchGrid(params = {}) {
  const query = new URLSearchParams();
  if (params.min_lst !== undefined) query.append('min_lst', params.min_lst);
  if (params.max_lst !== undefined) query.append('max_lst', params.max_lst);
  if (params.hotspot_only) query.append('hotspot_only', 'true');
  if (params.limit) query.append('limit', params.limit);

  const url = `${BASE_URL}/grid${query.toString() ? `?${query.toString()}` : ''}`;
  const res = await fetch(url);
  return handleResponse(res);
}

export async function fetchInterventionCatalog() {
  const res = await fetch(`${BASE_URL}/interventions/catalog`);
  return handleResponse(res);
}

export async function optimizeInterventions(payload) {
  const res = await fetch(`${BASE_URL}/interventions/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}

export async function fetchRoutingLandmarks() {
  const res = await fetch(`${BASE_URL}/routing/landmarks`);
  return handleResponse(res);
}

export async function calculateHeatSafeRoute(payload) {
  const res = await fetch(`${BASE_URL}/routing/heat-safe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}

export async function fetchHeatAlerts(limit = 20) {
  const res = await fetch(`${BASE_URL}/alerts?limit=${limit}`);
  return handleResponse(res);
}

export async function fetchMLMetrics() {
  const res = await fetch(`${BASE_URL}/ml/metrics`);
  return handleResponse(res);
}
