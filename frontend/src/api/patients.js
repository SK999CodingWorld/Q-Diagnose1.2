import { apiFetch } from './client';

const json = (method, body) => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const patientsApi = {
  list: () => apiFetch('/patients'),
  get: patientId => apiFetch(`/patients/${encodeURIComponent(patientId)}`),
  trend: patientId => apiFetch(`/patients/${encodeURIComponent(patientId)}/trend`),
  create: values => apiFetch('/patients', json('POST', values)),
  addRecord: (patientId, values) => apiFetch(`/patients/${encodeURIComponent(patientId)}/records`, json('POST', values)),
  predict: (patientId, values) => apiFetch(`/patients/${encodeURIComponent(patientId)}/predict`, json('POST', values)),
};
