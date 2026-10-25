import { apiFetch } from './client';

const json = (method, body) => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const labReportsApi = {
  presets: () => apiFetch('/lab-reports/presets'),
  analyze: values => apiFetch('/lab-reports/analyze', json('POST', values)),
  upload: formData => apiFetch('/lab-reports/upload', { method: 'POST', body: formData }),
  saveToPatient: values => apiFetch('/lab-reports/save-to-patient', json('POST', values)),
};

export const patientReportsApi = {
  upload: formData => apiFetch('/reports/upload', { method: 'POST', body: formData }),
  confirm: (reportId, values) => apiFetch(`/reports/${encodeURIComponent(reportId)}/confirm`, json('PUT', values)),
  predictRisk: values => apiFetch('/risk/predict', json('POST', values)),
  exportPdf: reportId => apiFetch(`/reports/${encodeURIComponent(reportId)}/export-pdf`),
};
