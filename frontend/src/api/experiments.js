import { apiFetch } from './client';

export const experimentsApi = {
  async list(datasetId, options) {
    const query = datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : '';
    return (await apiFetch(`/experiments${query}`, options)).json();
  },
  robustness: () => apiFetch('/experiments/robustness-study'),
  metricsAtThreshold: (experimentId, threshold) => apiFetch(`/experiments/${experimentId}/metrics-at-threshold?threshold=${encodeURIComponent(threshold)}`),
  explain: experimentId => apiFetch(`/explain/${experimentId}`),
  run: values => apiFetch('/experiments/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(values),
  }),
  comparisons: () => apiFetch('/comparisons'),
};
