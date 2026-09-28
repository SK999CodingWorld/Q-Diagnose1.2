import type {
  Calibration,
  Circuit,
  ComparisonResponse,
  Dataset,
  Explanation,
  Experiment,
  Health,
  SweepResponse,
} from './types'

const request = async <T>(path: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(path, init)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
    throw new Error(detail || `Request failed with status ${response.status}`)
  }
  return body as T
}

export const api = {
  health: () => request<Health>('/api/health'),
  datasets: () => request<Dataset[]>('/api/datasets'),
  experiments: () => request<Experiment[]>('/api/experiments'),
  experiment: (id: number) => request<Experiment>(`/api/experiments/${id}`),
  createBuiltinDataset: () => request<Dataset>('/api/datasets/builtin', { method: 'POST' }),
  createDiabetesDataset: () => request<Dataset>('/api/datasets/builtin/diabetes', { method: 'POST' }),
  createExperiment: (payload: Record<string, unknown>) =>
    request<Experiment>('/api/experiments', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  explanation: (id: number) => request<Explanation>(`/api/experiments/${id}/explanation`),
  circuit: (id: number) => request<Circuit>(`/api/experiments/${id}/circuit`),
  calibration: (id: number) => request<Calibration>(`/api/experiments/${id}/calibration`),
  recalibrate: (id: number) =>
    request<Calibration>(`/api/experiments/${id}/calibrate`, { method: 'POST' }),
  sweep: (payload: Record<string, unknown>) =>
    request<SweepResponse>('/api/sweeps', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  crossDatasetComparison: (payload: Record<string, unknown>) =>
    request<ComparisonResponse>('/api/phase5/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  upload: (file: File, targetColumn: string) => {
    const body = new FormData()
    body.append('file', file)
    body.append('target_column', targetColumn)
    return request<Dataset>('/api/datasets/upload', { method: 'POST', body })
  },
}
