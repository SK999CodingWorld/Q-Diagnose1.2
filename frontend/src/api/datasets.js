import { apiFetch } from './client';

export const datasetsApi = {
  async list(options) {
    return (await apiFetch('/datasets', options)).json();
  },
  async upload(formData, options) {
    return apiFetch('/datasets/upload', { method: 'POST', body: formData, ...options });
  },
};
