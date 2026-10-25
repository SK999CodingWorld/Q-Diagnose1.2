/** API origin is configured outside components; same-origin is the production default. */
const configuredOrigin = import.meta.env.VITE_API_BASE_URL?.trim();
const developmentOrigin = window.location.port === '5173' ? 'http://127.0.0.1:8000' : '';

export const API_BASE = (configuredOrigin || developmentOrigin).replace(/\/$/, '');
export const API_PREFIX = '/api/v1';
