import { API_BASE, API_PREFIX } from './config';
import { supabase } from '../auth/supabase';

export class ApiError extends Error {
  constructor(message, { status = 0, requestId = null, cause = null } = {}) {
    super(message, { cause });
    this.name = 'ApiError';
    this.status = status;
    this.requestId = requestId;
  }
}

function resolveUrl(path) {
  if (/^https?:\/\//i.test(path)) {
    const parsed = new URL(path);
    if (parsed.pathname.startsWith('/api/v1/')) return path;
    const normalizedPath = parsed.pathname.startsWith('/api/')
      ? `${API_PREFIX}/${parsed.pathname.slice('/api/'.length)}`
      : `${API_PREFIX}${parsed.pathname}`;
    return `${parsed.origin}${normalizedPath}${parsed.search}${parsed.hash}`;
  }

  const normalizedPath = path.startsWith('/api/')
    ? `${API_PREFIX}/${path.slice('/api/'.length)}`
    : `${API_PREFIX}${path.startsWith('/') ? path : `/${path}`}`;
  return `${API_BASE}${normalizedPath}`;
}

function publishError(error) {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('qdiagnose:api-error', {
      detail: { message: error.message, status: error.status, requestId: error.requestId },
    }));
  }
}

/** Send an API request, normalize legacy paths, and surface actionable failures. */
export async function apiFetch(path, options = {}) {
  try {
    const { data: { session } = {} } = supabase
      ? await supabase.auth.getSession()
      : { data: {} };
    const headers = { Accept: 'application/json', ...options.headers };
    if (session?.access_token) headers.Authorization = `Bearer ${session.access_token}`;
    const response = await fetch(resolveUrl(path), {
      ...options,
      headers,
    });
    if (!response.ok) {
      let payload = null;
      try { payload = await response.clone().json(); } catch { /* Non-JSON proxy errors are expected. */ }
      const detail = payload?.detail;
      const message = typeof detail === 'string'
        ? detail
        : `The request failed (${response.status} ${response.statusText}).`;
      const error = new ApiError(message, {
        status: response.status,
        requestId: payload?.request_id || response.headers.get('X-Request-ID'),
      });
      publishError(error);
      throw error;
    }
    return response;
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause;
    if (cause instanceof ApiError) throw cause;
    const error = new ApiError('Could not reach the Q-Diagnose API. Check that the backend is running and try again.', { cause });
    publishError(error);
    throw error;
  }
}

