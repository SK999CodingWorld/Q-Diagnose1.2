import React, { useEffect, useState } from 'react';
import { AlertTriangle, X } from 'lucide-react';

export default function ApiErrorNotice() {
  const [error, setError] = useState(null);

  useEffect(() => {
    const handleError = event => setError(event.detail);
    window.addEventListener('qdiagnose:api-error', handleError);
    return () => window.removeEventListener('qdiagnose:api-error', handleError);
  }, []);

  if (!error) return null;
  return (
    <div className="fixed bottom-4 right-4 z-[100] flex max-w-[min(92vw,28rem)] items-start gap-3 rounded-xl border border-rose-400/30 bg-[#1a1320] px-4 py-3 text-sm text-rose-100 shadow-2xl" role="alert">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-300" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <p className="font-semibold">Request could not be completed</p>
        <p className="mt-1 break-words text-xs text-rose-100/80">{error.message}</p>
        {error.requestId && <p className="mt-1 font-mono text-[10px] text-rose-200/60">Request ID: {error.requestId}</p>}
      </div>
      <button type="button" onClick={() => setError(null)} aria-label="Dismiss error" className="rounded p-1 text-rose-200/70 hover:bg-white/10 hover:text-white">
        <X className="h-4 w-4" aria-hidden="true" />
      </button>
    </div>
  );
}
