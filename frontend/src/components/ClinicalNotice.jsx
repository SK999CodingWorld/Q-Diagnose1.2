import React from 'react';
import { ShieldAlert } from 'lucide-react';

export default function ClinicalNotice() {
  return (
    <div className="clinical-notice flex items-center justify-center gap-2 border-b px-4 py-2.5 text-center text-xs font-medium" role="note">
      <ShieldAlert className="h-4 w-4 shrink-0" aria-hidden="true" />
      <span><strong className="font-semibold">Research use only.</strong> Model results have not been validated to guide patient care.</span>
    </div>
  );
}
