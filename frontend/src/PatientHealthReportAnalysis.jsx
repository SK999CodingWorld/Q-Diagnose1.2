import React, { useState } from 'react';
import { AlertCircle, ArrowLeft, ArrowRight, Check, CheckCircle2, FileText, RotateCcw, UploadCloud, X } from 'lucide-react';
import { API_BASE } from './api/config';
import { apiFetch } from './api/client';

const STEPS = ['Upload report', 'Check values', 'Saved summary'];

function toEditableMarker(marker, index) {
  return {
    key: marker.key || `measurement_${index + 1}`,
    label: marker.label || marker.key || `Measurement ${index + 1}`,
    value: marker.value == null ? '' : String(marker.value),
    unit: marker.unit || '',
    status: marker.status || '',
    badge: marker.badge || '',
    referenceRange: marker.reference_range || '',
    category: marker.category || 'General',
  };
}

function makeBiomarkerKey(label) {
  return label.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
}

async function getErrorMessage(response, fallback) {
  try {
    const body = await response.json();
    return body.detail || body.message || fallback;
  } catch {
    return fallback;
  }
}

export default function PatientHealthReportAnalysis() {
  const [step, setStep] = useState(1);
  const [files, setFiles] = useState([]);
  const [markers, setMarkers] = useState([]);
  const [reviewFlags, setReviewFlags] = useState([]);
  const [reportId, setReportId] = useState('');
  const [manualDraft, setManualDraft] = useState({ label: '', value: '', unit: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [privacyConfirmed, setPrivacyConfirmed] = useState(false);

  const updateMarker = (index, field, value) => {
    const changedMarker = markers[index];
    if (changedMarker) setReviewFlags(flags => flags.filter(flag => flag.key !== changedMarker.key));
    setMarkers(current => current.map((marker, currentIndex) => (
      currentIndex === index ? { ...marker, [field]: value } : marker
    )));
  };

  const addManualMarker = () => {
    const label = manualDraft.label.trim();
    const value = Number(manualDraft.value);
    if (!label || manualDraft.value === '' || !Number.isFinite(value)) {
      setError('Enter a measurement name and a valid number.');
      return;
    }
    const key = makeBiomarkerKey(label);
    if (!key) {
      setError('Use a name with letters or numbers.');
      return;
    }
    setMarkers(current => [
      ...current.filter(marker => marker.key !== key),
      { key, label, value: manualDraft.value, unit: manualDraft.unit.trim() },
    ]);
    setReviewFlags(current => current.filter(flag => flag.key !== key));
    setManualDraft({ label: '', value: '', unit: '' });
    setError('');
  };

  const removeMarker = (index) => {
    const removedKey = markers[index]?.key;
    setMarkers(current => current.filter((_, currentIndex) => currentIndex !== index));
    setReviewFlags(current => current.filter(flag => flag.key !== removedKey));
  };

  const handleUpload = async () => {
    if (!files.length) {
      setError('Choose at least one report file first.');
      return;
    }
    if (!privacyConfirmed) {
      setError('Confirm that this workspace is approved for the information in these files.');
      return;
    }

    setError('');
    setLoading(true);
    try {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      const response = await apiFetch(`${API_BASE}/api/reports/upload`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(await getErrorMessage(response, 'Could not read these files.'));
      const data = await response.json();
      setReportId(data.report_id || '');
      const extracted = data.extracted_biomarkers || [];
      setMarkers(extracted.map(toEditableMarker));
      setReviewFlags(extracted);
      setStep(2);
    } catch (uploadError) {
      setError(uploadError.message || 'Could not read these files. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveReview = async () => {
    const validMarkers = markers.filter(marker => (
      marker.key && marker.label.trim() && marker.value !== '' && Number.isFinite(Number(marker.value))
    ));
    if (!validMarkers.length) {
      setError('Add or correct at least one measurement before saving.');
      return;
    }
    if (!reportId) {
      setError('The report session is missing. Upload the report again to continue.');
      return;
    }

    setError('');
    setLoading(true);
    try {
      const biomarkers = Object.fromEntries(validMarkers.map(marker => [marker.key, Number(marker.value)]));
      const response = await apiFetch(`${API_BASE}/api/reports/${encodeURIComponent(reportId)}/confirm`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ biomarkers }),
      });
      if (!response.ok) throw new Error(await getErrorMessage(response, 'Could not save the reviewed values.'));
      const data = await response.json();
      setMarkers(validMarkers);
      setReviewFlags(data.flagged_biomarkers || []);
      setStep(3);
    } catch (saveError) {
      setError(saveError.message || 'Could not save the reviewed values. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const startNewReview = () => {
    setFiles([]);
    setMarkers([]);
    setReviewFlags([]);
    setReportId('');
    setError('');
    setPrivacyConfirmed(false);
    setManualDraft({ label: '', value: '', unit: '' });
    setStep(1);
  };

  return (
    <section className="report-workspace mx-auto w-full max-w-5xl space-y-5" aria-label="Report review">
      <header className="report-page-heading">
        <p className="analysis-eyebrow">PATIENT REPORT WORKFLOW</p>
        <h2>Review a blood or lab report</h2>
        <p>Upload a file, check the extracted values, and save a verified summary. This tool does not predict disease.</p>
      </header>
      <div className="report-workflow-card rounded-2xl border p-5 shadow-sm sm:p-7">
        <div className="report-stepper mb-6 grid grid-cols-3 gap-2" aria-label="Report review steps">
          {STEPS.map((title, index) => {
            const stepNumber = index + 1;
            const complete = step > stepNumber;
            const current = step === stepNumber;
            return (
              <div key={title} className={`report-step ${current ? 'is-current' : ''} ${complete ? 'is-complete' : ''}`} aria-current={current ? 'step' : undefined}>
                <span className="report-step-number">{complete ? <Check className="h-3.5 w-3.5" aria-hidden="true" /> : stepNumber}</span>
                <span>{title}</span>
              </div>
            );
          })}
        </div>

        {step === 1 && (
          <div className="space-y-5">
            <div>
              <h3 className="text-lg font-semibold">Upload a report</h3>
              <p className="report-muted mt-1 text-sm">We’ll find supported measurement names and values. Check each one against the original report before saving.</p>
            </div>

            <label className="report-upload-zone flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center transition-colors sm:p-10">
              <span className="report-upload-icon mb-3 flex h-12 w-12 items-center justify-center rounded-2xl"><UploadCloud className="h-6 w-6" aria-hidden="true" /></span>
              <span className="text-sm font-semibold">Choose report files</span>
              <span className="report-muted mt-1 text-xs">PDF, DOCX, XLSX, CSV, TXT, JSON, or scanned image · multiple files allowed</span>
              <input type="file" accept=".pdf,.docx,.xlsx,.xls,.png,.jpg,.jpeg,.txt,.csv,.json" multiple className="sr-only" onChange={event => { setFiles(Array.from(event.target.files || [])); setError(''); }} />
            </label>

            {files.length > 0 && (
              <ul className="report-file-list divide-y rounded-xl border" aria-label="Selected files">
                {files.map((file, index) => (
                  <li key={`${file.name}-${index}`} className="flex items-center gap-3 px-3 py-2.5 text-sm">
                    <FileText className="report-icon h-4 w-4 shrink-0" aria-hidden="true" />
                    <span className="min-w-0 flex-1 truncate">{file.name}</span>
                    <span className="report-muted text-xs">{Math.max(1, Math.round(file.size / 1024))} KB</span>
                    <button type="button" aria-label={`Remove ${file.name}`} onClick={() => setFiles(current => current.filter((_, i) => i !== index))} className="report-remove rounded-md p-1"><X className="h-4 w-4" aria-hidden="true" /></button>
                  </li>
                ))}
              </ul>
            )}

            <label className="report-privacy flex items-start gap-3 rounded-xl border p-3 text-sm">
              <input type="checkbox" checked={privacyConfirmed} onChange={event => setPrivacyConfirmed(event.target.checked)} className="mt-0.5 h-4 w-4 accent-teal-700" />
              <span>Use de-identified research or sample files only. This local prototype is not configured for real patient-identifiable information.</span>
            </label>

            {error && <p className="report-error" role="alert"><AlertCircle className="h-4 w-4 shrink-0" aria-hidden="true" />{error}</p>}
            <button type="button" onClick={handleUpload} disabled={loading} className="report-primary-button flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold disabled:cursor-wait disabled:opacity-60">
              {loading ? 'Reading report…' : <>Read report <ArrowRight className="h-4 w-4" aria-hidden="true" /></>}
            </button>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-5">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <h3 className="text-lg font-semibold">Review extracted measurements</h3>
                <p className="report-muted mt-1 text-sm">Check each value against the original report. Correct extraction mistakes before saving.</p>
              </div>
              <span className="report-count-chip shrink-0 rounded-full px-3 py-1 text-xs font-medium">{markers.length} found</span>
            </div>

            {markers.length === 0 ? (
              <div className="report-empty-state rounded-xl border p-5 text-sm">
                <strong>No measurements found.</strong> Add them below if you can read them from the original report.
              </div>
            ) : (
              <div className="overflow-x-auto rounded-xl border">
                <table className="report-table w-full min-w-[680px] text-left text-sm">
                  <thead><tr><th>Measurement</th><th>Value</th><th>Unit</th><th>App range check</th><th><span className="sr-only">Remove</span></th></tr></thead>
                  <tbody>
                    {markers.map((marker, index) => (
                      <tr key={`${marker.key}-${index}`}>
                        <td><input aria-label={`Measurement ${index + 1} name`} value={marker.label} onChange={event => updateMarker(index, 'label', event.target.value)} className="report-input w-full rounded-lg border px-2.5 py-2" /></td>
                        <td><input aria-label={`${marker.label} value`} type="number" step="any" value={marker.value} onChange={event => updateMarker(index, 'value', event.target.value)} className="report-input w-full rounded-lg border px-2.5 py-2" /></td>
                        <td><input aria-label={`${marker.label} unit`} value={marker.unit} onChange={event => updateMarker(index, 'unit', event.target.value)} className="report-input w-24 rounded-lg border px-2.5 py-2" placeholder="Unit" /></td>
                        <td>{(() => {
                          const flag = reviewFlags.find(item => item.key === marker.key);
                          const generic = !flag || flag.category === 'General';
                          return <div className="space-y-1"><span className={`report-status-badge ${generic ? 'is-unavailable' : `is-${flag.status || 'normal'}`}`}>{generic ? 'Check report range' : (flag.badge || 'Reviewed')}</span><span className="report-muted block text-[11px]">{generic ? 'No mapped range' : `App guide: ${flag.reference_range}`}</span></div>;
                        })()}</td>
                        <td><button type="button" aria-label={`Remove ${marker.label}`} onClick={() => removeMarker(index)} className="report-remove rounded-md p-2"><X className="h-4 w-4" aria-hidden="true" /></button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div className="report-manual-add rounded-xl border p-4">
              <h4 className="text-sm font-semibold">Add a missing measurement</h4>
              <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-[1.5fr_1fr_1fr_auto]">
                <input aria-label="Measurement name" value={manualDraft.label} onChange={event => setManualDraft(current => ({ ...current, label: event.target.value }))} className="report-input rounded-lg border px-3 py-2.5 text-sm" placeholder="Name from report" />
                <input aria-label="Measurement value" type="number" step="any" value={manualDraft.value} onChange={event => setManualDraft(current => ({ ...current, value: event.target.value }))} className="report-input rounded-lg border px-3 py-2.5 text-sm" placeholder="Value" />
                <input aria-label="Measurement unit" value={manualDraft.unit} onChange={event => setManualDraft(current => ({ ...current, unit: event.target.value }))} className="report-input rounded-lg border px-3 py-2.5 text-sm" placeholder="Unit" />
                <button type="button" onClick={addManualMarker} className="report-secondary-button rounded-lg px-4 py-2.5 text-sm font-semibold">Add</button>
              </div>
            </div>

            <p className="report-caution rounded-xl border px-4 py-3 text-xs leading-5">This tool extracts values only; it does not diagnose disease. App range checks are general guides and may not match your laboratory. Prefer the interval printed on the original report.</p>
            {error && <p className="report-error" role="alert"><AlertCircle className="h-4 w-4 shrink-0" aria-hidden="true" />{error}</p>}
            <div className="flex flex-col-reverse gap-2 border-t pt-4 sm:flex-row sm:justify-between">
              <button type="button" onClick={() => { setStep(1); setError(''); }} className="report-secondary-button flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold"><ArrowLeft className="h-4 w-4" aria-hidden="true" /> Back</button>
              <button type="button" onClick={handleSaveReview} disabled={loading} className="report-primary-button flex items-center justify-center gap-2 rounded-xl px-5 py-3 text-sm font-semibold disabled:cursor-wait disabled:opacity-60">{loading ? 'Saving…' : <>Save reviewed values <CheckCircle2 className="h-4 w-4" aria-hidden="true" /></>}</button>
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="space-y-5">
            <div className="report-success flex items-start gap-3 rounded-xl border p-4">
              <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
              <div><h3 className="text-base font-semibold">Values saved for review</h3><p className="report-muted mt-1 text-sm">{markers.length} measurements were saved. No disease score was calculated.</p></div>
            </div>
            <div className="overflow-x-auto rounded-xl border">
              <table className="report-table w-full min-w-[560px] text-left text-sm">
                <thead><tr><th>Measurement</th><th>Value</th><th>Unit</th><th>App range check</th></tr></thead>
                <tbody>{markers.map((marker, index) => { const flag = reviewFlags.find(item => item.key === marker.key); const generic = !flag || flag.category === 'General'; return <tr key={`${marker.key}-${index}`}><td>{marker.label}</td><td className="font-mono">{marker.value}</td><td>{marker.unit || '—'}</td><td><span className={`report-status-badge ${generic ? 'is-unavailable' : `is-${flag.status || 'normal'}`}`}>{generic ? 'Check report range' : flag.badge}</span></td></tr>; })}</tbody>
              </table>
            </div>
            <div className="report-caution rounded-xl border px-4 py-3 text-sm leading-6"><strong>Disease prediction is unavailable.</strong> This workspace has no validated disease model enabled. Use the original report and qualified clinical judgment for care decisions.</div>
            <button type="button" onClick={startNewReview} className="report-secondary-button flex items-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold"><RotateCcw className="h-4 w-4" aria-hidden="true" /> Review another report</button>
          </div>
        )}
      </div>
    </section>
  );
}
