import React, { useState, useEffect } from 'react';
import { apiFetch } from './api/client';
import { API_BASE } from './api/config';
import {
  FileSpreadsheet,
  UploadCloud,
  FileText,
  AlertTriangle,
  CheckCircle2,
  Activity,
  HeartPulse,
  Sparkles,
  RefreshCw,
  Info,
  ChevronRight,
  TrendingUp,
  Cpu,
  ShieldAlert,
  Sliders,
  PlusCircle,
  Stethoscope,
  Atom,
  Printer,
  User,
  Download
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

export default function LabReportAnalyzer() {
  const [reportText, setReportText] = useState('');
  const [reportTitle, setReportTitle] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [biomarkers, setBiomarkers] = useState({});
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [customKey, setCustomKey] = useState('');
  const [customVal, setCustomVal] = useState('');
  const [showTextEditor, setShowTextEditor] = useState(false);

  // Patient link state
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState('');
  const [saveToPatientMsg, setSaveToPatientMsg] = useState('');
  const [isSavingPatient, setIsSavingPatient] = useState(false);

  // Patient records are fetched only for optional local history linking.
  useEffect(() => {
    fetchPatients();
  }, []);

  const fetchPatients = async () => {
    try {
      const res = await apiFetch(`${API_BASE}/patients`);
      if (res.ok) {
        const data = await res.json();
        setPatients(data.patients || []);
        if (data.patients && data.patients.length > 0) {
          setSelectedPatientId(data.patients[0].patient_id);
        }
      }
    } catch (e) {
      console.error("Failed to fetch patients", e);
    }
  };

  const handleSaveToPatient = async () => {
    if (!selectedPatientId || !analysisResult) return;
    setIsSavingPatient(true);
    setSaveToPatientMsg('');
    try {
      const res = await apiFetch(`${API_BASE}/api/lab-reports/save-to-patient`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: selectedPatientId,
          report_title: reportTitle,
          biomarkers: biomarkers,
          primary_disease: analysisResult.diseases?.find(disease => disease.status === 'evaluated')?.name ?? null,
          chance_percentage: analysisResult.diseases?.find(disease => disease.status === 'evaluated')?.chances_percentage ?? null,
          risk_tier: analysisResult.diseases?.find(disease => disease.status === 'evaluated')?.risk_tier ?? null
        })
      });
      if (res.ok) {
        const d = await res.json();
        setSaveToPatientMsg(`Report attached to Patient ${selectedPatientId}! Visit the Patient Trajectory tab to view.`);
      } else {
        setSaveToPatientMsg("Failed to attach report to patient record.");
      }
    } catch (e) {
      setSaveToPatientMsg("Network error saving report to patient record.");
    } finally {
      setIsSavingPatient(false);
    }
  };

  const runAnalysis = async (currentMarkers, title, text) => {
    setLoading(true);
    setErrorMsg('');
    try {
      const res = await apiFetch(`${API_BASE}/api/lab-reports/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          report_title: title || reportTitle,
          report_text: text !== undefined ? text : reportText,
          biomarkers: currentMarkers || biomarkers
        })
      });
      if (res.ok) {
        const data = await res.json();
        setAnalysisResult(data);
        if (data.parsed_biomarkers) {
          setBiomarkers(data.parsed_biomarkers);
        }
      } else {
        const err = await res.json();
        setErrorMsg(err.detail || "Analysis failed.");
      }
    } catch (e) {
      setErrorMsg("Network error connecting to analysis engine.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setReportTitle(file.name);
    setLoading(true);
    setErrorMsg('');

    const formData = new FormData();
    formData.append("file", file);
    formData.append("report_title", file.name);

    try {
      const res = await apiFetch(`${API_BASE}/api/lab-reports/upload`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        setAnalysisResult(data);
        if (data.parsed_biomarkers) {
          setBiomarkers(data.parsed_biomarkers);
        }
        if (data.raw_text_snippet) {
          setReportText(data.raw_text_snippet);
        }
      } else {
        const err = await res.json();
        setErrorMsg(err.detail || "File upload failed.");
      }
    } catch (err) {
      setErrorMsg("Error uploading lab report file.");
    } finally {
      setLoading(false);
    }
  };

  const handleBiomarkerChange = (key, val) => {
    const num = parseFloat(val);
    const updated = { ...biomarkers, [key]: isNaN(num) ? 0 : num };
    setBiomarkers(updated);
    runAnalysis(updated, reportTitle, reportText);
  };

  const handleAddCustomBiomarker = (e) => {
    e.preventDefault();
    if (!customKey.trim()) return;
    const num = parseFloat(customVal);
    const updated = { ...biomarkers, [customKey.trim().toLowerCase()]: isNaN(num) ? 0 : num };
    setBiomarkers(updated);
    setCustomKey('');
    setCustomVal('');
    runAnalysis(updated, reportTitle, reportText);
  };

  const handlePrintSummary = () => {
    window.print();
  };

  // Prepare chart data for Recharts
  const chartData = analysisResult?.diseases?.map(d => ({
    name: d.name.length > 22 ? d.name.substring(0, 20) + "..." : d.name,
    fullName: d.name,
    chance: d.chances_percentage,
    classical: Math.round(d.classical_probability * 100),
    quantum: Math.round(d.quantum_probability * 100),
    tier: d.risk_tier
  })) || [];

  const getTierBadge = (tier) => {
    switch (tier) {
      case 'very_high':
        return <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-rose-950/80 text-rose-400 border border-rose-800/80">🔴 Very High Risk</span>;
      case 'high':
        return <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-orange-950/80 text-orange-400 border border-orange-800/80">🟠 High Risk</span>;
      case 'moderate':
        return <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-amber-950/80 text-amber-400 border border-amber-800/80">🟡 Moderate Risk</span>;
      default:
        return <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-800/80">🟢 Low Risk</span>;
    }
  };

  const getStatusBadge = (status, badge) => {
    switch (status) {
      case 'critical_high':
      case 'critical_low':
        return <span className="px-2 py-0.5 text-[10px] font-semibold rounded bg-rose-950/90 text-rose-400 border border-rose-800/70">{badge}</span>;
      case 'high':
      case 'low':
        return <span className="px-2 py-0.5 text-[10px] font-semibold rounded bg-amber-950/90 text-amber-400 border border-amber-800/70">{badge}</span>;
      default:
        return <span className="px-2 py-0.5 text-[10px] font-semibold rounded bg-emerald-950/90 text-emerald-400 border border-emerald-800/70">Normal</span>;
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-8 animate-fade-in text-slate-100">
      
      {/* 1. HERO HEADER */}
      <div className="relative rounded-2xl bg-gradient-to-br from-[#0E172A] via-[#0D1527] to-[#0A101D] border border-cyan-500/30 p-6 md:p-8 backdrop-blur-xl shadow-2xl overflow-hidden">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-72 h-72 rounded-full bg-gradient-to-br from-cyan-500/10 to-indigo-500/10 blur-3xl pointer-events-none" />
        
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-xs font-mono font-semibold uppercase tracking-wider text-cyan-400">
                PATIENT LAB REPORT REVIEW
              </span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white flex items-center gap-2.5">
              <FileSpreadsheet className="w-8 h-8 text-cyan-400" />
              Review a blood or laboratory report
            </h2>
            <p className="text-sm text-slate-300 max-w-3xl leading-relaxed">
              Upload a report to extract recognizable lab values and compare them with general reference intervals. Check every extracted value against the original report. Disease-specific ML and quantum predictions are not currently connected to this report workflow, so this tool cannot tell which disease a patient has or predict future illness.
            </p>
          </div>

          <div className="flex items-center gap-3 self-start md:self-auto">
            <button
              onClick={handlePrintSummary}
              className="flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/80 transition-all shadow-sm"
              title="Print Clinical Summary"
            >
              <Printer className="w-4 h-4 text-slate-400" />
              Print Report
            </button>
            <span className="px-3.5 py-2 rounded-xl bg-amber-950/50 border border-amber-800/50 text-amber-200 text-xs">Research use only</span>
          </div>
        </div>
      </div>

      {/* Primary patient workflow: upload an actual report. */}
      <div className="rounded-2xl bg-[#0D1424]/90 border border-slate-800 p-5 sm:p-7 space-y-5">
          <div>
            <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              <UploadCloud className="w-5 h-5 text-teal-400" />
              Start with a patient blood or lab report
            </h3>
            <p className="text-sm text-slate-400 mt-1 max-w-3xl">
              Choose a PDF, DOCX, CSV, TXT, or JSON report. Q-Diagnose extracts recognizable values for you to verify; it does not currently run disease-specific ML or quantum predictions from these reports.
            </p>
          </div>

          <label className="border-2 border-dashed border-slate-700/80 hover:border-cyan-500/60 rounded-2xl p-7 sm:p-10 flex flex-col items-center justify-center cursor-pointer bg-[#0A101D]/70 transition-all text-center">
            <UploadCloud className="w-9 h-9 text-cyan-400 mb-3" />
            <span className="text-sm font-semibold text-slate-100">
              {selectedFile ? selectedFile.name : "Choose a lab report file"}
            </span>
            <span className="text-xs text-slate-500 mt-1">
              {selectedFile ? `${Math.round(selectedFile.size / 1024)} KB` : "PDF, DOCX, CSV, TXT, JSON"}
            </span>
            <input
              type="file"
              accept=".pdf,.docx,.txt,.csv,.json,.rtf,.log"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>

          <button
            onClick={() => setShowTextEditor(!showTextEditor)}
            className="text-[11px] text-slate-400 hover:text-cyan-300 flex items-center justify-center gap-1 transition-colors"
          >
            <FileText className="w-3.5 h-3.5" />
            {showTextEditor ? "Hide report text" : "Paste or review report text instead"}
          </button>
      </div>

      {/* Raw Text Inspector (Collapsible) */}
      {showTextEditor && (
        <div className="rounded-xl bg-[#0B1220] border border-slate-800 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-slate-300 flex items-center gap-2">
              <FileText className="w-4 h-4 text-indigo-400" />
              Raw Laboratory Text Parser
            </h4>
            <button
              onClick={() => runAnalysis(biomarkers, reportTitle, reportText)}
              disabled={loading}
              className="px-3 py-1 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 text-xs font-medium flex items-center gap-1.5 transition-all"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              Re-read report values
            </button>
          </div>
          <textarea
            value={reportText}
            onChange={(e) => setReportText(e.target.value)}
            rows={6}
            placeholder="Paste clinical lab test values, reference intervals, or pathology observations..."
            className="w-full bg-[#080E1A] border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500/60 resize-y"
          />
        </div>
      )}

      {/* ERROR MESSAGE */}
      {errorMsg && (
        <div className="rounded-xl bg-rose-950/50 border border-rose-800 p-4 text-xs text-rose-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* 3. PRIMARY CLINICAL ALERT & SUMMARY HERO */}
      {analysisResult && (
        <div className="rounded-2xl bg-[#0D1528] border border-slate-800 p-6 md:p-8 space-y-6 shadow-xl">
          
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-6 border-b border-slate-800/80">
            <div className="space-y-1.5">
              <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
                Report review summary
              </span>
              <div className="flex flex-wrap items-center gap-3">
                <h3 className="text-xl sm:text-2xl font-extrabold text-white">
                  {analysisResult.biomarkers_extracted_count > 0 ? 'Lab values extracted for review' : 'No lab values were recognized'}
                </h3>
              </div>
              <p className="text-xs text-slate-400">
                {analysisResult.biomarkers_extracted_count} values recognized · {analysisResult.abnormal_biomarkers_count} values flagged against general reference intervals. Check values, units, and ranges against the original report.
              </p>
            </div>

            {/* Big Stat Hero Pill */}
            <div className="bg-[#121E36] border border-cyan-500/40 rounded-2xl p-4 sm:p-5 flex items-center gap-5 shadow-lg shadow-cyan-500/5">
              <div className="text-center">
                <div className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider">Disease prediction</div>
                <div className="text-base sm:text-lg font-bold text-slate-300">Not available</div>
                <div className="text-[10px] text-slate-400 font-medium">No disease-specific ML or quantum model is connected here</div>
              </div>
              <div className="w-[1px] h-12 bg-slate-700" />
              <div className="text-xs space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-slate-400">Patient disease predictions:</span>
                  <span className="font-bold text-slate-300">None</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-slate-400">Abnormal Markers:</span>
                  <span className="font-bold text-amber-400">{analysisResult.abnormal_biomarkers_count}</span>
                </div>
              </div>
            </div>
          </div>

          {analysisResult.diseases?.some(disease => disease.status === 'evaluated') ? <>
          {/* Model output is shown only when the API returns an evaluated disease-specific model. */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-cyan-400" />
                Multi-Disease Risk Differential (% Chance Comparison)
              </h4>
              <div className="flex items-center gap-4 text-[11px]">
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded bg-cyan-400" />
                  <span className="text-slate-300">Classical ML</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded bg-indigo-400" />
                  <span className="text-slate-300">Quantum Hilbert Space</span>
                </div>
              </div>
            </div>

            <div className="h-64 w-full bg-[#0A101E] rounded-xl p-4 border border-slate-800/80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#94A3B8', fontSize: 10 }}
                    interval={0}
                    angle={-15}
                    textAnchor="end"
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fill: '#94A3B8', fontSize: 11 }}
                    unit="%"
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                    formatter={(val, name) => [`${val}%`, name === 'chance' ? 'Classical Probability' : 'Quantum-Modulated Probability']}
                    labelFormatter={(label) => `Condition: ${label}`}
                  />
                  <Bar dataKey="classical" name="chance" fill="#34D399" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="quantum" name="quantum" fill="#EAB308" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 5. INDIVIDUAL DISEASE RISK CARDS (WHICH DISEASE & WHAT CHANCES) */}
          <div className="space-y-4 pt-2">
            <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              Detailed Disease Risk Analysis & Recommendations
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {analysisResult.diseases.map((disease) => {
                const isVeryHigh = disease.risk_tier === 'very_high';
                const isHigh = disease.risk_tier === 'high';
                const isMod = disease.risk_tier === 'moderate';

                return (
                  <div
                    key={disease.id}
                    className={`rounded-xl p-5 border transition-all flex flex-col justify-between space-y-4 ${
                      isVeryHigh
                        ? 'bg-[#15101E]/90 border-rose-500/50 shadow-lg shadow-rose-950/20'
                        : isHigh
                        ? 'bg-[#16131C]/90 border-orange-500/40 shadow-lg shadow-orange-950/20'
                        : isMod
                        ? 'bg-[#141824]/90 border-amber-500/30'
                        : 'bg-[#0E1626]/80 border-slate-800'
                    }`}
                  >
                    <div>
                      {/* Top Header of Card */}
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">
                            {disease.organ_system}
                          </span>
                          <h5 className="text-sm font-bold text-white mt-1.5 leading-snug">
                            {disease.name}
                          </h5>
                        </div>
                        {getTierBadge(disease.risk_tier)}
                      </div>

                      {/* Percentage Chance & Progress Bar */}
                      <div className="mt-4 space-y-2">
                        <div className="flex items-baseline justify-between">
                          <span className="text-xs text-slate-400">Disease Probability:</span>
                          <span className="text-lg font-black text-white">
                            {disease.chances_percentage}% Chance
                          </span>
                        </div>

                        {/* Progress Bar */}
                        <div className="w-full h-2.5 rounded-full bg-slate-800 overflow-hidden relative">
                          <div
                            className={`h-full rounded-full transition-all duration-700 ${
                              isVeryHigh
                                ? 'bg-gradient-to-r from-orange-500 to-rose-500'
                                : isHigh
                                ? 'bg-gradient-to-r from-amber-500 to-orange-500'
                                : isMod
                                ? 'bg-gradient-to-r from-yellow-500 to-amber-500'
                                : 'bg-gradient-to-r from-teal-500 to-emerald-500'
                            }`}
                            style={{ width: `${Math.max(disease.chances_percentage, 5)}%` }}
                          />
                        </div>

                        {/* Dual-Track Model Breakdown */}
                        <div className="grid grid-cols-2 gap-2 pt-1 text-[11px] font-mono">
                          <div className="p-1.5 rounded bg-slate-900/60 border border-slate-800/80">
                            <span className="text-slate-400 block text-[9px] uppercase">Classical ML</span>
                            <span className="text-cyan-300 font-bold">{Math.round(disease.classical_probability * 100)}%</span>
                          </div>
                          <div className="p-1.5 rounded bg-slate-900/60 border border-slate-800/80">
                            <span className="text-slate-400 block text-[9px] uppercase">Quantum Model</span>
                            <span className="text-indigo-300 font-bold">{Math.round(disease.quantum_probability * 100)}%</span>
                          </div>
                        </div>
                      </div>

                      {/* Contributing Biomarkers / Diagnostic Culprits */}
                      {disease.contributing_markers && disease.contributing_markers.length > 0 && (
                        <div className="mt-3.5 space-y-1.5">
                          <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold">
                            Diagnostic Culprit Markers:
                          </span>
                          <div className="flex flex-wrap gap-1.5">
                            {disease.contributing_markers.map((m) => (
                              <span
                                key={m.key}
                                className={`text-[10px] px-2 py-0.5 rounded border flex items-center gap-1 ${
                                  m.status.includes('critical')
                                    ? 'bg-rose-950/70 text-rose-300 border-rose-800/80 font-semibold'
                                    : m.status.includes('high') || m.status.includes('low')
                                    ? 'bg-amber-950/70 text-amber-300 border-amber-800/80'
                                    : 'bg-emerald-950/50 text-emerald-300 border-emerald-800/50'
                                }`}
                              >
                                <span>{m.label}:</span>
                                <strong>{m.value} {m.unit}</strong>
                                <span>({m.badge})</span>
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Clinical Findings */}
                      <p className="text-xs text-slate-300 mt-3 bg-slate-900/50 p-2.5 rounded-lg border border-slate-800/60 leading-relaxed">
                        <strong className="text-slate-200">Finding: </strong>
                        {disease.clinical_findings}
                      </p>
                    </div>

                    {/* Action Recommendation */}
                    <div className="pt-2 border-t border-slate-800/70 text-[11px] text-cyan-300 flex items-start gap-1.5">
                      <ChevronRight className="w-3.5 h-3.5 mt-0.5 text-cyan-400 shrink-0" />
                      <span>
                        <strong className="text-slate-200">Next Action: </strong>
                        {disease.action_recommendation}
                      </span>
                    </div>

                  </div>
                );
              })}
            </div>
          </div>
          </> : (
            <div className="rounded-xl bg-amber-950/35 border border-amber-800/60 p-4 text-xs leading-relaxed text-amber-100/90" role="status">
              <strong className="block text-sm text-amber-200">No disease prediction was run</strong>
              <span className="mt-1 block">This version extracts lab values and flags possible out-of-range results. A validated disease-specific machine-learning or quantum model is not connected to patient reports, so this result cannot identify a disease or predict future illness.</span>
            </div>
          )}

          {/* ATTACH REPORT TO PATIENT RECORD SECTION */}
          <div className="rounded-xl bg-[#090F1E] border border-cyan-500/30 p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="space-y-1">
              <h4 className="text-xs font-bold text-white flex items-center gap-2">
                <User className="w-4 h-4 text-cyan-400" />
                Save extracted report values to patient history
              </h4>
              <p className="text-[11px] text-slate-400">
                Only save reports to a local record when you are authorized to manage that information. No disease prediction is stored.
              </p>
              {saveToPatientMsg && (
                <div className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5 pt-1">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>{saveToPatientMsg}</span>
                </div>
              )}
            </div>

            <div className="flex items-center gap-2 self-start sm:self-auto">
              <select
                value={selectedPatientId}
                onChange={(e) => setSelectedPatientId(e.target.value)}
                className="bg-[#10192E] border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-400"
              >
                {patients.map((p) => (
                  <option key={p.patient_id} value={p.patient_id}>
                    {p.patient_id} — {p.name} ({p.sex}, {p.age || 'N/A'}y)
                  </option>
                ))}
              </select>

              <button
                type="button"
                onClick={handleSaveToPatient}
                disabled={isSavingPatient || !selectedPatientId}
                className="px-4 py-1.5 rounded-lg bg-gradient-to-r from-cyan-500 to-indigo-500 hover:from-cyan-400 hover:to-indigo-400 text-slate-950 font-bold text-xs disabled:opacity-50 transition-all flex items-center gap-1.5 shadow-md shadow-cyan-500/10 cursor-pointer"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                {isSavingPatient ? "Saving..." : "Save to Patient Record"}
              </button>
            </div>
          </div>

        </div>
      )}

      {/* Extracted report values for review */}
      <div className="rounded-xl bg-[#0D1424]/90 border border-slate-800 p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              Extracted lab values and reference intervals
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Check each extracted value and unit against the uploaded report. Reference intervals vary by lab and patient context; editing a value only refreshes the marker flags.
            </p>
          </div>

          {/* Quick Add Custom Marker */}
          <form onSubmit={handleAddCustomBiomarker} className="flex items-center gap-2">
            <input
              type="text"
              placeholder="e.g. hba1c, ldl"
              value={customKey}
              onChange={(e) => setCustomKey(e.target.value)}
              className="bg-[#0A101D] border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-white focus:outline-none focus:border-cyan-500 w-28"
            />
            <input
              type="number"
              step="any"
              placeholder="Value"
              value={customVal}
              onChange={(e) => setCustomVal(e.target.value)}
              className="bg-[#0A101D] border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-white focus:outline-none focus:border-cyan-500 w-20"
            />
            <button
              type="submit"
              className="px-2.5 py-1 rounded-lg bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 text-xs font-medium hover:bg-cyan-500/30 transition-all flex items-center gap-1"
            >
              <PlusCircle className="w-3.5 h-3.5" />
              Add
            </button>
          </form>
        </div>

        {/* Biomarkers Grid Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-[#090E1A] text-slate-400 font-mono text-[11px]">
                <th className="py-2.5 px-3">Biomarker</th>
                <th className="py-2.5 px-3">Category</th>
                <th className="py-2.5 px-3">Measured Value</th>
                <th className="py-2.5 px-3">Standard Reference Interval</th>
                <th className="py-2.5 px-3">Status Flag</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-sans">
              {analysisResult?.flagged_biomarkers?.map((m) => (
                <tr key={m.key} className="hover:bg-[#121B30]/50 transition-colors">
                  <td className="py-2.5 px-3 font-medium text-slate-200">
                    {m.label}
                  </td>
                  <td className="py-2.5 px-3 text-slate-400 font-mono text-[10px]">
                    {m.category}
                  </td>
                  <td className="py-2.5 px-3">
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        step="any"
                        value={biomarkers[m.key] !== undefined ? biomarkers[m.key] : m.value}
                        onChange={(e) => handleBiomarkerChange(m.key, e.target.value)}
                        className="w-24 bg-[#0A101D] border border-slate-700 rounded px-2 py-1 text-xs text-cyan-300 font-mono focus:outline-none focus:border-cyan-400"
                      />
                      <span className="text-[11px] text-slate-400">{m.unit}</span>
                    </div>
                  </td>
                  <td className="py-2.5 px-3 text-slate-400 font-mono text-[11px]">
                    {m.reference_range}
                  </td>
                  <td className="py-2.5 px-3">
                    {getStatusBadge(m.status, m.badge)}
                  </td>
                </tr>
              ))}
              {(!analysisResult?.flagged_biomarkers || analysisResult.flagged_biomarkers.length === 0) && (
                <tr>
                  <td colSpan={5} className="py-6 text-center text-slate-500">
                    No biomarkers were recognized yet. Upload a lab report or paste report text above.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 7. REGULATORY & CLINICAL SAFETY DISCLAIMER */}
      <div className="p-4 rounded-xl bg-[#0B1222]/80 border border-slate-800 text-[11px] text-slate-400 leading-relaxed flex items-start gap-2.5">
        <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
        <span>
          <strong className="text-slate-300">Important: </strong>
          This prototype extracts selected laboratory values and flags them against general reference intervals. It does not currently connect a disease-specific machine-learning or quantum model to patient reports, diagnose disease, or predict future illness. Reference ranges vary between laboratories and patients; discuss results with a qualified clinician.
        </span>
      </div>

    </div>
  );
}

