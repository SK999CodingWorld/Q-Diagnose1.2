import React from 'react';
import {
  Activity,
  ArrowRight,
  ChevronRight,
  CirclePlay,
  FileUp,
  FileSpreadsheet,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { useTheme } from './state/ThemeContext';

const workflows = [
  {
    id: 'lab_reports',
    icon: FileUp,
    eyebrow: 'Patient report',
    title: 'Review a blood or lab report',
    description: 'Upload a report, check extracted values, and review what the available analysis can and cannot tell you.',
    action: 'Upload a report',
    accent: 'teal',
  },
  {
    id: 'guide',
    icon: CirclePlay,
    eyebrow: 'Step-by-step',
    title: 'How to review a report',
    description: 'Follow the upload, value-checking, and results workflow in a short guided walkthrough.',
    action: 'Watch walkthrough',
    accent: 'sky',
  },
  {
    id: 'patients',
    icon: Activity,
    eyebrow: 'History',
    title: 'Patient report history',
    description: 'Review reports saved to a local patient record, when you are authorized to use that information.',
    action: 'Open history',
    accent: 'violet',
  },
];

export default function LandingPage({ onExplorePlatform }) {
  const { theme } = useTheme();
  const dark = theme === 'dark';

  const palette = {
    page: dark ? 'bg-[#0d1e18] text-slate-100' : 'bg-[#f7faf7] text-slate-900',
    card: dark ? 'border-emerald-900/60 bg-[#132820]/90' : 'border-[#dce8df] bg-white',
    subtle: dark ? 'text-slate-400' : 'text-[#5b7467]',
    heading: dark ? 'text-white' : 'text-[#18372e]',
    line: dark ? 'border-emerald-900/60' : 'border-[#e3ebe5]',
  };

  return (
    <div className={`relative isolate overflow-hidden ${palette.page}`}>
      <div aria-hidden="true" className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[560px] overflow-hidden">
        <div className={`absolute -top-56 left-1/2 h-[520px] w-[900px] -translate-x-1/2 rounded-full blur-3xl ${dark ? 'bg-emerald-500/10' : 'bg-emerald-200/45'}`} />
        <div className={`absolute right-[6%] top-24 h-64 w-64 rounded-full blur-3xl ${dark ? 'bg-lime-500/10' : 'bg-lime-200/35'}`} />
      </div>

      <main className="mx-auto w-full max-w-[1440px] px-4 pb-16 pt-7 sm:px-7 sm:pt-10 lg:px-10">
        <section className="grid items-center gap-8 lg:grid-cols-[1.05fr_.95fr] lg:gap-12">
          <div className="space-y-6 sm:space-y-7">
            <div className={`inline-flex max-w-full items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold ${dark ? 'border-emerald-400/20 bg-emerald-400/10 text-emerald-200' : 'border-emerald-200 bg-white/85 text-emerald-800 shadow-sm'}`}>
              <Sparkles className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
              <span>Patient laboratory report</span>
              <span className={`hidden h-1 w-1 rounded-full sm:block ${dark ? 'bg-teal-300' : 'bg-teal-500'}`} />
              <span className={`hidden sm:inline ${dark ? 'text-slate-300' : 'text-slate-500'}`}>Research prototype</span>
            </div>

            <div className="max-w-3xl space-y-4">
              <h1 className={`text-[2.65rem] font-bold leading-[1.05] tracking-[-0.045em] sm:text-5xl lg:text-[3.8rem] ${palette.heading}`}>
                Start with the report.<br /><span className="text-gradient-cyan">Understand the results.</span>
              </h1>
              <p className={`max-w-2xl text-base leading-7 sm:text-lg ${palette.subtle}`}>
                Upload a blood or laboratory report, review the values we can extract, and see which analysis is actually available. This research prototype does not currently provide validated disease predictions from patient reports or forecast future illness.
              </p>
            </div>

            <div className="flex flex-col gap-3 min-[440px]:flex-row min-[440px]:flex-wrap">
              <button
                type="button"
                onClick={() => onExplorePlatform('lab_reports')}
                className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-emerald-700 px-5 text-sm font-semibold text-white shadow-lg shadow-emerald-900/10 transition hover:-translate-y-0.5 hover:bg-emerald-800 focus-visible:outline-emerald-600"
              >
                <FileUp className="h-4 w-4" aria-hidden="true" />
                Review a lab report
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </button>
              <button
                type="button"
                onClick={() => onExplorePlatform('guide')}
                className={`inline-flex min-h-12 items-center justify-center gap-2 rounded-xl border px-5 text-sm font-semibold transition hover:-translate-y-0.5 focus-visible:outline-sky-600 ${dark ? 'border-slate-600 bg-slate-900/70 text-slate-100 hover:bg-slate-800' : 'border-slate-300 bg-white text-slate-700 shadow-sm hover:border-slate-400 hover:bg-slate-50'}`}
              >
                <CirclePlay className="h-4 w-4" aria-hidden="true" />
                How it works
              </button>
            </div>

            <div className={`inline-flex max-w-2xl items-start gap-2 rounded-xl border px-3 py-3 text-xs leading-5 ${dark ? 'border-amber-300/20 bg-amber-300/[.06] text-amber-100/80' : 'border-amber-200 bg-amber-50/80 text-amber-900'}`}>
              <ShieldCheck className={`mt-0.5 h-4 w-4 shrink-0 ${dark ? 'text-amber-200' : 'text-amber-700'}`} aria-hidden="true" />
              <span>Reference ranges can vary by laboratory and patient context. Always compare extracted values with the original report and discuss concerns with a healthcare professional.</span>
            </div>
          </div>

          <section aria-label="Patient report workflow" className={`relative rounded-[1.6rem] border p-4 shadow-xl sm:p-6 ${palette.card} ${dark ? 'shadow-black/20' : 'shadow-slate-300/30'}`}>
            <div className={`absolute inset-x-8 top-0 h-px ${dark ? 'bg-gradient-to-r from-transparent via-emerald-400/60 to-transparent' : 'bg-gradient-to-r from-transparent via-emerald-300 to-transparent'}`} />
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className={`text-xs font-semibold uppercase tracking-[.14em] ${dark ? 'text-emerald-300' : 'text-emerald-700'}`}>Your report, step by step</p>
                <h2 className={`mt-2 line-clamp-2 text-xl font-bold leading-tight sm:text-2xl ${palette.heading}`}>
                  Upload. Check. Understand.
                </h2>
                <p className={`mt-2 text-sm ${palette.subtle}`}>
                  Start from a blood test or other laboratory report.
                </p>
              </div>
              <span className={`inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl ${dark ? 'bg-emerald-400/10 text-emerald-200' : 'bg-emerald-50 text-emerald-700'}`}>
                <Activity className="h-5 w-5" aria-hidden="true" />
              </span>
            </div>

            <div className={`my-5 border-t ${palette.line}`} />

            <div className="space-y-3">
              {[
                { step: '01', title: 'Upload a lab report', detail: 'Choose a PDF, DOCX, CSV, TXT, or JSON report.', icon: FileUp },
                { step: '02', title: 'Check extracted values', detail: 'Verify units and flags against the original report.', icon: FileSpreadsheet },
                { step: '03', title: 'Review what is available', detail: 'This build does not return patient-level disease predictions.', icon: ShieldCheck },
              ].map(item => {
                const Icon = item.icon;
                return (
                  <div key={item.step} className={`flex items-center gap-3 rounded-xl px-3 py-3 ${dark ? 'bg-slate-800/65' : 'bg-slate-50'}`}>
                    <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${dark ? 'bg-slate-700 text-emerald-200' : 'bg-white text-emerald-700 shadow-sm'}`}>
                      <Icon className="h-4 w-4" aria-hidden="true" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className={`text-sm font-semibold ${palette.heading}`}>{item.title}</div>
                      <p className={`mt-0.5 text-xs leading-5 ${palette.subtle}`}>{item.detail}</p>
                    </div>
                    <span className={`hidden font-mono text-[10px] sm:block ${dark ? 'text-slate-500' : 'text-slate-400'}`}>{item.step}</span>
                  </div>
                );
              })}
            </div>

            <div className={`mt-4 flex items-start gap-2.5 rounded-xl border px-3 py-3 ${dark ? 'border-slate-700 bg-slate-800/50' : 'border-slate-200 bg-slate-50'}`}>
              <ShieldCheck className={`mt-0.5 h-4 w-4 shrink-0 ${dark ? 'text-amber-200' : 'text-amber-700'}`} aria-hidden="true" />
              <p className={`text-xs leading-5 ${palette.subtle}`}>
                A reference-range flag is not a diagnosis. Review the original lab report with a qualified healthcare professional.
              </p>
            </div>
          </section>
        </section>

        <section aria-labelledby="workflow-heading" className="mt-12 sm:mt-16">
          <div className="mb-5 flex items-end justify-between gap-4">
            <div>
              <p className={`text-xs font-semibold uppercase tracking-[.14em] ${dark ? 'text-teal-300' : 'text-teal-700'}`}>Patient report workflow</p>
              <h2 id="workflow-heading" className={`mt-1 text-xl font-bold tracking-tight sm:text-2xl ${palette.heading}`}>What would you like to do?</h2>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3 sm:gap-4">
            {workflows.map(item => {
              const Icon = item.icon;
              const iconColor = dark ? 'bg-emerald-400/10 text-emerald-200' : 'bg-emerald-50 text-emerald-700';
              return (
                <button
                  type="button"
                  key={item.id}
                  onClick={() => onExplorePlatform(item.id)}
                  className={`group flex min-h-[172px] flex-col items-start rounded-2xl border p-4 text-left transition hover:-translate-y-0.5 focus-visible:outline-emerald-600 sm:p-5 ${palette.card} ${dark ? 'hover:border-emerald-700 hover:bg-[#183629]' : 'shadow-sm hover:border-emerald-300 hover:shadow-md'}`}
                >
                  <div className="flex w-full items-start justify-between">
                    <span className={`inline-flex h-10 w-10 items-center justify-center rounded-xl ${iconColor}`}>
                      <Icon className="h-5 w-5" aria-hidden="true" />
                    </span>
                    <ArrowRight className={`h-4 w-4 transition-transform group-hover:translate-x-1 ${dark ? 'text-slate-500 group-hover:text-emerald-200' : 'text-slate-400 group-hover:text-emerald-700'}`} aria-hidden="true" />
                  </div>
                  <span className={`mt-4 text-[10px] font-bold uppercase tracking-[.14em] ${dark ? 'text-slate-500' : 'text-slate-500'}`}>{item.eyebrow}</span>
                  <span className={`mt-1 text-base font-bold ${palette.heading}`}>{item.title}</span>
                  <span className={`mt-1 text-xs leading-5 ${palette.subtle}`}>{item.description}</span>
                </button>
              );
            })}
          </div>
        </section>

        <section className={`mt-8 flex flex-col gap-3 rounded-2xl border p-4 sm:flex-row sm:items-center sm:justify-between sm:px-5 ${dark ? 'border-slate-700/80 bg-slate-900/50' : 'border-slate-200 bg-white/75'}`}>
          <div className="flex items-start gap-3">
            <span className={`mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${dark ? 'bg-slate-800 text-teal-200' : 'bg-teal-50 text-teal-700'}`}>
              <ShieldCheck className="h-4 w-4" aria-hidden="true" />
            </span>
            <div>
              <h3 className={`text-sm font-semibold ${palette.heading}`}>Looking for model research tools?</h3>
              <p className={`mt-1 text-xs leading-5 ${palette.subtle}`}>Model comparisons are a separate research workspace. Their cohort scores are not patient-specific predictions from an uploaded report.</p>
            </div>
          </div>
          <button type="button" onClick={() => onExplorePlatform('benchmarks')} className={`inline-flex shrink-0 items-center gap-1 self-start text-sm font-semibold sm:self-center ${dark ? 'text-emerald-300 hover:text-emerald-200' : 'text-emerald-800 hover:text-emerald-950'}`}>
            Open model comparison <ChevronRight className="h-4 w-4" aria-hidden="true" />
          </button>
        </section>
      </main>
    </div>
  );
}
