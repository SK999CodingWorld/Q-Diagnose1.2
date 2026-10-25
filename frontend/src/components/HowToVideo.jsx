import React, { useEffect, useState } from 'react';
import { Activity, ArrowRight, BarChart3, Check, ChevronLeft, ChevronRight, CirclePlay, Database, FileSpreadsheet, Microscope, Pause, Play, RotateCcw, ShieldAlert, ShieldCheck, UploadCloud, Volume2, VolumeX } from 'lucide-react';

const steps = [
  {
    title: 'Upload the patient’s lab report',
    label: 'Start with the report',
    icon: UploadCloud,
    section: 'lab_reports',
    destination: 'Open report review',
    narration: 'Open Blood and lab report, then choose a report file. The current page accepts PDF, DOCX, CSV, TXT, or JSON. Use only information you are authorized to process. This local research prototype is not a patient-care system.',
    bullets: ['Open Blood & lab report from the navigation.', 'Choose a supported report file.', 'Use only authorized, de-identified information where possible.'],
  },
  {
    title: 'Check the extracted values',
    label: 'Verify the report reading',
    icon: FileSpreadsheet,
    section: 'lab_reports',
    destination: 'Review extracted values',
    narration: 'After upload, compare each extracted name, value, and unit with the original report. OCR and parsing can make mistakes. Correct any value that does not match before you use the reference-range review.',
    bullets: ['Compare values and units with the original report.', 'Correct parsing mistakes before reviewing flags.', 'Keep the original report available for context.'],
  },
  {
    title: 'Review reference-range flags',
    label: 'Understand the flags',
    icon: Activity,
    section: 'lab_reports',
    destination: 'Review report values',
    narration: 'The current report tool can flag extracted biomarkers against general reference intervals. These intervals can differ by laboratory and by patient context. A high or low flag is not a diagnosis and does not predict that a disease will develop.',
    bullets: ['Check the units and the lab’s own reference interval.', 'A flag describes a value, not a disease.', 'Do not treat an out-of-range flag as a future prediction.'],
  },
  {
    title: 'Check whether a disease model ran',
    label: 'See what is available',
    icon: BarChart3,
    section: 'lab_reports',
    destination: 'View report analysis',
    narration: 'This build currently extracts lab values and shows reference-range flags. It does not connect a validated disease-specific classical machine-learning or quantum model to the uploaded patient report, so no patient disease prediction is produced. A research-cohort benchmark score cannot be applied to an individual report.',
    bullets: ['No patient-level disease prediction is produced here today.', 'Research dataset metrics are not personal risk scores.', 'Future disease prediction requires validated disease-specific models.'],
  },
  {
    title: 'Decide what to do next',
    label: 'Keep clinical context',
    icon: Microscope,
    section: 'lab_reports',
    destination: 'Return to report review',
    narration: 'Use this screen to organize extracted values, not to decide that someone has or will get a disease. Consider symptoms, medications, history, and the laboratory’s own ranges with a qualified healthcare professional.',
    bullets: ['Consider symptoms, medication, and medical history.', 'Ask a clinician to interpret concerning results.', 'Seek urgent care for urgent symptoms, not a model score.'],
  },
  {
    title: 'Open model comparisons only for research',
    label: 'Separate research tools',
    icon: ShieldCheck,
    section: 'benchmarks',
    destination: 'Open research comparisons',
    narration: 'The separate research area compares models on labeled cohort datasets. Those experimental results are not predictions for the person whose report you uploaded and should not be used for diagnosis or treatment.',
    bullets: ['Cohort benchmarks are separate from report review.', 'They are not patient-specific predictions.', 'This prototype is not validated to guide care.'],
  },
];

function SceneIllustration({ step }) {
  const Icon = steps[step].icon;
  const scenes = [
    <div className="guide-scene-stack" key="report-upload">
      <div className="guide-ui-window"><div className="guide-ui-top"><span /><span /><span /><i>Patient report review</i></div><div className="guide-upload-box"><UploadCloud size={25} /><strong>Choose a blood or lab report</strong><small>PDF · DOCX · CSV · TXT · JSON</small><span className="guide-status-chip"><Check size={12} /> Upload your authorized report</span></div></div>
    </div>,
    <div className="guide-scene-stack" key="verify-values">
      <div className="guide-ui-window"><div className="guide-ui-top"><span /><span /><span /><i>Extracted report values</i></div><div className="guide-report-row"><div><small>TEST NAME</small><b>Check the label</b></div><div><small>VALUE</small><b>Match source report</b></div><div><small>UNIT</small><b>Confirm units</b></div></div><div className="guide-caveat"><FileSpreadsheet size={15} /> Check every extracted entry against the original document.</div></div>
    </div>,
    <div className="guide-scene-stack" key="range-flags">
      <div className="guide-ui-window"><div className="guide-ui-top"><span /><span /><span /><i>Reference range review</i></div><div className="guide-review-row"><span className="guide-review-icon"><Activity size={19} /></span><div><b>Read the flag in context</b><small>Compare with the lab’s own range and units</small></div></div><div className="guide-caveat"><ShieldAlert size={15} /> An out-of-range flag is not a disease diagnosis.</div></div>
    </div>,
    <div className="guide-scene-stack" key="model-status">
      <div className="guide-ui-window"><div className="guide-ui-top"><span /><span /><span /><i>Disease model availability</i></div><div className="guide-review-row"><span className="guide-review-icon"><Microscope size={19} /></span><div><b>No patient prediction returned</b><small>Disease-specific ML and quantum models are not connected</small></div></div><div className="guide-caveat"><ShieldAlert size={15} /> No future illness prediction is available in this build.</div></div>
    </div>,
    <div className="guide-scene-stack" key="next-step">
      <div className="guide-ui-window"><div className="guide-ui-top"><span /><span /><span /><i>Next steps</i></div><div className="guide-review-row"><span className="guide-review-icon"><ShieldCheck size={19} /></span><div><b>Keep clinical context</b><small>Bring the original report to a qualified clinician</small></div></div><div className="guide-ui-pills"><span>Symptoms</span><span>History</span><span>Medication</span></div></div>
    </div>,
    <div className="guide-scene-stack" key="research-only">
      <div className="guide-ui-window"><div className="guide-ui-top"><span /><span /><span /><i>Separate research workspace</i></div><div className="guide-result-row"><div><small>COHORT</small><b>Group study data</b></div><div><small>MODEL</small><b>Experimental scores</b></div><div><small>PERSONAL RESULT</small><b>Not provided</b></div></div><div className="guide-caveat"><Database size={15} /> Cohort comparisons are not patient risk estimates.</div></div>
    </div>,
  ];

  return <div className={`guide-player-stage guide-stage-${step}`} aria-label={`Illustration: ${steps[step].label}`}><div className="guide-stage-glow" /><div className="guide-stage-heading"><span className="guide-stage-icon"><Icon size={18} /></span><span>{steps[step].label}</span><span className="guide-scene-count">{String(step + 1).padStart(2, '0')} / {String(steps.length).padStart(2, '0')}</span></div>{scenes[step]}<div className="guide-stage-caption">Q-Diagnose workflow illustration · no patient data shown</div></div>;
}

export default function HowToVideo({ onNavigate }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [narrationEnabled, setNarrationEnabled] = useState(false);

  useEffect(() => {
    if (!playing) return undefined;
    if (currentStep >= steps.length - 1) {
      setPlaying(false);
      return undefined;
    }
    const timer = window.setTimeout(() => setCurrentStep(current => Math.min(current + 1, steps.length - 1)), 9000);
    return () => window.clearTimeout(timer);
  }, [currentStep, playing]);

  useEffect(() => {
    if (!playing || !narrationEnabled || !('speechSynthesis' in window)) return undefined;
    window.speechSynthesis.cancel();
    const speech = new SpeechSynthesisUtterance(steps[currentStep].narration);
    speech.rate = 0.96;
    speech.pitch = 1;
    window.speechSynthesis.speak(speech);
    return () => window.speechSynthesis.cancel();
  }, [currentStep, narrationEnabled, playing]);

  useEffect(() => () => {
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  }, []);

  const move = direction => setCurrentStep(index => (index + direction + steps.length) % steps.length);
  const active = steps[currentStep];

  return (
    <div className="howto-page">
      <header className="howto-heading">
        <div><p className="howto-eyebrow">PATIENT REPORT REVIEW · ABOUT 1 MINUTE</p><h2>How to review a blood report</h2><p>Upload a report, verify the extracted values, understand reference flags, and see what this build can currently report.</p></div>
        <span className="howto-research-chip"><ShieldCheck size={15} /> Research use only</span>
      </header>

      <div className="howto-layout">
        <section className="howto-player-card" aria-label="Interactive video walkthrough">
          <SceneIllustration step={currentStep} />
          <div className="howto-player-controls">
            <div className="howto-timeline" aria-label={`Step ${currentStep + 1} of ${steps.length}`}>
              {steps.map((item, index) => <button key={item.label} type="button" aria-label={`Go to step ${index + 1}: ${item.label}`} aria-current={currentStep === index ? 'step' : undefined} className={currentStep === index ? 'is-current' : index < currentStep ? 'is-done' : ''} onClick={() => { setCurrentStep(index); setPlaying(false); }} />)}
            </div>
            <div className="howto-controls-row">
              <div className="howto-transport">
                <button type="button" className="howto-play-button" onClick={() => { if (!playing && currentStep === steps.length - 1) setCurrentStep(0); setPlaying(value => !value); }} aria-label={playing ? 'Pause walkthrough' : 'Play walkthrough'}>{playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}</button>
                <button type="button" className="howto-icon-button" onClick={() => { setPlaying(false); setCurrentStep(0); }} aria-label="Restart walkthrough"><RotateCcw size={16} /></button>
                <span className="howto-step-time">{String(currentStep + 1).padStart(2, '0')} <i>/</i> {String(steps.length).padStart(2, '0')}</span>
              </div>
              <div className="howto-transport">
                <button type="button" className={`howto-icon-button ${narrationEnabled ? 'is-on' : ''}`} onClick={() => setNarrationEnabled(value => !value)} aria-pressed={narrationEnabled} aria-label={narrationEnabled ? 'Turn narration off' : 'Turn narration on'}>{narrationEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}</button>
                <button type="button" className="howto-icon-button" onClick={() => move(-1)} aria-label="Previous step"><ChevronLeft size={18} /></button>
                <button type="button" className="howto-icon-button" onClick={() => move(1)} aria-label="Next step"><ChevronRight size={18} /></button>
              </div>
            </div>
          </div>

          <div className="howto-caption" aria-live="polite">
            <div className="howto-caption-title"><span>STEP {String(currentStep + 1).padStart(2, '0')}</span><h3>{active.title}</h3></div>
            <p>{active.narration}</p>
          </div>
        </section>

        <aside className="howto-chapters" aria-label="Walkthrough chapters">
          <div className="howto-chapters-heading"><span>IN THIS WALKTHROUGH</span><strong>{steps.length} steps</strong></div>
          {steps.map((step, index) => {
            const Icon = step.icon;
            return <button className={`howto-chapter ${index === currentStep ? 'is-active' : ''}`} type="button" key={step.title} onClick={() => { setCurrentStep(index); setPlaying(false); }} aria-current={index === currentStep ? 'step' : undefined}><span className="howto-chapter-number">{index < currentStep ? <Check size={13} /> : String(index + 1).padStart(2, '0')}</span><span className="howto-chapter-copy"><strong>{step.title}</strong><small>{step.label}</small></span><Icon size={16} className="howto-chapter-icon" aria-hidden="true" /></button>;
          })}
          <div className="howto-chapter-footer"><span className="howto-footer-shield"><ShieldAlert size={16} /></span><p>Research support only. Model outputs are not validated to guide care.</p></div>
        </aside>
      </div>

      <section className="howto-step-detail" aria-labelledby="howto-detail-heading">
        <div className="howto-detail-heading"><div><p className="howto-eyebrow">STEP {String(currentStep + 1).padStart(2, '0')}</p><h3 id="howto-detail-heading">{active.title}</h3></div><button type="button" className="howto-open-section" onClick={() => onNavigate(active.section)}>{active.destination}<ArrowRight size={15} /></button></div>
        <ul>{active.bullets.map(item => <li key={item}><Check size={15} />{item}</li>)}</ul>
      </section>
      <p className="howto-video-note"><CirclePlay size={14} /> This guided walkthrough is interactive. Turn narration on for spoken guidance, or use the captions and chapter list.</p>
    </div>
  );
}
