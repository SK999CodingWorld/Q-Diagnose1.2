import { BarChart2, CirclePlay, Eye, FileSpreadsheet, Globe, HeartPulse, Sliders, User, Zap } from 'lucide-react';

export const APP_TABS = [
  { id: 'overview', label: 'Home', fullLabel: 'Workspace overview', icon: Globe },
  { id: 'lab_reports', label: 'Blood & lab report', fullLabel: 'Upload and review a patient laboratory report', icon: FileSpreadsheet },
  { id: 'patients', label: 'Patient history', fullLabel: 'Patient report history', icon: User },
  { id: 'guide', label: 'How to use', fullLabel: 'Patient report walkthrough and user guide', icon: CirclePlay },
  { id: 'benchmarks', label: 'Research comparisons', fullLabel: 'Compare research model performance', icon: BarChart2 },
  { id: 'predict', label: 'Analyze a sample', fullLabel: 'Review a sample with a research model', icon: HeartPulse },
  { id: 'explain', label: 'Model factors', fullLabel: 'See which data features influenced a model', icon: Eye },
  { id: 'robustness', label: 'Reliability checks', fullLabel: 'Review model robustness and stability', icon: Sliders },
  { id: 'run', label: 'Run research experiment', fullLabel: 'Set up a research cohort model experiment', icon: Zap },
];
