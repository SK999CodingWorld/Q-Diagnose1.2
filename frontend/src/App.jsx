import React, { useState, useEffect } from 'react';
import { apiFetch } from './api/client';
import { API_BASE } from './api/config';
import { useDashboard } from './state/DashboardContext';
import { useTheme } from './state/ThemeContext';
import { APP_TABS } from './config/navigation';
import AppFooter from './components/AppFooter';
import AppHeader from './components/AppHeader';
import AppNavigation from './components/AppNavigation';
import ClinicalNotice from './components/ClinicalNotice';
import HowToVideo from './components/HowToVideo';
import {
  Activity,
  Cpu,
  BarChart2,
  Sliders,
  Layers,
  Sparkles,
  Info,
  RefreshCw,
  Zap,
  TrendingDown,
  Database,
  Eye,
  FileSpreadsheet,
  HeartPulse,
  User,
  UserPlus,
  Calendar,
  PlusCircle,
  Scale,
  ArrowUpDown
} from 'lucide-react';
import LandingPage from './LandingPage';
import PatientHealthReportAnalysis from './PatientHealthReportAnalysis';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

// Color palette for models
const MODEL_COLORS = {
  logistic_regression: "#34D399", // Emerald
  svm: "#EAB308",                 // Gold
  random_forest: "#34D399",       // Emerald
  extra_trees: "#14B8A6",
  hist_gradient_boosting: "#F97316",
  xgboost: "#FBBF24",             // Amber
  qsvc: "#F472B6",                // Pink
  vqc: "#F97316",                 // Orange
  qnn: "#FB7185",                 // Rose
};

const MODEL_LABELS = {
  logistic_regression: "Logistic Regression",
  svm: "Classical SVM (RBF)",
  random_forest: "Random Forest",
  extra_trees: "Extra Trees",
  hist_gradient_boosting: "Histogram Gradient Boosting",
  xgboost: "XGBoost",
  qsvc: "Quantum SVC (4-Qubit)",
  vqc: "Variational Quantum Clf (VQC)",
  qnn: "Quantum Neural Network (QNN)",
};

function getInferenceFeatureGroup(featureName) {
  const name = String(featureName).trim().toLowerCase();
  if (name.startsWith('mean ')) return 'Mean measurements';
  if (name.startsWith('worst ')) return 'Largest measurements';
  if (name.includes('error')) return 'Measurement variation';
  return 'Other features';
}

export default function App() {
  const { theme } = useTheme();
  const {
    datasets,
    selectedDataset,
    setSelectedDataset,
    experiments,
    loading,
    refreshDatasets: fetchDatasets,
    refreshExperiments: fetchExperiments,
  } = useDashboard();
  const [activeTab, setActiveTab] = useState('overview');

  // Tab 1 state
  const [selectedConfusionModel, setSelectedConfusionModel] = useState('svm');
  const [visibleRocModels, setVisibleRocModels] = useState({
    logistic_regression: true,
    svm: true,
    random_forest: true,
    extra_trees: true,
    hist_gradient_boosting: true,
    xgboost: true,
    qsvc: true,
    vqc: true,
    qnn: true,
  });

  // Tab 2 state
  const [selectedExplainModel, setSelectedExplainModel] = useState('xgboost');
  const [explanationData, setExplanationData] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);

  // Tab 3 state (Noise study)
  const [robustnessStudy, setRobustnessStudy] = useState(null);

  // Tab 4 state (Run Experiment)
  const [newExp, setNewExp] = useState({
    model_type: 'svm',
    n_features: 4,
    backend: 'ideal_simulator',
    circuit_depth: 1,
    cv_folds: 5,
    seeds: '42, 123, 7',
  });
  const [isTraining, setIsTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState('');

  // Upload modal state
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showDatasetCatalog, setShowDatasetCatalog] = useState(false);
  const [datasetCatalog, setDatasetCatalog] = useState([]);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [importingDataset, setImportingDataset] = useState('');
  const [catalogError, setCatalogError] = useState('');
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadTargetCol, setUploadTargetCol] = useState('');
  const [uploadName, setUploadName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');

  // Empirical tuning comparisons state
  const [comparisonsData, setComparisonsData] = useState([]);
  const [showComparisons, setShowComparisons] = useState(false);

  const fetchComparisons = async () => {
    try {
      const res = await apiFetch(`${API_BASE}/api/comparisons`);
      if (res.ok) {
        const data = await res.json();
        setComparisonsData(data.comparison_summary || data.comparisons || []);
      }
    } catch (e) {
      console.error("Failed to fetch empirical comparisons", e);
    }
  };

  const openDatasetCatalog = async () => {
    setShowDatasetCatalog(true);
    setCatalogError('');
    setCatalogLoading(true);
    try {
      const response = await apiFetch('/datasets/catalog');
      const data = await response.json();
      setDatasetCatalog(Array.isArray(data.datasets) ? data.datasets : []);
    } catch (error) {
      setCatalogError(error.message || 'Could not load the public dataset catalog.');
    } finally {
      setCatalogLoading(false);
    }
  };

  const importPublicDataset = async (dataset) => {
    setImportingDataset(dataset.key);
    setCatalogError('');
    try {
      const response = await apiFetch(`/datasets/catalog/${encodeURIComponent(dataset.key)}/import`, { method: 'POST' });
      const result = await response.json();
      await fetchDatasets();
      setSelectedDataset(String(result.dataset.id));
      setShowDatasetCatalog(false);
    } catch (error) {
      setCatalogError(error.message || 'Could not import this dataset.');
    } finally {
      setImportingDataset('');
    }
  };

  // 1. Initial Data Fetch
  useEffect(() => {
    fetchRobustnessStudy();
    fetchPatients();
    fetchComparisons();
  }, []);

  // Tab 1 Threshold Tuning state (Gap 1)
  const [activeThreshold, setActiveThreshold] = useState(0.5);
  const [thresholdMetrics, setThresholdMetrics] = useState(null);

  // Tab 5 / Prediction state (Gap 2)
  const [predictSubTab, setPredictSubTab] = useState('single_sample'); // 'report_analysis' | 'single_sample'
  const [predictModel, setPredictModel] = useState('svm');
  const [predictResult, setPredictResult] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [inferenceValues, setInferenceValues] = useState({});
  const [predictError, setPredictError] = useState('');

  const activeInferenceDataset = datasets.find(dataset => String(dataset.id) === String(selectedDataset));
  const inferenceFeatureNames = activeInferenceDataset?.feature_names || [];
  const inferenceFeatureGroups = inferenceFeatureNames.reduce((groups, name, index) => {
    const groupName = getInferenceFeatureGroup(name);
    if (!groups[groupName]) groups[groupName] = [];
    groups[groupName].push({ name, index });
    return groups;
  }, {});
  const enteredInferenceCount = inferenceFeatureNames.filter(name => {
    const value = inferenceValues[name];
    return value !== '' && value != null;
  }).length;
  const missingInferenceCount = Math.max(0, inferenceFeatureNames.length - enteredInferenceCount);

  useEffect(() => {
    setInferenceValues({});
    setPredictResult(null);
    setPredictError('');
  }, [selectedDataset]);

  useEffect(() => {
    updateMetricsAtThreshold(activeThreshold);
  }, [activeThreshold, selectedConfusionModel, experiments]);

  const updateMetricsAtThreshold = async (threshold) => {
    const exp = experiments.find(e => e.model_type === selectedConfusionModel);
    if (!exp) return;
    try {
      const res = await apiFetch(`${API_BASE}/experiments/${exp.id}/metrics-at-threshold?threshold=${threshold}`);
      if (res.ok) {
        const data = await res.json();
        setThresholdMetrics(data);
      } else {
        setThresholdMetrics(null);
      }
    } catch (e) {
      setThresholdMetrics(null);
    }
  };

  const handleRunPredict = async () => {
    const featureNames = activeInferenceDataset?.feature_names || [];
    if (!featureNames.length) {
      setPredictError('The selected dataset does not expose a supported numeric feature schema.');
      return;
    }
    const sampleFeatures = featureNames.map(name => {
      const raw = inferenceValues[name];
      return raw === '' || raw == null ? null : Number(raw);
    });
    if (sampleFeatures.every(value => value == null)) {
      setPredictError('Enter at least one feature value before running inference.');
      return;
    }
    if (sampleFeatures.some(value => value != null && !Number.isFinite(value))) {
      setPredictError('Each entered feature must be a finite number.');
      return;
    }
    setPredictError('');
    setPredictResult(null);
    setPredictLoading(true);
    try {
      const res = await apiFetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selectedDataset,
          model_type: predictModel,
          features: sampleFeatures,
          threshold: activeThreshold,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Inference could not be completed. Check the feature values and try again.');
      setPredictResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
      setPredictError(err.message || 'Inference failed. Please try again.');
    } finally {
      setPredictLoading(false);
    }
  };

  // --- Patient Details Module State & Handlers (Part A) ---
  const [patientsList, setPatientsList] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [selectedPatientDetail, setSelectedPatientDetail] = useState(null);
  const [patientLoading, setPatientLoading] = useState(false);
  const [patientSearch, setPatientSearch] = useState('');
  const [showNewPatientModal, setShowNewPatientModal] = useState(false);
  const [newPatientData, setNewPatientData] = useState({ name: '', age: '', sex: 'Female', notes: '' });
  const [showNewRecordModal, setShowNewRecordModal] = useState(false);
  const [newRecordPreset, setNewRecordPreset] = useState('malignant');

  const fetchPatients = async () => {
    try {
      const res = await apiFetch(`${API_BASE}/patients`);
      if (res.ok) {
        const data = await res.json();
        setPatientsList(data.patients || []);
      }
    } catch (e) {
      console.error("Failed to fetch patients", e);
    }
  };

  const fetchPatientDetail = async (patientId) => {
    setPatientLoading(true);
    try {
      const detailRes = await apiFetch(`${API_BASE}/patients/${patientId}`);
      if (detailRes.ok) {
        const d = await detailRes.json();
        setSelectedPatientDetail(d.patient);
      }
    } catch (e) {
      console.error("Failed to fetch patient detail/trend", e);
    } finally {
      setPatientLoading(false);
    }
  };

  const handleCreatePatient = async (e) => {
    e.preventDefault();
    try {
      const res = await apiFetch(`${API_BASE}/patients`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newPatientData.name || "De-identified Subject",
          age: newPatientData.age ? parseInt(newPatientData.age) : null,
          sex: newPatientData.sex,
          notes: newPatientData.notes
        })
      });
      if (res.ok) {
        setShowNewPatientModal(false);
        setNewPatientData({ name: '', age: '', sex: 'Female', notes: '' });
        await fetchPatients();
      }
    } catch (e) {
      console.error("Failed to create patient", e);
    }
  };

  const handleAddPatientRecord = async () => {
    if (!selectedPatientId) return;
    let feat;
    if (newRecordPreset === 'malignant') {
      feat = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189];
    } else if (newRecordPreset === 'intermediate') {
      feat = [14.26, 19.65, 97.83, 629.9, 0.07837, 0.2233, 0.3009, 0.07798, 0.1704, 0.07769, 0.3661, 1.026, 2.744, 29.04, 0.005266, 0.03846, 0.04323, 0.01507, 0.01288, 0.003886, 15.3, 23.73, 107.0, 709.0, 0.1082, 0.3761, 0.4346, 0.1487, 0.2298, 0.08762];
    } else {
      feat = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259];
    }

    try {
      const recRes = await apiFetch(`${API_BASE}/patients/${selectedPatientId}/records`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          feature_values: feat,
          source: `WDBC research sample ${newRecordPreset.toUpperCase()}`
        })
      });
      if (recRes.ok) {
        setShowNewRecordModal(false);
        await recRes.json();
        await fetchPatientDetail(selectedPatientId);
        await fetchPatients();
      }
    } catch (e) {
      console.error("Failed to add patient record", e);
    }
  };

  const fetchRobustnessStudy = async () => {
    try {
      const res = await apiFetch(`${API_BASE}/experiments/robustness-study`);
      const data = await res.json();
      if (data.study) setRobustnessStudy(data.study);
    } catch (e) {
      console.error("Failed to fetch robustness study", e);
    }
  };

  const fetchExplanation = async (modelType) => {
    const exp = experiments.find(e => e.model_type === modelType);
    if (!exp) return;
    setExplainLoading(true);
    try {
      const res = await apiFetch(`${API_BASE}/explain/${exp.id}`);
      const data = await res.json();
      if (data.explanation) {
        setExplanationData(data.explanation);
      }
    } catch (e) {
      console.error("Failed to fetch explanation", e);
    } finally {
      setExplainLoading(false);
    }
  };

  const handleUploadDataset = async (e) => {
    e.preventDefault();
    if (!uploadFile) {
      setUploadError("Please select a CSV file.");
      return;
    }
    setUploading(true);
    setUploadError('');

    const formData = new FormData();
    formData.append("file", uploadFile);
    if (uploadName) formData.append("name", uploadName);
    if (uploadTargetCol) formData.append("target_column", uploadTargetCol);

    try {
      const res = await apiFetch(`${API_BASE}/datasets/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setShowUploadModal(false);
        setUploadFile(null);
        setUploadName('');
        setUploadTargetCol('');
        await fetchDatasets();
        setSelectedDataset(data.dataset.id);
      } else {
        setUploadError(data.detail || "Upload failed");
      }
    } catch (err) {
      setUploadError(err.message || "Network error");
    } finally {
      setUploading(false);
    }
  };

  const handleRunExperiment = async (e) => {
    e.preventDefault();
    setIsTraining(true);
    setTrainProgress(`Initializing ${MODEL_LABELS[newExp.model_type]} training across ${newExp.cv_folds} folds...`);

    const seedList = newExp.seeds
      .split(',')
      .map(s => parseInt(s.trim()))
      .filter(s => !isNaN(s));

    try {
      const res = await apiFetch(`${API_BASE}/experiments/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selectedDataset,
          model_type: newExp.model_type,
          n_features: parseInt(newExp.n_features),
          backend: newExp.backend,
          circuit_depth: parseInt(newExp.circuit_depth),
          cv_folds: parseInt(newExp.cv_folds),
          seeds: seedList.length > 0 ? seedList : [42],
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setTrainProgress(`Training completed in ${data.runtime_seconds.toFixed(1)}s! Updating benchmark tables.`);
        await fetchExperiments(selectedDataset);
        setTimeout(() => {
          setIsTraining(false);
          setActiveTab('benchmarks');
        }, 1200);
      } else {
        setTrainProgress(`Error: ${data.detail || "Execution failed"}`);
        setIsTraining(false);
      }
    } catch (err) {
      setTrainProgress(`Error: ${err.message}`);
      setIsTraining(false);
    }
  };

  // Find top metric values for highlighting
  const bestMetrics = React.useMemo(() => {
    const keys = ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"];
    const best = {};
    keys.forEach(k => {
      let maxVal = -1;
      let bestModel = null;
      experiments.forEach(exp => {
        const val = exp.metrics?.[k]?.mean;
        if (typeof val === 'number' && val > maxVal) {
          maxVal = val;
          bestModel = exp.id;
        }
      });
      best[k] = bestModel;
    });
    return best;
  }, [experiments]);

  // Construct ROC Curve multi-series data
  const rocChartData = React.useMemo(() => {
    // Collect 100 interpolated points from 0 to 1
    const points = [];
    for (let i = 0; i <= 20; i++) {
      const fprVal = i / 20;
      const point = { fpr: Number(fprVal.toFixed(2)) };

      experiments.forEach(exp => {
        const curve = exp.roc_curve;
        if (curve && curve.fpr && curve.tpr && curve.fpr.length > 0) {
          // Find closest tpr for this fpr
          let closestTpr = 0;
          for (let j = 0; j < curve.fpr.length; j++) {
            if (curve.fpr[j] <= fprVal) {
              closestTpr = curve.tpr[j];
            } else {
              break;
            }
          }
          point[exp.model_type] = Number(closestTpr.toFixed(3));
        }
      });
      points.push(point);
    }
    return points;
  }, [experiments]);

  // Confusion matrix for selected model
  const selectedConfusionExp = experiments.find(e => e.model_type === selectedConfusionModel);
  const cm = selectedConfusionExp?.confusion_matrix || { tn: 0, fp: 0, fn: 0, tp: 0 };
  const cmTotal = (cm.tn || 0) + (cm.fp || 0) + (cm.fn || 0) + (cm.tp || 0);

  const navigationTabs = APP_TABS;

  return (
    <div className={`app-root theme-${theme} min-h-screen flex flex-col font-sans`}>
      
      <ClinicalNotice />
      <AppHeader datasets={datasets} selectedDataset={selectedDataset} onDatasetChange={setSelectedDataset} onUpload={() => setShowUploadModal(true)} onBrowseDatasets={openDatasetCatalog} showResearchDatasetControls={['benchmarks', 'run', 'predict', 'explain', 'robustness'].includes(activeTab)} />
      <AppNavigation tabs={navigationTabs} activeTab={activeTab} onChange={setActiveTab} />

      {/* 4. MAIN CONTENT AREA */}
      <main className="flex-1 w-full mx-auto">

        {/* ========================================================
            TAB 0: PLATFORM OVERVIEW & SCIENTIFIC FOUNDATION (LANDING PAGE)
        ======================================================== */}
        {activeTab === 'overview' && (
          <div id="panel-overview" role="region" aria-label="Platform overview" className="w-full">
            <LandingPage
              onExplorePlatform={(target) => setActiveTab(typeof target === 'string' ? target : 'lab_reports')}
            />
          </div>
        )}

        {activeTab !== 'overview' && (
          <div id={`panel-${activeTab}`} role="region" aria-label={navigationTabs.find(tab => tab.id === activeTab)?.fullLabel} className="max-w-7xl mx-auto p-4 sm:p-6 space-y-6">

        {activeTab === 'guide' && <HowToVideo onNavigate={(target) => target === 'upload' ? setShowUploadModal(true) : setActiveTab(target)} />}

        {/* ========================================================
            TAB: LAB REPORT ANALYZER & MULTI-DISEASE RISK DIFFERENTIAL
        ======================================================== */}
        {activeTab === 'lab_reports' && (
          <PatientHealthReportAnalysis />
        )}

        {/* ========================================================
            TAB 1: MODEL COMPARISON & BENCHMARKS
        ======================================================== */}
        {activeTab === 'benchmarks' && (
          <div className="max-w-7xl mx-auto p-4 sm:p-6 space-y-6">

            {experiments.some(experiment => experiment.requires_rerun_for_label_semantics) && (
              <div role="status" className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm leading-5 text-amber-100">
                Some saved breast-cancer runs use the earlier reversed class labels. Their accuracy is unchanged, but class-specific metrics and interpretations are stale. Rerun those benchmarks to refresh them.
              </div>
            )}
            
            {/* Top Metrics Cards Row — computed from actual experiment data */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {(() => {
                const classicalTypes = ['logistic_regression', 'svm', 'random_forest', 'extra_trees', 'hist_gradient_boosting', 'xgboost'];
                const quantumTypes = ['qsvc', 'vqc', 'qnn'];
                const classicalExps = experiments.filter(e => classicalTypes.includes(e.model_type));
                const quantumExps = experiments.filter(e => quantumTypes.includes(e.model_type));

                // Best classical accuracy
                let bestClassical = null;
                classicalExps.forEach(e => {
                  const acc = e.metrics?.accuracy?.mean;
                  if (acc != null && (!bestClassical || acc > bestClassical.metrics.accuracy.mean))
                    bestClassical = e;
                });

                // Best quantum accuracy
                let bestQuantum = null;
                quantumExps.forEach(e => {
                  const acc = e.metrics?.accuracy?.mean;
                  if (acc != null && (!bestQuantum || acc > bestQuantum.metrics.accuracy.mean))
                    bestQuantum = e;
                });

                // Best quantum sensitivity
                let bestQSens = null;
                quantumExps.forEach(e => {
                  const sens = e.metrics?.sensitivity?.mean;
                  if (sens != null && (!bestQSens || sens > bestQSens.metrics.sensitivity.mean))
                    bestQSens = e;
                });

                // Speed ratio: fastest classical runtime / slowest quantum runtime
                const classicalRuntimes = classicalExps.map(e => e.runtime_seconds).filter(Boolean);
                const quantumRuntimes = quantumExps.map(e => e.runtime_seconds).filter(Boolean);
                const minClassical = classicalRuntimes.length > 0 ? Math.min(...classicalRuntimes) : null;
                const maxQuantum = quantumRuntimes.length > 0 ? Math.max(...quantumRuntimes) : null;
                const speedRatio = (minClassical != null && maxQuantum != null && minClassical > 0)
                  ? Math.round(maxQuantum / minClassical)
                  : null;

                return (
                  <>
                    <div className="rounded-2xl bg-[#0D1424]/75 border border-slate-800/80 hover:border-sky-500/40 p-5 backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-sky-500/10 group">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-slate-400 font-medium">Top Classical Accuracy</span>
                        <span className="w-2 h-2 rounded-full bg-sky-400" />
                      </div>
                      <div className="text-3xl font-extrabold font-mono text-sky-400 mt-2 tracking-tight">
                        {bestClassical ? `${(bestClassical.metrics.accuracy.mean * 100).toFixed(1)}%` : '—'}
                      </div>
                      <span className="text-[11px] text-slate-400 mt-1 block font-mono">
                        {bestClassical ? `${MODEL_LABELS[bestClassical.model_type] || bestClassical.model_type}` : 'No classical runs'}
                      </span>
                    </div>

                    <div className="rounded-2xl bg-[#0D1424]/75 border border-slate-800/80 hover:border-pink-500/40 p-5 backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-pink-500/10 group">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-slate-400 font-medium">Top Quantum Accuracy</span>
                        <span className="w-2 h-2 rounded-full bg-pink-400" />
                      </div>
                      <div className="text-3xl font-extrabold font-mono text-pink-400 mt-2 tracking-tight">
                        {bestQuantum ? `${(bestQuantum.metrics.accuracy.mean * 100).toFixed(1)}%` : '—'}
                      </div>
                      <span className="text-[11px] text-slate-400 mt-1 block font-mono">
                        {bestQuantum ? `${MODEL_LABELS[bestQuantum.model_type] || bestQuantum.model_type}` : 'No quantum runs'}
                      </span>
                    </div>

                    <div className="rounded-2xl bg-[#0D1424]/75 border border-slate-800/80 hover:border-emerald-500/40 p-5 backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-emerald-500/10 group">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-slate-400 font-medium">Top Quantum Sensitivity</span>
                        <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      </div>
                      <div className="text-3xl font-extrabold font-mono text-emerald-400 mt-2 tracking-tight">
                        {bestQSens ? `${(bestQSens.metrics.sensitivity.mean * 100).toFixed(1)}%` : '—'}
                      </div>
                      <span className="text-[11px] text-slate-400 mt-1 block font-mono">
                        {bestQSens ? `Early recall on positives` : 'No quantum runs'}
                      </span>
                    </div>

                    <div className="rounded-2xl bg-[#0D1424]/75 border border-slate-800/80 hover:border-amber-500/40 p-5 backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-amber-500/10 group">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-slate-400 font-medium">Classical Speedup Factor</span>
                        <span className="w-2 h-2 rounded-full bg-amber-400" />
                      </div>
                      <div className="text-3xl font-extrabold font-mono text-amber-400 mt-2 tracking-tight">
                        {speedRatio != null ? `~${speedRatio}× Faster` : '—'}
                      </div>
                      <span className="text-[11px] text-slate-400 mt-1 block font-mono">
                        {(minClassical != null && maxQuantum != null)
                          ? `${minClassical.toFixed(1)}s (CPU) vs ${maxQuantum.toFixed(1)}s (QPU)`
                          : 'Run both to compare'}
                      </span>
                    </div>
                  </>
                );
              })()}
            </div>

            {/* Side-by-Side Metric Comparison Table */}
            <div className="rounded-2xl bg-[#0D1424]/80 border border-slate-800/90 overflow-hidden backdrop-blur-xl shadow-2xl">
              <div className="px-6 py-4 border-b border-slate-800 flex justify-between items-center">
                <div>
                  <h2 className="text-base font-semibold text-white flex items-center gap-2">
                    <BarChart2 className="w-5 h-5 text-cyan-400" />
                    Standardized Benchmark Table (Stratified 5-Fold CV)
                  </h2>
                  <p className="text-xs text-slate-400">
                    All 7 architectures evaluated under strictly identical zero-leakage splits and random seeds.
                  </p>
                </div>
                <button
                  onClick={() => fetchExperiments(selectedDataset)}
                  className="p-2 rounded-xl bg-slate-900/80 hover:bg-slate-800 text-slate-300 border border-slate-800 transition-all cursor-pointer"
                  title="Refresh Table"
                >
                  <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-[#121929] text-slate-300 font-semibold uppercase tracking-wider text-[11px] border-b border-slate-800 font-mono">
                    <tr>
                      <th className="py-3 px-4">Architecture</th>
                      <th className="py-3 px-4">Family</th>
                      <th className="py-3 px-3 text-right">Accuracy</th>
                      <th className="py-3 px-3 text-right">Precision</th>
                      <th className="py-3 px-3 text-right">Sensitivity</th>
                      <th className="py-3 px-3 text-right">Specificity</th>
                      <th className="py-3 px-3 text-right">F1 Score</th>
                      <th className="py-3 px-3 text-right">ROC-AUC</th>
                      <th className="py-3 px-3 text-right">PR-AUC</th>
                      <th className="py-3 px-4 text-right">Runtime</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/80 font-mono">
                    {experiments.length === 0 ? (
                      <tr>
                        <td colSpan={10} className="text-center py-8 text-slate-400">
                          No experiments logged yet. Run a model in Tab 4!
                        </td>
                      </tr>
                    ) : (
                      experiments.map(exp => {
                        const m = exp.metrics;
                        const isQuantum = ['qsvc', 'vqc', 'qnn'].includes(exp.model_type);
                        return (
                          <tr key={exp.id} className="hover:bg-slate-800/30 transition-colors">
                            <td className="py-3 px-4 font-sans font-medium text-slate-200 flex items-center gap-2">
                              <span
                                className="w-2.5 h-2.5 rounded-full"
                                style={{ backgroundColor: MODEL_COLORS[exp.model_type] || '#94A3B8' }}
                              />
                              {MODEL_LABELS[exp.model_type] || exp.model_type}
                            </td>
                            <td className="py-3 px-4 font-sans">
                              <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                                isQuantum 
                                  ? 'bg-purple-950/80 text-purple-300 border border-purple-800/50' 
                                  : 'bg-blue-950/80 text-blue-300 border border-blue-800/50'
                              }`}>
                                {isQuantum ? 'Quantum (NISQ)' : 'Classical'}
                              </span>
                            </td>

                            {/* Metrics Columns with Top Performer Badges */}
                            {[
                              { key: 'accuracy', val: m?.accuracy },
                              { key: 'precision', val: m?.precision },
                              { key: 'sensitivity', val: m?.sensitivity },
                              { key: 'specificity', val: m?.specificity },
                              { key: 'f1', val: m?.f1 },
                              { key: 'roc_auc', val: m?.roc_auc },
                              { key: 'pr_auc', val: m?.pr_auc },
                            ].map(col => {
                              const isTop = bestMetrics[col.key] === exp.id;
                              const numStr = col.val?.mean != null ? col.val.mean.toFixed(4) : "N/A";
                              return (
                                <td key={col.key} className="py-3 px-3 text-right">
                                  <span className={isTop ? "text-emerald-400 font-bold bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-800/50" : "text-slate-300"}>
                                    {numStr}
                                  </span>
                                </td>
                              );
                            })}

                            <td className="py-3 px-4 text-right text-slate-400 font-sans">
                              {exp.runtime_seconds != null ? `${exp.runtime_seconds.toFixed(1)}s` : 'N/A'}
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Empirical Model Tuning & Statistical Significance Section */}
            <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-slate-800">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-indigo-950/60 border border-indigo-700/50 rounded-lg text-indigo-400">
                    <Scale className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                      Empirical Model Tuning & Statistical Significance Study
                      <span className="text-[10px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-800/60 px-2 py-0.5 rounded-full">
                        Stratified 5-Fold CV (Paired t-test)
                      </span>
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Direct pre- vs post-tuning measurements across classical hyperparameter sweeps, feature selectors, quantum circuit depth, and voting ensembling.
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setShowComparisons(!showComparisons)}
                  className="px-3 py-1.5 bg-[#1E293B] hover:bg-slate-700 border border-slate-700 rounded-lg text-xs font-medium text-slate-200 flex items-center gap-2 transition-all self-start md:self-center"
                >
                  <ArrowUpDown className="w-3.5 h-3.5 text-indigo-400" />
                  {showComparisons ? "Hide Study Matrix" : "View Study Matrix"}
                </button>
              </div>

              {showComparisons && (
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-[#1E293B]/70 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                      <tr>
                        <th className="py-2.5 px-3 font-semibold">Model / Intervention</th>
                        <th className="py-2.5 px-3 font-semibold">Metric</th>
                        <th className="py-2.5 px-3 text-right font-semibold">Before Tuning</th>
                        <th className="py-2.5 px-3 text-right font-semibold">After Tuning</th>
                        <th className="py-2.5 px-3 text-right font-semibold">&Delta; Change</th>
                        <th className="py-2.5 px-3 text-right font-semibold">p-value</th>
                        <th className="py-2.5 px-3 text-center font-semibold">Statistically Significant?</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {comparisonsData.length === 0 ? (
                        <tr>
                          <td colSpan={7} className="py-6 text-center text-slate-500 font-sans">
                            Loading model comparison records...
                          </td>
                        </tr>
                      ) : (
                        comparisonsData.map((row, idx) => {
                          const isSig = row["Statistically Significant?"]?.startsWith("Yes");
                          const changePositive = row.Change?.startsWith("+");
                          const changeNegative = row.Change?.startsWith("-");

                          return (
                            <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                              <td className="py-2.5 px-3 font-sans font-medium text-slate-200">
                                {row["Model / Intervention"]}
                              </td>
                              <td className="py-2.5 px-3 text-slate-400">
                                {row.Metric}
                              </td>
                              <td className="py-2.5 px-3 text-right text-slate-300">
                                {typeof row.Before === "number" ? (row.Before * 100).toFixed(2) + "%" : row.Before}
                              </td>
                              <td className="py-2.5 px-3 text-right text-slate-200 font-bold">
                                {typeof row.After === "number" ? (row.After * 100).toFixed(2) + "%" : row.After}
                              </td>
                              <td className={`py-2.5 px-3 text-right font-bold ${
                                changePositive ? "text-emerald-400" : changeNegative ? "text-rose-400" : "text-slate-300"
                              }`}>
                                {row.Change}
                              </td>
                              <td className="py-2.5 px-3 text-right text-slate-400 font-mono">
                                {typeof row["p-value"] === "number" ? row["p-value"].toFixed(4) : row["p-value"]}
                              </td>
                              <td className="py-2.5 px-3 text-center">
                                <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
                                  isSig
                                    ? "bg-emerald-950/80 text-emerald-300 border border-emerald-800/60"
                                    : "bg-slate-800/80 text-slate-400 border border-slate-700/60"
                                }`}>
                                  {row["Statistically Significant?"]}
                                </span>
                              </td>
                            </tr>
                          );
                        })
                      )}
                    </tbody>
                  </table>
                  <div className="mt-3 p-3 bg-[#0A1020] rounded-lg border border-slate-800 text-[11px] text-slate-400 flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
                    <span>
                      * Rigorous two-tailed paired t-tests conducted across identical 5 folds. Significance threshold: &alpha; = 0.05.
                    </span>
                    <span className="text-slate-500 font-mono">
                      Source: <span className="text-slate-300">tuning_study_results.json</span>
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Interactive ROC and PR Curves */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              
              {/* ROC Curves Multi-overlay */}
              <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                      <TrendingDown className="w-4 h-4 text-cyan-400" />
                      Receiver Operating Characteristic (ROC) Curves
                    </h3>
                    <p className="text-[11px] text-slate-400">Overlaid performance across 7 architectures</p>
                  </div>
                </div>

                {/* Model Toggle Checkboxes */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {experiments.map(exp => (
                    <button
                      key={exp.model_type}
                      onClick={() => setVisibleRocModels(prev => ({
                        ...prev,
                        [exp.model_type]: !prev[exp.model_type]
                      }))}
                      className={`text-[10px] px-2 py-0.5 rounded-full border transition-all flex items-center gap-1 ${
                        visibleRocModels[exp.model_type]
                          ? 'border-transparent text-white'
                          : 'border-slate-700 bg-slate-800/40 text-slate-400 line-through'
                      }`}
                      style={{
                        backgroundColor: visibleRocModels[exp.model_type]
                          ? MODEL_COLORS[exp.model_type] + "33"
                          : undefined,
                        borderColor: visibleRocModels[exp.model_type]
                          ? MODEL_COLORS[exp.model_type]
                          : undefined,
                      }}
                    >
                      <span
                        className="w-1.5 h-1.5 rounded-full"
                        style={{ backgroundColor: MODEL_COLORS[exp.model_type] }}
                      />
                      {MODEL_LABELS[exp.model_type] || exp.model_type}
                    </button>
                  ))}
                </div>

                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={rocChartData} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                      <XAxis dataKey="fpr" stroke="#64748B" tick={{ fontSize: 11 }} label={{ value: 'False Positive Rate (FPR)', position: 'insideBottom', offset: -5, fontSize: 10, fill: '#94A3B8' }} />
                      <YAxis domain={[0, 1]} stroke="#64748B" tick={{ fontSize: 11 }} label={{ value: 'True Positive Rate (TPR)', angle: -90, position: 'insideLeft', offset: 25, fontSize: 10, fill: '#94A3B8' }} />
                      <Tooltip contentStyle={{ backgroundColor: '#0B132B', borderColor: '#334155', fontSize: '11px' }} />
                      <ReferenceLine stroke="#475569" strokeDasharray="3 3" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} />
                      
                      {experiments.map(exp => {
                        if (!visibleRocModels[exp.model_type]) return null;
                        return (
                          <Line
                            key={exp.model_type}
                            type="monotone"
                            dataKey={exp.model_type}
                            stroke={MODEL_COLORS[exp.model_type] || '#FFFFFF'}
                            strokeWidth={2}
                            dot={false}
                          />
                        );
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Confusion Matrix Visualizer */}
              <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                      <Layers className="w-4 h-4 text-purple-400" />
                      Confusion Matrix Visualizer
                    </h3>
                    <p className="text-[11px] text-slate-400">Held-out test set representative fold</p>
                  </div>
                  <select
                    className="bg-[#1E293B] border border-slate-700 text-xs rounded-lg px-2.5 py-1 text-slate-200 focus:outline-none"
                    value={selectedConfusionModel}
                    onChange={(e) => setSelectedConfusionModel(e.target.value)}
                  >
                    {experiments.map(exp => (
                      <option key={exp.model_type} value={exp.model_type}>
                        {MODEL_LABELS[exp.model_type] || exp.model_type}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Threshold Slider (Gap 1) */}
                <div className="mb-4 bg-[#0A1020] p-3 rounded-lg border border-slate-800">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                      <Sliders className="w-3.5 h-3.5 text-teal-400" />
                      Classification Threshold (&tau;):
                    </span>
                    <span className="text-xs font-mono font-bold text-teal-400 bg-teal-950/80 px-2 py-0.5 rounded border border-teal-800/40">
                      {activeThreshold.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.01"
                    max="0.99"
                    step="0.01"
                    value={activeThreshold}
                    onChange={(e) => setActiveThreshold(parseFloat(e.target.value))}
                    className="w-full accent-teal-400 bg-slate-700 h-1.5 rounded-lg cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
                    <span>0.01 (High Sensitivity / Early Recall)</span>
                    <span>0.50 (Default)</span>
                    <span>0.99 (High Specificity)</span>
                  </div>
                </div>

                {(() => {
                  const effectiveCm = thresholdMetrics?.confusion_matrix || cm;
                  const effTotal = (effectiveCm.tn || 0) + (effectiveCm.fp || 0) + (effectiveCm.fn || 0) + (effectiveCm.tp || 0);
                  const effSens = thresholdMetrics ? (thresholdMetrics.metrics.sensitivity * 100).toFixed(1) : ((effectiveCm.tp / ((effectiveCm.tp + effectiveCm.fn) || 1)) * 100).toFixed(1);
                  const effSpec = thresholdMetrics ? (thresholdMetrics.metrics.specificity * 100).toFixed(1) : ((effectiveCm.tn / ((effectiveCm.tn + effectiveCm.fp) || 1)) * 100).toFixed(1);
                  const effPrec = thresholdMetrics ? (thresholdMetrics.metrics.precision * 100).toFixed(1) : ((effectiveCm.tp / ((effectiveCm.tp + effectiveCm.fp) || 1)) * 100).toFixed(1);
                  const effF1 = thresholdMetrics ? (thresholdMetrics.metrics.f1 * 100).toFixed(1) : "—";

                  return (
                    <>
                      <div className="grid grid-cols-2 gap-3 p-2 bg-[#0A1020] rounded-xl border border-slate-800">
                        {/* True Negatives */}
                        <div className="bg-sky-950/30 border border-sky-800/40 rounded-lg p-3 text-center">
                          <span className="text-[10px] uppercase font-bold text-sky-400 tracking-wider">True Negative (TN)</span>
                          <div className="text-2xl font-extrabold text-white mt-0.5">{effectiveCm.tn}</div>
                          <span className="text-[11px] text-slate-400 mt-0.5 block">
                            {effTotal ? ((effectiveCm.tn / effTotal) * 100).toFixed(1) : 0}% of cohort
                          </span>
                        </div>

                        {/* False Positives */}
                        <div className="bg-rose-950/20 border border-rose-800/40 rounded-lg p-3 text-center">
                          <span className="text-[10px] uppercase font-bold text-rose-400 tracking-wider">False Positive (FP)</span>
                          <div className="text-2xl font-extrabold text-white mt-0.5">{effectiveCm.fp}</div>
                          <span className="text-[11px] text-slate-400 mt-0.5 block">
                            {effTotal ? ((effectiveCm.fp / effTotal) * 100).toFixed(1) : 0}% of cohort
                          </span>
                        </div>

                        {/* False Negatives */}
                        <div className="bg-amber-950/20 border border-amber-800/40 rounded-lg p-3 text-center">
                          <span className="text-[10px] uppercase font-bold text-amber-400 tracking-wider">False Negative (FN)</span>
                          <div className="text-2xl font-extrabold text-white mt-0.5">{effectiveCm.fn}</div>
                          <span className="text-[11px] text-slate-400 mt-0.5 block">
                            {effTotal ? ((effectiveCm.fn / effTotal) * 100).toFixed(1) : 0}% of cohort
                          </span>
                        </div>

                        {/* True Positives */}
                        <div className="bg-emerald-950/30 border border-emerald-800/40 rounded-lg p-3 text-center">
                          <span className="text-[10px] uppercase font-bold text-emerald-400 tracking-wider">True Positive (TP)</span>
                          <div className="text-2xl font-extrabold text-white mt-0.5">{effectiveCm.tp}</div>
                          <span className="text-[11px] text-slate-400 mt-0.5 block">
                            {effTotal ? ((effectiveCm.tp / effTotal) * 100).toFixed(1) : 0}% of cohort
                          </span>
                        </div>
                      </div>

                      <div className="mt-3 grid grid-cols-4 gap-2 text-center text-xs font-mono bg-[#0A1020] p-2.5 rounded-lg border border-slate-800/80">
                        <div>
                          <span className="text-slate-500 text-[10px] block">Sensitivity</span>
                          <strong className="text-emerald-400 font-bold">{effSens}%</strong>
                        </div>
                        <div>
                          <span className="text-slate-500 text-[10px] block">Specificity</span>
                          <strong className="text-sky-400 font-bold">{effSpec}%</strong>
                        </div>
                        <div>
                          <span className="text-slate-500 text-[10px] block">Precision</span>
                          <strong className="text-amber-400 font-bold">{effPrec}%</strong>
                        </div>
                        <div>
                          <span className="text-slate-500 text-[10px] block">F1 Score</span>
                          <strong className="text-purple-400 font-bold">{effF1}%</strong>
                        </div>
                      </div>
                    </>
                  );
                })()}
              </div>

            </div>

          </div>
        )}

        {/* ========================================================
            TAB 2: EXPLAINABILITY & INTERPRETABILITY
        ======================================================== */}
        {activeTab === 'explain' && (
          <div className="space-y-6">
            <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div>
                <h2 className="text-base font-semibold text-white flex items-center gap-2">
                  <Eye className="w-5 h-5 text-teal-400" />
                  Dual Explainability Engine
                </h2>
                <p className="text-xs text-slate-400">
                  SHAP feature attribution for classical models | Parameterized circuit structure & perturbation sensitivity for quantum models.
                </p>
              </div>

              {/* Model Picker */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400">Select Architecture:</span>
                <select
                  className="bg-[#1E293B] border border-slate-700 text-xs rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none"
                  value={selectedExplainModel}
                  onChange={(e) => setSelectedExplainModel(e.target.value)}
                >
                  {experiments.map(exp => (
                    <option key={exp.model_type} value={exp.model_type}>
                      {MODEL_LABELS[exp.model_type] || exp.model_type}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Explanation Content */}
            {explainLoading ? (
              <div className="p-12 text-center text-slate-400 flex flex-col items-center justify-center gap-3">
                <RefreshCw className="w-6 h-6 text-teal-400 animate-spin" />
                <span>Computing live model explanations...</span>
              </div>
            ) : explanationData ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                
                {/* Left Panel: Feature Importance / Sensitivity Bar Chart */}
                <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl">
                  <h3 className="text-sm font-semibold text-white mb-1 flex items-center gap-2">
                    <BarChart2 className="w-4 h-4 text-cyan-400" />
                    {explanationData.type === 'classical_shap'
                      ? 'SHAP Mean Absolute Feature Attribution'
                      : 'Quantum Perturbation Sensitivity (dP / dx_i)'}
                  </h3>
                  <p className="text-[11px] text-slate-400 mb-4">
                    {explanationData.disclaimer}
                  </p>

                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={explanationData.feature_importances || explanationData.feature_sensitivities || []}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                        <XAxis type="number" stroke="#64748B" tick={{ fontSize: 11 }} />
                        <YAxis type="category" dataKey="feature" stroke="#64748B" tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#0B132B', borderColor: '#334155', fontSize: '11px' }} />
                        <Bar
                          dataKey={explanationData.type === 'classical_shap' ? "importance" : "sensitivity"}
                          fill="#34D399"
                          radius={[0, 4, 4, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Right Panel: Quantum Circuit or Classical Weights Detail */}
                <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-white mb-1 flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-purple-400" />
                      {explanationData.type === 'classical_shap'
                        ? 'Classical Attribution Summary'
                        : 'Quantum Circuit Decomposition'}
                    </h3>
                    <p className="text-[11px] text-slate-400 mb-4">
                      {explanationData.type === 'classical_shap'
                        ? 'Global Shapley values aggregated over the held-out test split'
                        : `Encoding: ${explanationData.qubit_count} Qubits | Circuit Depth: ${explanationData.circuit_depth} | Total Gates: ${explanationData.total_gates}`}
                    </p>

                    {explanationData.circuit_diagram ? (
                      <div className="bg-[#070D1E] p-3 rounded-lg border border-slate-800 font-mono text-[11px] text-cyan-300 overflow-x-auto whitespace-pre leading-tight">
                        {explanationData.circuit_diagram}
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className="bg-[#070D1E] p-4 rounded-lg border border-slate-800 text-xs text-slate-300 leading-relaxed">
                          <p>
                            SHAP values estimate the average marginal contribution of each dimensionality-reduced component across the entire test cohort.
                          </p>
                          <p className="mt-2 text-cyan-400 font-mono">
                            Top predictive dimension: <strong>{explanationData.feature_importances?.[0]?.feature}</strong> (Importance: {explanationData.feature_importances?.[0]?.importance?.toFixed(3)})
                          </p>
                        </div>
                      </div>
                    )}
                  </div>

                  {explanationData.gate_breakdown && (
                    <div className="mt-4 pt-4 border-t border-slate-800 flex flex-wrap gap-2 text-xs font-mono">
                      {Object.entries(explanationData.gate_breakdown).map(([gate, cnt]) => (
                        <span key={gate} className="px-2 py-0.5 rounded bg-purple-950/60 border border-purple-800/40 text-purple-300">
                          {gate.toUpperCase()}: {cnt}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

              </div>
            ) : (
              <div className="p-8 text-center text-slate-400">Select a model to view explainability.</div>
            )}
          </div>
        )}

        {/* ========================================================
            TAB 3: NOISE & ROBUSTNESS STUDY
        ======================================================== */}
        {activeTab === 'robustness' && (
          <div className="space-y-6">
            <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl">
              <h2 className="text-base font-semibold text-white flex items-center gap-2">
                <Sliders className="w-5 h-5 text-teal-400" />
                Quantum Noise & Depth Sensitivity Analysis
              </h2>
              <p className="text-xs text-slate-400">
                Empirical investigation of depolarizing noise channels and circuit repetition scaling on quantum classification accuracy.
              </p>
            </div>

            {robustnessStudy ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                
                {/* 1. Performance vs Depolarizing Noise Level */}
                <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl">
                  <h3 className="text-sm font-semibold text-white mb-1 flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-rose-400" />
                    Performance vs. Depolarizing Error Rate (p)
                  </h3>
                  <p className="text-[11px] text-slate-400 mb-4">
                    Noise rate increased from p=0.0 (ideal) to p=0.10
                  </p>

                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={robustnessStudy.noise_sweep} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                        <XAxis dataKey="noise_rate" stroke="#64748B" tick={{ fontSize: 11 }} label={{ value: 'Error Rate (p)', position: 'insideBottom', offset: -5, fontSize: 10, fill: '#94A3B8' }} />
                        <YAxis domain={[0.6, 1.0]} stroke="#64748B" tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#0B132B', borderColor: '#334155', fontSize: '11px' }} />
                        <Legend />
                        <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke="#F43F5E" strokeWidth={2} dot={{ r: 4 }} />
                        <Line type="monotone" dataKey="f1" name="F1 Score" stroke="#A855F7" strokeWidth={2} dot={{ r: 4 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* 2. Performance vs Circuit Depth */}
                <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl">
                  <h3 className="text-sm font-semibold text-white mb-1 flex items-center gap-2">
                    <Layers className="w-4 h-4 text-amber-400" />
                    Performance vs. Circuit Depth (Repetitions)
                  </h3>
                  <p className="text-[11px] text-slate-400 mb-4">
                    ZZFeatureMap depth sweep (reps = 1, 2, 3, 4)
                  </p>

                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={robustnessStudy.depth_sweep} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                        <XAxis dataKey="circuit_depth" stroke="#64748B" tick={{ fontSize: 11 }} label={{ value: 'Repetitions (reps)', position: 'insideBottom', offset: -5, fontSize: 10, fill: '#94A3B8' }} />
                        <YAxis domain={[0.6, 1.0]} stroke="#64748B" tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#0B132B', borderColor: '#334155', fontSize: '11px' }} />
                        <Legend />
                        <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke="#F59E0B" strokeWidth={2} dot={{ r: 4 }} />
                        <Line type="monotone" dataKey="f1" name="F1 Score" stroke="#06B6D4" strokeWidth={2} dot={{ r: 4 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

              </div>
            ) : null}

            {/* Scientific Callout Box */}
            <div className="bg-gradient-to-r from-slate-900 to-[#121B33] border border-cyan-500/30 rounded-xl p-5 shadow-xl">
              <h3 className="text-sm font-bold text-cyan-300 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                What Does This Mean for Near-Term Quantum Deployment?
              </h3>
              <div className="mt-2 text-xs text-slate-300 space-y-2 leading-relaxed">
                <p>
                  <strong>1. The Depolarizing Flattening Bottleneck:</strong> As gate error rates climb past $p = 0.05$, the fidelity quantum Gram matrix decays toward uniform overlap ($1/2^n$), eroding decision boundary separability. Near-term hardware requires active error mitigation (zero-noise extrapolation or readout calibration) to preserve quantum kernel contrast.
                </p>
                <p>
                  <strong>2. Barren Plateau & Depth Congestion:</strong> Increasing circuit depth beyond 2 repetitions does not improve classification accuracy on 4 PCA features; rather, additional entangling gates introduce phase wrap congestion and gradient diffusion. Shallow circuits with linear entanglement provide optimal empirical NISQ stability.
                </p>
                <p className="text-slate-400 italic">
                  Grounding: This finding corroborates Gupta et al. (2025, npj Digital Medicine) and Huang et al. (2021, Nature Communications) demonstrating that scaling quantum depth without error-suppression guarantees produces no advantage over classical SVM baselines.
                </p>
              </div>
            </div>

          </div>
        )}

        {/* ========================================================
            TAB 4: RUN NEW EXPERIMENT
        ======================================================== */}
        {activeTab === 'run' && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-6 shadow-2xl">
              <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-800">
                <div className="w-10 h-10 rounded-lg bg-teal-500/10 border border-teal-500/30 flex items-center justify-center">
                  <Zap className="w-5 h-5 text-teal-400" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-white">Execute New Model Benchmark</h2>
                  <p className="text-xs text-slate-400">Configure parameters for reproducible, zero-leakage cross-validation</p>
                </div>
              </div>

              <form onSubmit={handleRunExperiment} className="space-y-5">
                {/* Dataset Picker */}
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1.5">Target Dataset</label>
                  <select
                    className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500"
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                  >
                    {datasets.map(d => (
                      <option key={d.id} value={d.id}>
                        {d.name} ({d.n_rows} samples, {d.n_cols} features)
                      </option>
                    ))}
                  </select>
                </div>

                {/* Model Architecture */}
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1.5">Model Architecture</label>
                  <select
                    className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500"
                    value={newExp.model_type}
                    onChange={(e) => setNewExp({ ...newExp, model_type: e.target.value })}
                  >
                    <optgroup label="Classical Baselines">
                      <option value="logistic_regression">Logistic Regression (L2 regularized)</option>
                      <option value="svm">Classical Support Vector Machine (RBF kernel)</option>
                      <option value="random_forest">Random Forest Classifier (400 trees)</option>
                      <option value="extra_trees">Extra Trees (balanced randomized trees)</option>
                      <option value="hist_gradient_boosting">Histogram Gradient Boosting</option>
                      <option value="xgboost">XGBoost (Gradient Boosted Trees)</option>
                    </optgroup>
                    <optgroup label="Hybrid Quantum Models">
                      <option value="qsvc">Quantum SVC (ZZFeatureMap Kernel)</option>
                      <option value="vqc">Variational Quantum Classifier (RealAmplitudes)</option>
                      <option value="qnn">Quantum Neural Network (EstimatorQNN)</option>
                    </optgroup>
                  </select>
                </div>

                {['vqc', 'qnn', 'qsvc'].includes(newExp.model_type) ? (
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-xs font-semibold text-slate-300">Quantum input dimensions</label>
                      <span className="text-xs font-mono font-bold text-teal-400">{newExp.n_features} qubits</span>
                    </div>
                    <input
                      type="range"
                      min="2"
                      max="8"
                      value={newExp.n_features}
                      onChange={(e) => setNewExp({ ...newExp, n_features: e.target.value })}
                      className="w-full accent-teal-400 bg-slate-700 h-1.5 rounded-lg cursor-pointer"
                    />
                    <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-500">
                      <span>2 · faster</span><span>4 · default</span><span>8 · higher circuit cost</span>
                    </div>
                  </div>
                ) : (
                  <div className="rounded-lg border border-teal-500/20 bg-teal-500/5 px-3 py-2.5 text-xs leading-5 text-slate-300">
                    Classical models use all {activeInferenceDataset?.feature_names?.length || Math.max(0, (datasets.find(d => String(d.id) === String(selectedDataset))?.n_cols || 1) - 1)} numeric input features. Preprocessing stays inside each cross-validation fold.
                  </div>
                )}

                {/* Quantum Circuit Depth Slider */}
                {['vqc', 'qnn', 'qsvc'].includes(newExp.model_type) && (
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-xs font-semibold text-slate-300">Circuit Depth (Repetitions)</label>
                      <span className="text-xs font-mono font-bold text-purple-400">{newExp.circuit_depth} Reps</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="4"
                      value={newExp.circuit_depth}
                      onChange={(e) => setNewExp({ ...newExp, circuit_depth: e.target.value })}
                      className="w-full accent-purple-400 bg-slate-700 h-1.5 rounded-lg cursor-pointer"
                    />
                  </div>
                )}

                {/* Backend Selector (for quantum models) */}
                {['vqc', 'qnn', 'qsvc'].includes(newExp.model_type) && (
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1.5">Simulation Backend</label>
                    <select
                      className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500"
                      value={newExp.backend}
                      onChange={(e) => setNewExp({ ...newExp, backend: e.target.value })}
                    >
                      <option value="ideal_simulator">Ideal Statevector Simulator (noiseless)</option>
                      <option value="noisy_simulator">Noisy Simulator (depolarizing noise)</option>
                    </select>
                  </div>
                )}

                {/* CV Folds and Seeds Grid */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1.5">Stratified CV Folds</label>
                    <input
                      type="number"
                      min="2"
                      max="10"
                      value={newExp.cv_folds}
                      onChange={(e) => setNewExp({ ...newExp, cv_folds: e.target.value })}
                      className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500 font-mono"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1.5">Random Seeds</label>
                    <input
                      type="text"
                      value={newExp.seeds}
                      onChange={(e) => setNewExp({ ...newExp, seeds: e.target.value })}
                      placeholder="42, 123, 7"
                      className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500 font-mono"
                    />
                  </div>
                </div>

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={isTraining}
                  className="w-full py-3 rounded-lg bg-gradient-to-r from-teal-500 to-indigo-600 hover:from-teal-400 hover:to-indigo-500 text-white font-semibold text-xs transition-all shadow-lg shadow-teal-500/20 disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isTraining ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin text-white" />
                      Executing Pipeline & Training Model...
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4 text-white" />
                      Train & Benchmark Model
                    </>
                  )}
                </button>

                {trainProgress && (
                  <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg text-xs font-mono text-cyan-300 text-center animate-fade-in">
                    {trainProgress}
                  </div>
                )}
              </form>
            </div>
          </div>
        )}

        {/* ========================================================
            TAB 5: CLINICAL RISK STRATIFICATION & INFERENCE (GAP 2)
        ======================================================== */}
        {activeTab === 'predict' && (
          <div className="analysis-page space-y-6">
            <div className="analysis-page-heading">
              <p className="analysis-eyebrow">WORKSPACE / ANALYSIS</p>
              <h2>{predictSubTab === 'report_analysis' ? 'Review a health report' : 'Review a sample'}</h2>
              <p>Follow the data from source to model output. Results are for research and require human review.</p>
            </div>
            {/* View Mode Sub-tabs Header */}
            <div className="analysis-mode-row flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
              <div className="analysis-mode-switcher flex items-center gap-1 rounded-xl border p-1" role="group" aria-label="Choose analysis workflow">
                <button
                  type="button"
                  aria-pressed={predictSubTab === 'report_analysis'}
                  onClick={() => setPredictSubTab('report_analysis')}
                  className={`analysis-mode-button flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all cursor-pointer ${
                    predictSubTab === 'report_analysis'
                      ? 'is-active'
                      : ''
                  }`}
                >
                  <FileSpreadsheet className="w-4 h-4" />
                  <span>Report review</span>
                </button>

                <button
                  type="button"
                  aria-pressed={predictSubTab === 'single_sample'}
                  onClick={() => setPredictSubTab('single_sample')}
                  className={`analysis-mode-button flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all cursor-pointer ${
                    predictSubTab === 'single_sample'
                      ? 'is-active'
                      : ''
                  }`}
                >
                  <HeartPulse className="w-4 h-4" />
                  <span>Single sample</span>
                </button>
              </div>

              <div className="analysis-research-tag flex items-center gap-2 text-xs font-medium">
                <span className="w-2 h-2 rounded-full" />
                <span>Research workflow · not for diagnosis</span>
              </div>
            </div>

            {predictSubTab === 'report_analysis' ? (
              <PatientHealthReportAnalysis />
            ) : (
              <div className="inference-workspace mx-auto w-full max-w-5xl space-y-5">
                <div className="inference-card rounded-2xl border p-5 shadow-sm sm:p-7">
                  <div className="mb-5 flex items-start gap-3 border-b pb-5">
                    <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
                      <HeartPulse className="w-5 h-5 text-emerald-400" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold">Review a sample</h2>
                      <p className="mt-1 text-sm">Enter cohort measurements, then review the model output and its limitations.</p>
                    </div>
                  </div>

                  <div className="inference-context mb-5 rounded-xl border p-4 sm:p-5">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <div className="text-sm font-semibold">{activeInferenceDataset?.name || 'No dataset selected'}</div>
                        <p className="mt-1 text-sm">
                          Outcome of interest: <strong>{activeInferenceDataset?.positive_class_label || 'Positive class'}</strong>
                          {activeInferenceDataset?.negative_class_label ? ` · Comparison: ${activeInferenceDataset.negative_class_label}` : ''}
                          {' · '}{inferenceFeatureNames.length} numeric features
                        </p>
                      </div>
                      <span className="inference-badge rounded-full border px-3 py-1.5 text-xs font-semibold">
                        Research use only
                      </span>
                    </div>
                    <p className="mt-3 text-sm leading-6">
                      Use measurements from the same source and units as this dataset. Blank values are imputed by the fitted pipeline; substantial missing data can make an output less informative.
                    </p>
                    <div className="inference-completeness mt-4 flex flex-col gap-2 rounded-xl border px-3 py-3 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <p className="text-sm font-semibold">Input completeness</p>
                        <p className="mt-0.5 text-xs">{enteredInferenceCount} of {inferenceFeatureNames.length} values entered · {missingInferenceCount} left blank</p>
                      </div>
                      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200 sm:w-40" role="progressbar" aria-label="Input completeness" aria-valuemin="0" aria-valuemax={inferenceFeatureNames.length || 1} aria-valuenow={enteredInferenceCount}>
                        <div className="h-full rounded-full bg-teal-600" style={{ width: `${inferenceFeatureNames.length ? (enteredInferenceCount / inferenceFeatureNames.length) * 100 : 0}%` }} />
                      </div>
                    </div>
                    {missingInferenceCount > inferenceFeatureNames.length / 2 && inferenceFeatureNames.length > 0 && (
                      <p className="inference-missing-note mt-3 rounded-lg border px-3 py-2 text-xs leading-5" role="note">
                        More than half of the feature values are blank. The model will fill these from training data, so treat any resulting score with particular caution.
                      </p>
                    )}
                    <h3 className="mt-5 mb-2 text-sm font-semibold">Feature values</h3>
                    <div className="inference-fields max-h-[520px] space-y-4 overflow-y-auto pr-1">
                      {Object.entries(inferenceFeatureGroups).map(([groupName, features]) => (
                        <fieldset key={groupName} className="inference-field-group rounded-xl border p-3 sm:p-4">
                          <legend className="px-1 text-xs font-semibold">{groupName} <span className="font-normal opacity-70">({features.length})</span></legend>
                          <div className="grid grid-cols-1 gap-3 pt-1 sm:grid-cols-2">
                            {features.map(({ name, index }) => (
                              <label key={name} className="inference-field min-w-0 rounded-xl border px-3 py-2.5">
                                <span className="mb-1.5 block truncate text-xs font-medium" title={name}>
                                  <span className="mr-2 font-mono opacity-60">{String(index + 1).padStart(2, '0')}</span>{name}
                                </span>
                                <input
                                  type="number"
                                  step="any"
                                  inputMode="decimal"
                                  value={inferenceValues[name] ?? ''}
                                  onChange={event => setInferenceValues(current => ({ ...current, [name]: event.target.value }))}
                                  placeholder="Enter a value"
                                  aria-label={name}
                                  className="inference-input w-full rounded-lg border px-3 py-2.5 text-sm focus:border-teal-500 focus:outline-none focus:ring-2 focus:ring-teal-500/20"
                                />
                              </label>
                            ))}
                          </div>
                        </fieldset>
                      ))}
                    </div>
                    {!!activeInferenceDataset?.feature_names?.length && (
                      <div className="mt-3 flex justify-end border-t border-slate-700 pt-3 text-xs text-slate-400">
                        <button
                          type="button"
                          onClick={() => { setInferenceValues({}); setPredictResult(null); setPredictError(''); }}
                          className="rounded-md px-2.5 py-1.5 font-semibold text-teal-300 hover:bg-teal-400/10"
                        >
                          Clear values
                        </button>
                      </div>
                    )}
                    {!activeInferenceDataset?.feature_names?.length && (
                      <p className="mt-3 text-sm text-amber-300">Feature definitions are unavailable for this dataset. Choose another dataset or upload a numeric CSV.</p>
                    )}
                  </div>

                  <div className="inference-settings mb-5 grid grid-cols-1 gap-4 rounded-xl border p-4 sm:grid-cols-2">
                    <div>
                      <label className="mb-1.5 block text-sm font-semibold text-slate-300" htmlFor="inference-model">Model</label>
                      <select
                        id="inference-model"
                        className="w-full rounded-lg border border-slate-700 bg-[#1A2640] px-3 py-2.5 text-sm text-white focus:border-teal-400 focus:outline-none"
                        value={predictModel}
                        onChange={event => { setPredictModel(event.target.value); setPredictResult(null); }}
                      >
                        <option value="svm">Support Vector Machine</option>
                        <option value="logistic_regression">Logistic Regression</option>
                        <option value="random_forest">Random Forest</option>
                        <option value="extra_trees">Extra Trees</option>
                        <option value="hist_gradient_boosting">Histogram Gradient Boosting</option>
                        <option value="xgboost">XGBoost</option>
                        <option value="qsvc">Quantum SVC</option>
                      </select>
                      <p className="mt-1 text-xs">Use held-out benchmark results to compare models; a single sample score does not show model accuracy.</p>
                    </div>
                    <div>
                      <label className="mb-1.5 flex items-center justify-between text-sm font-semibold text-slate-300" htmlFor="inference-threshold">
                        <span>Score cutoff</span>
                        <span className="font-mono text-teal-300">{activeThreshold.toFixed(2)}</span>
                      </label>
                      <input
                        id="inference-threshold"
                        type="range"
                        min="0.05"
                        max="0.95"
                        step="0.01"
                        value={activeThreshold}
                        aria-valuetext={activeThreshold.toFixed(2)}
                        onChange={event => setActiveThreshold(Number(event.target.value))}
                        className="mt-2 w-full accent-teal-400"
                      />
                      <p className="mt-1 text-xs">Changes which class is displayed. This is a model setting, not a clinical cutoff.</p>
                    </div>
                  </div>

                  {predictError && (
                    <div role="alert" className="mb-4 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2.5 text-sm text-rose-200">
                      {predictError}
                    </div>
                  )}

                  {/* Run Inference Button */}
                  <button
                    onClick={handleRunPredict}
                    disabled={predictLoading || !activeInferenceDataset?.feature_names?.length}
                    className="w-full rounded-xl bg-sky-700 py-3.5 text-sm font-bold text-white shadow-sm transition-colors hover:bg-sky-800 focus-visible:outline-sky-600 disabled:cursor-not-allowed disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {predictLoading ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Preparing model and evaluating inputs…
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4" />
                        Review model output
                      </>
                    )}
                  </button>
                </div>

                {/* Inference Result Card */}
                {predictResult && (
                  <div className="inference-card rounded-2xl border p-5 shadow-sm sm:p-7 space-y-5">
                    <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3 pb-4 border-b border-slate-800">
                      <div>
                        <span className="text-xs font-semibold uppercase tracking-wide">Model output · {activeInferenceDataset?.name || 'Selected dataset'}</span>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="inference-class-label rounded-full border px-3 py-1.5 text-sm font-semibold">
                            {predictResult.probability >= activeThreshold ? predictResult.positive_class_label : predictResult.negative_class_label}
                          </span>
                          <span className="text-xs text-slate-400">
                            Shown using score cutoff {activeThreshold.toFixed(2)}
                          </span>
                        </div>
                      </div>

                      <div className="text-right">
                        <span className="text-xs block">Model score · 0–1 scale</span>
                        <span className="text-3xl font-extrabold text-white font-mono">
                          {predictResult.probability.toFixed(3)}
                        </span>
                      </div>
                    </div>

                    {/* Probability Bar */}
                    <div>
                      <div className="flex justify-between text-xs text-slate-400 mb-1 font-mono">
                        <span>0 · {predictResult.negative_class_label}</span>
                        <span>Score cutoff: {activeThreshold.toFixed(2)}</span>
                        <span>1 · {predictResult.positive_class_label}</span>
                      </div>
                      <div className="inference-score-track w-full h-2.5 rounded-full overflow-hidden relative" aria-label="Model score scale">
                        <div
                          className="inference-score-fill h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${predictResult.probability * 100}%`,
                          }}
                        />
                        <div
                          className="inference-score-cutoff absolute top-0 bottom-0 w-0.5"
                          style={{ left: `${activeThreshold * 100}%` }}
                          title={`Active threshold: ${activeThreshold}`}
                        />
                      </div>
                    </div>

                    <div className="inference-result-note rounded-xl border p-4 text-sm leading-6">
                      <strong>For research review only.</strong> This uncalibrated score is not a probability or diagnosis and is not validated for clinical use. Review the cohort, data completeness, and held-out benchmark results. A qualified clinician should evaluate any health concern.
                    </div>
                    <p className="text-xs">Input used: {enteredInferenceCount} entered, {missingInferenceCount} imputed or blank feature values. Model: {MODEL_LABELS[predictModel] || predictModel}.</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ========================================================
            TAB: PATIENT DETAILS & LONGITUDINAL TRAJECTORY (PART A)
        ======================================================== */}
        {activeTab === 'patients' && (
          <div className="space-y-6">
            <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
              <div>
                <h2 className="text-base font-semibold text-white flex items-center gap-2">
                  <User className="w-5 h-5 text-teal-400" />
                  Patient Record & Longitudinal Disease Risk Module
                </h2>
                <p className="text-xs text-slate-400">
                  Store local research records. No patient-specific disease score is available in this build.
                </p>
              </div>

              <button
                onClick={() => setShowNewPatientModal(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 transition-all shadow-sm"
              >
                <UserPlus className="w-4 h-4" />
                Register Subject
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left Panel: Searchable Patient Directory */}
              <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <Database className="w-4 h-4 text-cyan-400" />
                    Patient Directory ({patientsList.length})
                  </h3>
                  <button
                    onClick={fetchPatients}
                    className="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800"
                    title="Refresh List"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                  </button>
                </div>

                <input
                  type="text"
                  placeholder="Search by ID or name..."
                  value={patientSearch}
                  onChange={(e) => setPatientSearch(e.target.value)}
                  className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-teal-500"
                />

                <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
                  {patientsList
                    .filter(p => p.name.toLowerCase().includes(patientSearch.toLowerCase()) || p.patient_id.toLowerCase().includes(patientSearch.toLowerCase()))
                    .map(p => {
                      const isSelected = selectedPatientId === p.patient_id;
                      return (
                        <div
                          key={p.patient_id}
                          onClick={() => {
                            setSelectedPatientId(p.patient_id);
                            fetchPatientDetail(p.patient_id);
                          }}
                          className={`p-3 rounded-lg border cursor-pointer transition-all ${
                            isSelected
                              ? 'bg-teal-950/40 border-teal-500/60 shadow-md'
                              : 'bg-[#121B30] border-slate-800 hover:border-slate-700 hover:bg-[#16223D]'
                          }`}
                        >
                          <div className="flex items-start justify-between">
                            <div>
                              <span className="text-xs font-bold text-white block">{p.name}</span>
                              <span className="text-[10px] font-mono text-slate-400 block truncate max-w-[180px]">
                                {p.patient_id}
                              </span>
                            </div>
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">
                              {p.record_count || 0} visits
                            </span>
                          </div>

                          <div className="flex items-center gap-3 mt-2 text-[10px] text-slate-400 font-sans">
                            <span>Age: {p.age ?? 'N/A'}</span>
                            <span>Sex: {p.sex}</span>
                            <span className="text-slate-500">{p.date_registered ? p.date_registered.slice(0, 10) : ''}</span>
                          </div>
                        </div>
                      );
                    })}

                  {patientsList.length === 0 && (
                    <div className="text-center py-8 text-xs text-slate-500">
                      No patients registered yet. Click &quot;Register Subject&quot; above.
                    </div>
                  )}
                </div>
              </div>

              {/* Right Panel: Selected Patient Detail, Visit History & Longitudinal Chart */}
              <div className="lg:col-span-2 space-y-6">
                {patientLoading ? (
                  <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-12 text-center text-slate-400 flex flex-col items-center justify-center gap-3">
                    <RefreshCw className="w-6 h-6 text-teal-400 animate-spin" />
                    <span>Loading patient records & trajectory...</span>
                  </div>
                ) : selectedPatientDetail ? (
                  <>
                    {/* Patient Header Card */}
                    <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-base font-bold text-white">{selectedPatientDetail.name}</h3>
                          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700">
                            {selectedPatientDetail.patient_id.slice(0, 8)}...
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Age: {selectedPatientDetail.age ?? 'N/A'} | Sex: {selectedPatientDetail.sex} | Registered: {selectedPatientDetail.date_registered?.slice(0, 10)}
                        </p>
                        {selectedPatientDetail.notes && (
                          <p className="text-xs text-slate-400 italic mt-1 bg-slate-900/60 p-2 rounded border border-slate-800">
                            &quot;{selectedPatientDetail.notes}&quot;
                          </p>
                        )}
                      </div>

                      <button
                        onClick={() => setShowNewRecordModal(true)}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30 transition-all shadow-sm"
                      >
                        <PlusCircle className="w-4 h-4 text-emerald-400" />
                        Add Research Sample
                      </button>
                    </div>

                    {/* Longitudinal Trajectory Plot */}
                    <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                            <Activity className="w-4 h-4 text-teal-400" />
                            Longitudinal Risk Prediction
                          </h4>
                          <p className="text-[11px] text-slate-400">
                            Unavailable until a patient-specific model is independently validated
                          </p>
                        </div>
                      </div>

                      <div className="p-8 text-center text-xs text-slate-500 font-mono">
                        Patient risk scores are disabled. Research cohort results are not patient-specific predictions.
                      </div>
                    </div>

                    {/* Visit Records & Prediction Audit Trail Timeline */}
                    <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl space-y-4">
                      <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                        <Calendar className="w-4 h-4 text-cyan-400" />
                        Research Sample Records ({selectedPatientDetail.records?.length || 0})
                      </h4>

                      <div className="space-y-3">
                        {selectedPatientDetail.records?.map((rec, idx) => {
                          return (
                            <div key={rec.record_id} className="p-4 rounded-lg bg-[#0B1326] border border-slate-800 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                              <div>
                                <div className="flex items-center gap-2">
                                  <span className="text-xs font-bold text-white">Research record #{idx + 1}</span>
                                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                                    {rec.recorded_at.slice(0, 19).replace('T', ' ')}
                                  </span>
                                  <span className="text-[10px] text-teal-400 italic">
                                    [{rec.source}]
                                  </span>
                                </div>
                                <p className="text-[11px] font-mono text-slate-400 mt-1">
                                    Stored feature values: {rec.feature_values?.length} · no patient risk prediction
                                </p>
                              </div>

                              <span className="rounded border border-slate-700 px-3 py-1 text-xs text-slate-400">Prediction disabled</span>
                            </div>
                          );
                        })}

                        {(!selectedPatientDetail.records || selectedPatientDetail.records.length === 0) && (
                          <div className="text-center py-6 text-xs text-slate-500">
                          No research samples recorded. Add a sample to inspect stored feature values.
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-16 text-center text-slate-400">
                    <User className="w-10 h-10 text-slate-600 mx-auto mb-3" />
                    <h3 className="text-sm font-semibold text-white">Select a Patient Record</h3>
                    <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
                      Choose any patient from the directory on the left to inspect longitudinal disease trajectory and past clinical visits.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        </div>
        )}

      </main>

      {/* MODAL: REGISTER NEW PATIENT */}
      {showNewPatientModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-6 max-w-md w-full shadow-2xl space-y-4">
            <div className="flex justify-between items-center border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <UserPlus className="w-4 h-4 text-teal-400" />
                Register New Patient Subject
              </h3>
              <button
                onClick={() => setShowNewPatientModal(false)}
                className="text-slate-400 hover:text-white text-sm"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreatePatient} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Subject Name or Identifier</label>
                <input
                  type="text"
                  placeholder="e.g. Subject-QD-8821"
                  value={newPatientData.name}
                  onChange={(e) => setNewPatientData({ ...newPatientData, name: e.target.value })}
                  className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">Age</label>
                  <input
                    type="number"
                    min="0"
                    max="120"
                    placeholder="e.g. 52"
                    value={newPatientData.age}
                    onChange={(e) => setNewPatientData({ ...newPatientData, age: e.target.value })}
                    className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">Biological Sex</label>
                  <select
                    value={newPatientData.sex}
                    onChange={(e) => setNewPatientData({ ...newPatientData, sex: e.target.value })}
                    className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
                  >
                    <option value="Female">Female</option>
                    <option value="Male">Male</option>
                    <option value="Other">Other</option>
                    <option value="Unknown">Unknown</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Clinical Notes (Optional)</label>
                <textarea
                  placeholder="Clinical context, cohort notes, or referral info..."
                  value={newPatientData.notes}
                  onChange={(e) => setNewPatientData({ ...newPatientData, notes: e.target.value })}
                  rows={3}
                  className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
                />
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowNewPatientModal(false)}
                  className="px-3 py-1.5 text-xs text-slate-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-1.5 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-semibold text-xs"
                >
                  Save Subject
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* MODAL: ADD CLINICAL VISIT RECORD */}
      {showNewRecordModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-6 max-w-md w-full shadow-2xl space-y-4">
            <div className="flex justify-between items-center border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <PlusCircle className="w-4 h-4 text-emerald-400" />
                Save Research Feature Sample
              </h3>
              <button
                onClick={() => setShowNewRecordModal(false)}
                className="text-slate-400 hover:text-white text-sm"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Built-in WDBC research sample</label>
                <select
                  value={newRecordPreset}
                  onChange={(e) => setNewRecordPreset(e.target.value)}
                  className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
                >
                  <option value="malignant">Research sample A</option>
                  <option value="intermediate">Research sample B</option>
                  <option value="benign">Research sample C</option>
                </select>
                <span className="text-[10px] text-slate-500 mt-1 block">
                  Example cohort values only. These are not a patient's blood report or a clinical visit.
                </span>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowNewRecordModal(false)}
                  className="px-3 py-1.5 text-xs text-slate-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleAddPatientRecord}
                  className="px-4 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-xs"
                >
                  Save Research Sample
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 5. UPLOAD DATASET MODAL */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-6 max-w-md w-full shadow-2xl space-y-4">
            <div className="flex justify-between items-center border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <FileSpreadsheet className="w-4 h-4 text-teal-400" />
                Upload Healthcare CSV Dataset
              </h3>
              <button
                onClick={() => setShowUploadModal(false)}
                className="text-slate-400 hover:text-white text-sm"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleUploadDataset} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Select CSV File</label>
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setUploadFile(e.target.files[0])}
                  className="w-full text-xs text-slate-300 file:mr-3 file:py-2 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-teal-500/10 file:text-teal-300 hover:file:bg-teal-500/20 cursor-pointer"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Dataset Name (Optional)</label>
                <input
                  type="text"
                  value={uploadName}
                  onChange={(e) => setUploadName(e.target.value)}
                  placeholder="e.g. Cardiac Arrhythmia Panel"
                  className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Target Binary Column (Optional)</label>
                <input
                  type="text"
                  value={uploadTargetCol}
                  onChange={(e) => setUploadTargetCol(e.target.value)}
                  placeholder="Auto-detected if left empty (e.g. diagnosis, target)"
                  className="w-full bg-[#1A2640] border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
                />
              </div>

              {uploadError && (
                <div className="text-xs text-rose-400 bg-rose-950/40 p-2.5 rounded border border-rose-800">
                  {uploadError}
                </div>
              )}

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowUploadModal(false)}
                  className="px-3 py-1.5 text-xs text-slate-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={uploading}
                  className="px-4 py-1.5 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-semibold text-xs disabled:opacity-50 flex items-center gap-1.5"
                >
                  {uploading ? "Validating..." : "Upload & Validate"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showDatasetCatalog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm" role="presentation">
          <section className={`w-full max-w-3xl space-y-4 rounded-2xl border p-5 shadow-2xl sm:p-6 ${theme === 'dark' ? 'border-slate-700 bg-slate-900' : 'border-slate-200 bg-white'}`} role="dialog" aria-modal="true" aria-labelledby="public-dataset-title">
            <div className={`flex items-start justify-between gap-4 border-b pb-4 ${theme === 'dark' ? 'border-slate-700' : 'border-slate-200'}`}>
              <div>
                <h2 id="public-dataset-title" className={`text-lg font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Public disease datasets</h2>
                <p className={`mt-1 text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>Import a cited UCI benchmark into your local workspace. Imports require an internet connection.</p>
              </div>
              <button type="button" onClick={() => setShowDatasetCatalog(false)} aria-label="Close dataset catalog" className="rounded-lg px-2 py-1 text-slate-500 hover:bg-slate-100">✕</button>
            </div>

            {catalogError && <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800" role="alert">{catalogError}</div>}
            {catalogLoading ? (
              <p className="py-8 text-center text-sm text-slate-600">Loading catalog…</p>
            ) : (
              <div className="max-h-[65vh] space-y-3 overflow-y-auto pr-1">
                {datasetCatalog.map(dataset => (
                  <article key={dataset.key} className={`rounded-xl border p-4 ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>{dataset.name}</h3>
                          <span className="rounded-full bg-sky-100 px-2 py-0.5 text-[11px] font-medium text-sky-800">{dataset.disease_area}</span>
                          {dataset.imported && <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[11px] font-medium text-emerald-800">Added</span>}
                        </div>
                        <p className={`mt-2 text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>{dataset.description}</p>
                        <p className={`mt-2 text-xs ${theme === 'dark' ? 'text-slate-200' : 'text-slate-700'}`}><strong>Positive label:</strong> {dataset.target_definition}</p>
                        <p className={`mt-1 text-xs ${theme === 'dark' ? 'text-amber-300' : 'text-amber-800'}`}><strong>Study caveat:</strong> {dataset.caveat}</p>
                        <p className="mt-2 text-xs text-slate-500">{dataset.citation} · {dataset.license} · <a className="font-medium text-sky-700 underline" href={dataset.source_url} target="_blank" rel="noreferrer">Source record</a></p>
                      </div>
                      <button type="button" onClick={() => importPublicDataset(dataset)} disabled={Boolean(importingDataset)} className="shrink-0 rounded-lg bg-sky-700 px-3.5 py-2 text-sm font-semibold text-white hover:bg-sky-800 disabled:cursor-wait disabled:opacity-60">
                        {importingDataset === dataset.key ? 'Importing…' : dataset.imported ? 'Select dataset' : 'Import dataset'}
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            )}
            <p className="border-t border-slate-200 pt-3 text-xs text-slate-500">Research benchmarks only. Dataset availability and licensing are governed by each source; imported records are not clinical decision tools.</p>
          </section>
        </div>
      )}

      <AppFooter />

    </div>
  );
}


