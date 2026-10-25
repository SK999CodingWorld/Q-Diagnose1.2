import { useEffect, useMemo, useState } from 'react'
import { api } from './api'
import type {
  Circuit,
  ComparisonResponse,
  Dataset,
  Explanation,
  Experiment,
  Health,
  Metrics,
  SweepResponse,
} from './types'

const modelLabels: Record<string, string> = {
  logistic_regression: 'Logistic regression',
  svm: 'Classical SVM',
  random_forest: 'Random forest',
  xgboost: 'XGBoost',
  qsvc: 'QSVC',
  vqc: 'VQC',
  qnn: 'Quantum neural network',
}

const metricLabels = [
  ['accuracy', 'Accuracy'],
  ['precision', 'Precision'],
  ['sensitivity', 'Sensitivity'],
  ['specificity', 'Specificity'],
  ['f1', 'F1 score'],
  ['roc_auc', 'ROC-AUC'],
  ['pr_auc', 'PR-AUC'],
] as const

const LEGACY_MODEL_TYPE_ALIASES: Record<string, string> = { classical_svm: 'svm' }

const SUPPORTED_BACKENDS = ['ideal_simulator', 'noisy_simulator', 'real_hardware'] as const

const replayModelType = (value: string): string =>
  value in modelLabels ? value : LEGACY_MODEL_TYPE_ALIASES[value] ?? value

const replayBackend = (value: unknown): string => {
  if (typeof value === 'string') {
    if ((SUPPORTED_BACKENDS as readonly string[]).includes(value)) return value
    if (value === 'classical') return 'ideal_simulator'
    return 'ideal_simulator'
  }
  if (value && typeof value === 'object' && 'noiseless' in value) {
    return (value as { noiseless?: unknown }).noiseless === false ? 'noisy_simulator' : 'ideal_simulator'
  }
  return 'ideal_simulator'
}

const backendLabels: Record<string, string> = {
  ideal_simulator: 'Ideal simulator',
  noisy_simulator: 'Noisy simulator',
  real_hardware: 'Real hardware',
}

const backendLabel = (value: unknown): string => {
  const normalized = replayBackend(value)
  return backendLabels[normalized] ?? normalized
}

type FormState = {
  dataset_id: string
  model_type: string
  backend: string
  n_features: string
  circuit_depth: string
  cv_folds: string
  seeds: string
  optimizer: string
  max_iterations: string
  shots: string
  noise_multiplier: string
  test_size: string
}

const initialForm: FormState = {
  dataset_id: '',
  model_type: 'logistic_regression',
  backend: 'ideal_simulator',
  n_features: '4',
  circuit_depth: '2',
  cv_folds: '5',
  seeds: '42, 123, 2025',
  optimizer: 'cobyla',
  max_iterations: '25',
  shots: '512',
  noise_multiplier: '0.01',
  test_size: '0.2',
}

const metricMean = (metrics: Metrics, name: string): number | null => {
  const value = metrics[name]
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (value && typeof value === 'object' && 'mean' in value) {
    const mean = Number(value.mean)
    return Number.isFinite(mean) ? mean : null
  }
  return null
}

const metricStd = (metrics: Metrics, name: string): number | null => {
  const value = metrics[name]
  if (value && typeof value === 'object' && 'std' in value) {
    const std = Number(value.std)
    return Number.isFinite(std) ? std : null
  }
  return null
}

const labelFor = (value: string): string =>
  modelLabels[value] ?? modelLabels[LEGACY_MODEL_TYPE_ALIASES[value]] ?? value.replace(/_/g, ' ')

const formatNumber = (value: number, digits = 3): string => value.toFixed(digits)

const formatMetric = (value: number | null, digits = 3): string =>
  value === null ? 'n/a' : value.toFixed(digits)

const formatSeconds = (value: number | null | undefined): string =>
  value === null || value === undefined || !Number.isFinite(Number(value))
    ? 'not recorded'
    : `${Number(value).toFixed(2)}s`

const positiveClassNote = (config: Record<string, unknown> | undefined): string => {
  const dataset = (config?.dataset ?? {}) as Record<string, unknown>
  const positive = dataset.positive_class
  const label = dataset.positive_class_label
  const column = dataset.target_column
  if (positive === undefined || positive === null) {
    return 'Positive class for sensitivity, specificity, precision and F1 is not recorded for this run.'
  }
  const described = label === undefined || label === null || String(label) === String(positive)
    ? `class ${String(positive)}`
    : `class ${String(positive)} (${String(label)})`
  return `Sensitivity, specificity, precision and F1 treat ${described} as the positive class${column ? ` from column ${String(column)}` : ''}.`
}

function MetricCards({ metrics, config }: { metrics: Metrics; config?: Record<string, unknown> }) {
  return (
    <div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-7">
        {metricLabels.map(([name, label]) => {
          const mean = metricMean(metrics, name)
          const std = metricStd(metrics, name)
          const tone = mean === null ? 'none' : mean >= 0.85 ? 'good' : mean >= 0.65 ? 'warn' : 'bad'
          const toneClass = tone === 'none'
            ? 'text-slate-400'
            : tone === 'good'
              ? 'text-good'
              : tone === 'warn'
                ? 'text-warn'
                : 'text-bad'
          return (
            <div className="metric-card" key={name}>
              <p className="text-[10px] uppercase tracking-[0.18em] text-slate-400">{label}</p>
              <p className={`mt-2 text-2xl font-semibold ${toneClass}`}>{formatMetric(mean)}</p>
              <p className="mt-1 text-[11px] text-slate-500">
                {std === null ? 'spread not recorded' : `± ${formatMetric(std)}`}
              </p>
            </div>
          )
        })}
      </div>
      <p className="mt-2 text-[11px] text-slate-500">{positiveClassNote(config)}</p>
    </div>
  )
}

function CurveChart({ title, x, y }: { title: string; x: Array<number | null>; y: Array<number | null> }) {
  const width = 360
  const height = 180
  const points = (values: Array<number | null>, horizontal: boolean) => {
    let path = ''
    values.forEach((value, index) => {
      if (value === null || !Number.isFinite(value)) return
      const px = horizontal ? (index / Math.max(values.length - 1, 1)) * width : (1 - value) * height
      const py = horizontal ? (1 - value) * height : (index / Math.max(values.length - 1, 1)) * height
      path += `${path ? ' L' : 'M'} ${px.toFixed(2)} ${py.toFixed(2)}`
    })
    return path
  }
  return (
    <div className="chart-card">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-medium text-slate-200">{title}</h3>
        <span className="text-[10px] uppercase tracking-widest text-accent">live metric</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-44 w-full overflow-visible">
        <line x1="0" y1={height} x2={width} y2={height} className="stroke-line" />
        <line x1="0" y1="0" x2="0" y2={height} className="stroke-line" />
        <line x1="0" y1={height / 2} x2={width} y2={height / 2} className="stroke-dash" />
        <line x1={width / 2} y1="0" x2={width / 2} y2={height} className="stroke-dash" />
        <path d={points(y, true)} className="chart-line chart-line-accent" />
        <path d={points(x, true)} className="chart-line chart-line-violet" />
      </svg>
      <div className="mt-1 flex justify-between text-[10px] text-slate-500">
        <span>0</span><span>0.5</span><span>1</span>
      </div>
    </div>
  )
}

function Confusion({ matrix }: { matrix: number[][] | null | undefined }) {
  if (!matrix || matrix.length === 0) {
    return (
      <div className="chart-card">
        <h3 className="mb-2 text-sm font-medium text-slate-200">Representative confusion matrix</h3>
        <p className="text-sm text-slate-500">
          No confusion matrix was recorded for this run, so no cell values are shown.
        </p>
      </div>
    )
  }
  const values = matrix.flat()
  const max = Math.max(1, ...values)
  return (
    <div className="chart-card">
      <h3 className="mb-4 text-sm font-medium text-slate-200">Representative confusion matrix</h3>
      <div className="grid grid-cols-2 gap-2">
        {matrix.flatMap((row, rowIndex) => row.map((value, columnIndex) => (
          <div
            className="matrix-cell"
            key={`${rowIndex}-${columnIndex}`}
            style={{ backgroundColor: `rgba(34, 139, 76, ${0.08 + (value / max) * 0.65})` }}
          >
            <span className="text-xl font-semibold text-white">{value}</span>
            <span className="text-[10px] text-slate-300">T{rowIndex} P{columnIndex}</span>
          </div>
        )))}
      </div>
      <div className="mt-3 grid grid-cols-2 text-center text-[10px] uppercase tracking-widest text-slate-500">
        <span>Predicted 0</span><span>Predicted 1</span>
      </div>
    </div>
  )
}

function CalibrationPanel({ experiment }: { experiment: Experiment }) {
  const calibration = experiment.calibration
  if (!calibration) return null
  return (
    <div className="chart-card">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-medium text-slate-200">Probability calibration</h3>
        <span className="text-xs text-accent">Brier {formatNumber(calibration.brier_score, 4)}</span>
      </div>
      <div className="space-y-2">
        {calibration.bins.filter((bin) => bin.count > 0).map((bin) => (
          <div className="grid grid-cols-[42px_1fr_1fr] items-center gap-2 text-[10px] text-slate-400" key={bin.bin}>
            <span>{Math.round(bin.lower * 100)}–{Math.round(bin.upper * 100)}%</span>
            <div className="h-2 overflow-hidden rounded bg-slate-800">
              <div className="h-full rounded bg-violet" style={{ width: `${(bin.predicted_probability ?? 0) * 100}%` }} />
            </div>
            <div className="h-2 overflow-hidden rounded bg-slate-800">
              <div className="h-full rounded bg-accent" style={{ width: `${(bin.observed_frequency ?? 0) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 flex gap-4 text-[10px] text-slate-500">
        <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-violet" />Predicted</span>
        <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-accent" />Observed</span>
      </div>
    </div>
  )
}

function ExplanationPanel({ explanation }: { explanation: Explanation }) {
  const features = explanation.top_features ?? explanation.quantum?.feature_scores ?? []
  return (
    <div className="chart-card">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-medium text-slate-200">Feature attribution</h3>
        <span className="text-[10px] text-slate-500">{explanation.method ?? explanation.quantum?.method}</span>
      </div>
      {explanation.superseded && (
        <p className="mb-3 rounded border border-warn/40 bg-warn/10 p-2 text-[10px] leading-relaxed text-slate-200">
          Cached attribution computed under pipeline revision{' '}
          <span className="font-mono">{explanation.pipeline_revision ?? 'none recorded'}</span>; the current
          revision is <span className="font-mono">{explanation.current_pipeline_revision}</span>. Re-run this
          configuration to obtain current attributions.
        </p>
      )}
      <div className="space-y-3">
        {features.slice(0, 6).map((feature) => {
          const value = 'mean_absolute_contribution' in feature
            ? feature.mean_absolute_contribution
            : feature.mean_absolute_probability_change
          const max = Math.max(...features.map((item) => 'mean_absolute_contribution' in item
            ? item.mean_absolute_contribution
            : item.mean_absolute_probability_change), 0.0001)
          return (
            <div key={feature.feature}>
              <div className="mb-1 flex justify-between text-xs text-slate-300">
                <span>{feature.feature}</span><span className="text-accent">{formatNumber(value, 4)}</span>
              </div>
              <div className="h-1.5 rounded bg-slate-800">
                <div className="h-full rounded bg-gradient-to-r from-violet to-accent" style={{ width: `${(value / max) * 100}%` }} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function CircuitPanel({ circuit }: { circuit: Circuit }) {
  return (
    <div className="chart-card overflow-hidden">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-medium text-slate-200">Circuit preview</h3>
        {circuit.quantum && <span className="badge badge-accent">{circuit.qubits} qubits</span>}
      </div>
      {circuit.quantum ? (
        <pre className="max-h-80 overflow-auto whitespace-pre text-[10px] leading-4 text-slate-300">{circuit.combined}</pre>
      ) : (
        <p className="text-xs text-slate-500">{circuit.message}</p>
      )}
    </div>
  )
}

function ExperimentForm({
  form,
  setForm,
  datasets,
  busy,
  onSubmit,
}: {
  form: FormState
  setForm: React.Dispatch<React.SetStateAction<FormState>>
  datasets: Dataset[]
  busy: boolean
  onSubmit: () => void
}) {
  const update = (key: keyof FormState, value: string) => setForm((current) => ({ ...current, [key]: value }))
  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <Field label="Dataset">
          <select value={form.dataset_id} onChange={(event) => update('dataset_id', event.target.value)}>
            <option value="">Select a dataset</option>
            {datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}
          </select>
        </Field>
        <Field label="Model">
          <select value={form.model_type} onChange={(event) => update('model_type', event.target.value)}>
            {Object.entries(modelLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </Field>
        <Field label="Backend">
          <select value={form.backend} onChange={(event) => update('backend', event.target.value)}>
            <option value="ideal_simulator">Ideal simulator</option>
            <option value="noisy_simulator">Noisy simulator</option>
            <option value="real_hardware">Real hardware</option>
          </select>
        </Field>
        <Field label="PCA features">
          <input type="number" min="1" max="32" value={form.n_features} onChange={(event) => update('n_features', event.target.value)} />
        </Field>
        <Field label="Circuit depth">
          <input type="number" min="1" max="12" value={form.circuit_depth} onChange={(event) => update('circuit_depth', event.target.value)} />
        </Field>
        <Field label="CV folds">
          <input type="number" min="3" max="20" value={form.cv_folds} onChange={(event) => update('cv_folds', event.target.value)} />
        </Field>
        <Field label="Random seeds">
          <input value={form.seeds} onChange={(event) => update('seeds', event.target.value)} placeholder="42, 123, 2025" />
        </Field>
        <Field label="Optimizer">
          <select value={form.optimizer} onChange={(event) => update('optimizer', event.target.value)}>
            <option value="cobyla">COBYLA</option>
            <option value="spsa">SPSA</option>
          </select>
        </Field>
        <Field label="Iterations">
          <input type="number" min="10" max="500" value={form.max_iterations} onChange={(event) => update('max_iterations', event.target.value)} />
        </Field>
        <Field label="Shots">
          <input type="number" min="1" value={form.shots} onChange={(event) => update('shots', event.target.value)} />
        </Field>
        <Field label="Noise multiplier">
          <input type="number" min="0" max="0.25" step="0.001" value={form.noise_multiplier} onChange={(event) => update('noise_multiplier', event.target.value)} />
        </Field>
        <Field label="Test fraction">
          <input type="number" min="0.05" max="0.5" step="0.05" value={form.test_size} onChange={(event) => update('test_size', event.target.value)} />
        </Field>
      </div>
      <button className="primary-button" disabled={busy || !form.dataset_id} onClick={onSubmit}>
        {busy ? 'Running experiment…' : 'Run experiment'}
      </button>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="field-label"><span>{label}</span>{children}</label>
}

function ExperimentList({ experiments, selectedId, onSelect }: { experiments: Experiment[]; selectedId: number | null; onSelect: (id: number) => void }) {
  return (
    <div className="space-y-2">
      {experiments.length === 0 && <p className="empty-state">No experiments yet. Run the first reproducible benchmark.</p>}
      {experiments.slice().reverse().map((experiment) => (
        <button
          className={`experiment-row ${selectedId === experiment.id ? 'selected' : ''}`}
          key={experiment.id}
          onClick={() => onSelect(experiment.id)}
        >
          <div className="min-w-0 text-left">
            <p className="truncate text-sm font-medium text-slate-100">{labelFor(experiment.model_type)}</p>
            <p className="mt-1 text-[11px] text-slate-500">Dataset {experiment.dataset_id} · {formatSeconds(experiment.runtime_seconds)}</p>
          </div>
          <div className="flex shrink-0 flex-col items-end gap-1">
            <span className={`badge ${experiment.status === 'completed' ? 'badge-good' : experiment.status === 'failed' ? 'badge-bad' : 'badge-warn'}`}>{experiment.status}</span>
            {experiment.superseded && <span className="badge badge-warn" title={experiment.pipeline_revision ? `Pipeline revision ${experiment.pipeline_revision}` : 'No pipeline revision recorded'}>superseded</span>}
          </div>
        </button>
      ))}
    </div>
  )
}

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [selected, setSelected] = useState<Experiment | null>(null)
  const [explanation, setExplanation] = useState<Explanation | null>(null)
  const [circuit, setCircuit] = useState<Circuit | null>(null)
  const [form, setForm] = useState<FormState>(initialForm)
  const [tab, setTab] = useState<'overview' | 'run' | 'compare'>('overview')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [sweepVariable, setSweepVariable] = useState('n_features')
  const [sweepValues, setSweepValues] = useState('2, 3, 4')
  const [sweep, setSweep] = useState<SweepResponse | null>(null)
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null)
  const [phase5Models, setPhase5Models] = useState<string[]>(['logistic_regression', 'svm'])
  const [file, setFile] = useState<File | null>(null)
  const [targetColumn, setTargetColumn] = useState('')

  const registerDiabetes = async () => {
    setBusy(true)
    try {
      const result = await api.createDiabetesDataset()
      setDatasets((current) => current.some((dataset) => dataset.id === result.id) ? current : [...current, result])
      setForm((current) => ({ ...current, dataset_id: String(result.id) }))
      setNotice(`${result.name} was registered with its explicit median-split label definition.`)
      setError('')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Dataset registration failed')
    } finally {
      setBusy(false)
    }
  }

  const refresh = async () => {
    try {
      const [healthData, datasetData, experimentData] = await Promise.all([api.health(), api.datasets(), api.experiments()])
      setHealth(healthData)
      setDatasets(datasetData)
      setExperiments(experimentData)
      setForm((current) => ({ ...current, dataset_id: current.dataset_id || String(datasetData[0]?.id ?? '') }))
      if (experimentData.length > 0) await selectExperiment(experimentData[experimentData.length - 1].id)
      setError('')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'API is unavailable')
    }
  }

  const selectExperiment = async (id: number) => {
    try {
      const detail = await api.experiment(id)
      setSelected(detail)
      setExplanation(null)
      setCircuit(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not load experiment')
    }
  }

  useEffect(() => { void refresh() }, [])

  const loadInsights = async (id: number) => {
    try {
      const [nextExplanation, nextCircuit] = await Promise.all([api.explanation(id), api.circuit(id)])
      setExplanation(nextExplanation)
      setCircuit(nextCircuit)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not load explanations')
    }
  }

  const run = async () => {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const payload = {
        dataset_id: Number(form.dataset_id),
        model_type: form.model_type,
        backend: form.backend,
        n_features: Number(form.n_features),
        circuit_depth: Number(form.circuit_depth),
        cv_folds: Number(form.cv_folds),
        seeds: form.seeds.split(',').map((value) => Number(value.trim())).filter(Number.isFinite),
        optimizer: form.optimizer,
        max_iterations: Number(form.max_iterations),
        shots: Number(form.shots),
        noise_multiplier: Number(form.noise_multiplier),
        test_size: Number(form.test_size),
      }
      const result = await api.createExperiment(payload)
      setExperiments((current) => [...current, result])
      setSelected(result)
      setNotice(`Experiment ${result.id} completed and was persisted.`)
      setTab('overview')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Experiment failed')
    } finally {
      setBusy(false)
    }
  }

  const runSweep = async () => {
    setBusy(true)
    setError('')
    try {
      const values = sweepValues.split(',').map((value) => Number(value.trim())).filter(Number.isFinite)
      const result = await api.sweep({
        dataset_id: Number(form.dataset_id),
        model_type: form.model_type,
        backend: form.backend,
        n_features: Number(form.n_features),
        circuit_depth: Number(form.circuit_depth),
        cv_folds: Number(form.cv_folds),
        seeds: form.seeds.split(',').map((value) => Number(value.trim())).filter(Number.isFinite),
        optimizer: form.optimizer,
        max_iterations: Number(form.max_iterations),
        shots: Number(form.shots),
        noise_multiplier: Number(form.noise_multiplier),
        test_size: Number(form.test_size),
        variable: sweepVariable,
        values,
      })
      setSweep(result)
      await refresh()
      setNotice(`Sweep completed with ${result.results.length} persisted experiments.`)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Sweep failed')
    } finally {
      setBusy(false)
    }
  }

  const togglePhase5Model = (modelType: string) => {
    setPhase5Models((current) => current.includes(modelType)
      ? current.filter((item) => item !== modelType)
      : [...current, modelType])
  }

  const runCrossDatasetComparison = async () => {
    if (phase5Models.length === 0) {
      setError('Select at least one model for the cross-dataset run.')
      return
    }
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const registeredIds = datasets.map((dataset) => dataset.id)
      const payload: Record<string, unknown> = {
        model_types: phase5Models,
        n_features: Number(form.n_features),
        cv_folds: Number(form.cv_folds),
        seeds: form.seeds.split(',').map((value) => Number(value.trim())).filter(Number.isFinite),
        circuit_depth: Number(form.circuit_depth),
        max_iterations: Number(form.max_iterations),
        shots: Number(form.shots),
        noise_multiplier: Number(form.noise_multiplier),
        test_size: Number(form.test_size),
        optimizer: form.optimizer,
        backend: form.backend,
      }
      if (registeredIds.length >= 2) payload.dataset_ids = registeredIds
      const result = await api.crossDatasetComparison(payload)
      setComparison(result)
      await refresh()
      setNotice(`Cross-dataset validation persisted ${result.rows.length} experiments.`)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Cross-dataset validation failed')
    } finally {
      setBusy(false)
    }
  }

  const upload = async () => {
    if (!file) {
      setError('Choose a CSV file first.')
      return
    }
    setBusy(true)
    try {
      const result = await api.upload(file, targetColumn)
      setDatasets((current) => [...current, result])
      setForm((current) => ({ ...current, dataset_id: String(result.id) }))
      setNotice(`${result.name} was validated and registered as dataset ${result.id}.`)
      setError('')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  const comparisonRows = useMemo(() => experiments.filter((item) => item.status === 'completed'), [experiments])
  const completedCount = experiments.filter((item) => item.status === 'completed').length

  return (
    <div className="min-h-screen bg-page text-body">
      <header className="border-b border-line/70 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-5 py-5 lg:px-8">
          <div className="flex items-center gap-3">
            <img className="brand-logo" src="/q-diagnose-logo.png" alt="Q-Diagnose logo" />
            <div>
              <p className="text-lg font-semibold tracking-wide text-white">Q-Diagnose</p>
              <p className="text-[11px] uppercase tracking-[0.22em] text-accent">Quantum decision support lab</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className={`badge ${health?.hardware.execution_available ? 'badge-good' : 'badge-warn'}`}>
              {health?.hardware.execution_available ? 'Hardware ready' : 'Simulator mode'}
            </span>
            <span className="hidden text-xs text-slate-500 sm:inline">{completedCount} completed runs</span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1500px] px-5 py-7 lg:px-8">
        <div className="mb-7 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow">Research workspace / phase 2–5</p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white lg:text-4xl">Reproducible disease-classification experiments</h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">Compare leakage-safe classical baselines with quantum kernels and variational classifiers. Every displayed result is persisted with its preprocessing, folds, runtime, and probability calibration.</p>
          </div>
          <div className="rounded-xl border border-warn/30 bg-warn/5 px-4 py-3 text-xs leading-5 text-warn/90">Research prototype only.<br />Not a clinically validated diagnostic device.</div>
        </div>

        {error && <div className="mb-5 rounded-xl border border-bad/40 bg-bad/10 px-4 py-3 text-sm text-bad">{error}</div>}
        {notice && <div className="mb-5 rounded-xl border border-good/40 bg-good/10 px-4 py-3 text-sm text-good">{notice}</div>}

        <nav className="mb-6 flex gap-1 border-b border-line/70">
          {(['overview', 'run', 'compare'] as const).map((item) => (
            <button className={`tab ${tab === item ? 'active' : ''}`} key={item} onClick={() => setTab(item)}>
              {item === 'run' ? 'Run experiment' : item === 'compare' ? 'Compare runs' : 'Overview'}
            </button>
          ))}
        </nav>

        {tab === 'run' && (
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
            <section className="panel">
              <div className="mb-5 flex items-center justify-between">
                <div><p className="eyebrow">Experiment builder</p><h2 className="mt-1 text-xl font-semibold text-white">Configure a run</h2></div>
                <span className="text-xs text-slate-500">Preprocessing is fit per fold</span>
              </div>
              <ExperimentForm form={form} setForm={setForm} datasets={datasets} busy={busy} onSubmit={() => void run()} />
              <div className="my-7 border-t border-line/70" />
              <div className="mb-4 flex items-center justify-between"><div><p className="eyebrow">Parameter sweep</p><h3 className="mt-1 text-lg font-semibold text-white">Compare one variable</h3></div><span className="text-xs text-slate-500">Runs are saved individually</span></div>
              <div className="grid gap-3 md:grid-cols-[1fr_2fr_auto] md:items-end">
                <Field label="Variable"><select value={sweepVariable} onChange={(event) => setSweepVariable(event.target.value)}><option value="n_features">PCA features</option><option value="circuit_depth">Circuit depth</option><option value="max_iterations">Iterations</option><option value="shots">Shots</option><option value="noise_multiplier">Noise multiplier</option></select></Field>
                <Field label="Values"><input value={sweepValues} onChange={(event) => setSweepValues(event.target.value)} placeholder="2, 3, 4" /></Field>
                <button className="secondary-button" disabled={busy || !form.dataset_id} onClick={() => void runSweep()}>Run sweep</button>
              </div>
              {sweep && <div className="mt-5 overflow-x-auto"><table className="data-table"><thead><tr><th>Value</th><th>Experiment</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>{sweep.results.map((row) => <tr key={row.sweep_value}><td>{row.sweep_value}</td><td>#{row.experiment_id}</td><td>{formatMetric(metricMean(row.metrics, 'accuracy'))}</td><td>{formatMetric(metricMean(row.metrics, 'f1'))}</td><td>{formatMetric(metricMean(row.metrics, 'roc_auc'))}</td></tr>)}</tbody></table>{sweep.errors.length > 0 && <div className="mt-3 rounded border border-bad/40 bg-bad/10 p-3"><p className="text-xs font-medium text-bad">{sweep.errors.length} sweep value(s) failed and are excluded from the table above</p><ul className="mt-2 space-y-1 text-[11px] text-slate-300">{sweep.errors.map((row) => <li key={row.sweep_value}>value {row.sweep_value}: {row.error}</li>)}</ul></div>}</div>}
            </section>
            <aside className="space-y-6">
              <section className="panel">
                <p className="eyebrow">Data intake</p><h2 className="mt-1 text-lg font-semibold text-white">Register a CSV</h2>
                <div className="mt-4 space-y-3"><input type="file" accept=".csv,text/csv" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /><Field label="Target column (optional)"><input value={targetColumn} onChange={(event) => setTargetColumn(event.target.value)} placeholder="target / label / diagnosis" /></Field><button className="secondary-button w-full" disabled={busy} onClick={() => void upload()}>Validate and register</button><div className="grid grid-cols-2 gap-2"><button className="secondary-button" disabled={busy} onClick={async () => { setBusy(true); try { const result = await api.createBuiltinDataset(); setDatasets((current) => current.some((dataset) => dataset.id === result.id) ? current : [...current, result]); setForm((current) => ({ ...current, dataset_id: String(result.id) })); setNotice('Breast cancer dataset is ready.') } catch (caught) { setError(caught instanceof Error ? caught.message : 'Dataset registration failed') } finally { setBusy(false) } }}>Add breast cancer</button><button className="secondary-button" disabled={busy} onClick={() => void registerDiabetes()}>Add diabetes split</button></div></div>
              </section>
              <section className="panel"><div className="flex items-center justify-between gap-3"><div><p className="eyebrow">Dataset catalog</p><h2 className="mt-1 text-lg font-semibold text-white">Second-dataset validation</h2></div><button className="secondary-button" disabled={busy} onClick={() => void registerDiabetes()}>Add diabetes split</button></div><div className="mt-3 space-y-2">{datasets.length === 0 && <p className="empty-state">Register the built-in dataset to begin.</p>}{datasets.map((dataset) => <button className="dataset-row" key={dataset.id} onClick={() => setForm((current) => ({ ...current, dataset_id: String(dataset.id) }))}><span className="h-2 w-2 rounded-full bg-accent" /><span className="min-w-0 flex-1 truncate text-left text-sm text-slate-300">{dataset.name}</span><span className="text-xs text-slate-500">{dataset.n_rows}</span></button>)}</div></section>
            </aside>
          </div>
        )}

        {tab === 'overview' && (
          <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
            <aside className="panel h-fit"><div className="mb-4 flex items-center justify-between"><div><p className="eyebrow">Run history</p><h2 className="mt-1 text-lg font-semibold text-white">Experiments</h2></div><button className="text-xs text-accent hover:text-white" onClick={() => void refresh()}>Refresh</button></div><ExperimentList experiments={experiments} selectedId={selected?.id ?? null} onSelect={(id) => void selectExperiment(id)} /></aside>
            <div className="min-w-0 space-y-6">
              {!selected ? <div className="panel flex min-h-[420px] items-center justify-center text-center"><div><p className="eyebrow">Awaiting selection</p><p className="mt-3 text-lg text-slate-300">Select a completed run or configure a new experiment.</p></div></div> : <>
                <section className="panel"><div className="flex flex-wrap items-start justify-between gap-4"><div><p className="eyebrow">Experiment #{selected.id} · {selected.status}</p><h2 className="mt-1 text-2xl font-semibold text-white">{labelFor(selected.model_type)}</h2><p className="mt-2 text-xs text-slate-500">Dataset {selected.dataset_id} · {backendLabel((selected.config as Record<string, unknown>).backend)} · {formatSeconds(selected.runtime_seconds)}</p>
</div><div className="flex gap-2"><button className="secondary-button" onClick={() => void loadInsights(selected.id)}>Load insights</button><button className="secondary-button" onClick={() => { setTab('run'); setForm((current) => ({ ...current, dataset_id: String(selected.dataset_id), model_type: replayModelType(selected.model_type), backend: replayBackend((selected.config as Record<string, unknown>).backend) })) }}>Repeat</button></div></div>{selected.superseded && <div className="mt-4 rounded border border-warn/40 bg-warn/10 p-3 text-[11px] text-slate-200"><p className="font-medium text-warn">Superseded run</p><p className="mt-1">This record was produced by pipeline revision <span className="font-mono">{selected.pipeline_revision ?? 'none recorded'}</span>; the current revision is <span className="font-mono">{selected.current_pipeline_revision}</span>. Its stored metrics are kept for history and are not comparable to current runs. Re-run the same configuration to obtain current numbers.</p></div>}<div className="mt-6"><MetricCards metrics={selected.metrics} config={selected.config} />
</div></section>
                <div className="grid gap-6 lg:grid-cols-2"><CurveChart title="ROC curve" x={selected.roc_curve?.fpr ?? []} y={selected.roc_curve?.tpr ?? []} /><CurveChart title="Precision–recall curve" x={selected.pr_curve?.recall ?? []} y={selected.pr_curve?.precision ?? []} /></div>
                <div className="grid gap-6 lg:grid-cols-2"><Confusion matrix={selected.confusion_matrix} /><CalibrationPanel experiment={selected} />
</div>
                {explanation && circuit && <div className="grid gap-6 lg:grid-cols-2"><ExplanationPanel explanation={explanation} /><CircuitPanel circuit={circuit} /></div>}
                {!explanation && <div className="panel flex items-center justify-between gap-4"><p className="text-sm text-slate-400">Load SHAP, quantum sensitivity, calibration, and circuit previews for this run.</p><button className="secondary-button" onClick={() => void loadInsights(selected.id)}>Load insights</button></div>}
              </>}
            </div>
          </div>
        )}

                {tab === 'compare' && (
          <div className="space-y-6">
            <section className="panel">
              <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
                <div>
                  <p className="eyebrow">Phase 5 / second dataset</p>
                  <h2 className="mt-1 text-xl font-semibold text-white">Cross-dataset validation</h2>
                  <p className="mt-2 max-w-3xl text-sm text-slate-500">Runs the selected models on every registered dataset with identical leakage-safe settings. The bundled diabetes dataset uses an explicit median split of the regression target and is not a clinical diagnosis.</p>
                </div>
                <button className="primary-button" disabled={busy} onClick={() => void runCrossDatasetComparison()}>
                  {busy ? 'Running comparison…' : 'Run cross-dataset validation'}
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(modelLabels).map(([value, label]) => (
                  <button
                    className={`badge cursor-pointer ${phase5Models.includes(value) ? 'badge-accent' : 'badge-muted'}`}
                    key={value}
                    onClick={() => togglePhase5Model(value)}
                  >
                    {phase5Models.includes(value) ? '✓ ' : ''}{label}
                  </button>
                ))}
              </div>
              {comparison && (
                <div className="mt-6 overflow-x-auto">
                  <table className="data-table">
                    <thead><tr><th>Dataset</th><th>Model</th><th>Backend</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th><th>Runtime</th><th>Run</th></tr></thead>
                    <tbody>
                      {comparison.rows.map((row) => (
                        <tr key={row.experiment_id}>
                          <td>{row.dataset_name}</td>
                          <td>{labelFor(row.model_type)}</td>
                          <td>{row.backend ?? 'n/a'}</td>
                          <td>{formatMetric(metricMean(row.metrics, 'accuracy'))}</td>
                          <td>{formatMetric(metricMean(row.metrics, 'f1'))}</td>
                          <td>{formatMetric(metricMean(row.metrics, 'roc_auc'))}</td>
                          <td>{formatSeconds(row.runtime_seconds)}</td>
                          <td>
                            <button className="text-xs text-accent hover:text-white" onClick={() => { void selectExperiment(row.experiment_id); setTab('overview') }}>
                              #{row.experiment_id}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="mt-3 text-[11px] text-slate-500">{comparison.disclaimer}</p>
                </div>
              )}
            </section>
            <section className="panel"><div className="mb-5"><p className="eyebrow">Cross-run analysis</p>
<h2 className="mt-1 text-xl font-semibold text-white">Completed experiment comparison</h2><p className="mt-2 text-sm text-slate-500">All values below are read from persisted experiment records; no estimates are inferred.</p></div><div className="overflow-x-auto"><table className="data-table"><thead><tr><th>Run</th><th>Dataset</th><th>Model</th><th>Backend</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th><th>Runtime</th></tr></thead><tbody>{comparisonRows.length === 0 && <tr><td colSpan={8} className="py-8 text-center text-slate-500">No completed runs available.</td></tr>}{comparisonRows.map((row) => <tr key={row.id} className="cursor-pointer hover:bg-white/5" onClick={() => { void selectExperiment(row.id); setTab('overview') }}><td>#{row.id}</td><td>{row.dataset_id}</td><td>{labelFor(row.model_type)}</td><td>{backendLabel((row.config as Record<string, unknown>).backend)}</td><td>{formatMetric(metricMean(row.metrics, 'accuracy'))}</td><td>{formatMetric(metricMean(row.metrics, 'f1'))}</td><td>{formatMetric(metricMean(row.metrics, 'roc_auc'))}</td><td>{formatSeconds(row.runtime_seconds)}</td></tr>)}</tbody></table></div></section><section className="panel">
<p className="eyebrow">Dataset readiness</p><div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">{datasets.map((dataset) => <div className="rounded-lg border border-line/70 bg-ink/40 p-4" key={dataset.id}><p className="font-medium text-slate-200">{dataset.name}</p><p className="mt-2 text-xs text-slate-500">{dataset.n_rows} rows · {dataset.n_columns - 1} features</p><p className="mt-1 text-xs text-slate-500">Target: {dataset.target_column}</p></div>)}</div></section></div>
        )}
      </main>
      <footer className="border-t border-line/70 px-5 py-6 text-center text-xs text-slate-600">Q-Diagnose research prototype · Metrics are experimental and require clinical validation before use.</footer>
    </div>
  )
}

export default App
