export type MetricSummary = {
  mean: number
  std: number
  values: number[]
}

export type Metrics = Record<string, MetricSummary | number | number[][]>

export type Dataset = {
  id: number
  name: string
  source: string
  target_column: string
  n_rows: number
  n_columns: number
  class_distribution: Record<string, number>
  summary: Record<string, unknown> | null
  created_at: string
}

export type Curve = {
  fpr: Array<number | null>
  tpr: Array<number | null>
  thresholds: Array<number | null>
}

export type PrecisionRecallCurve = {
  precision: Array<number | null>
  recall: Array<number | null>
  thresholds: Array<number | null>
}

export type Experiment = {
  id: number
  dataset_id: number
  model_type: string
  status: string
  progress: number
  error_message: string | null
  config: Record<string, unknown>
  metrics: Metrics
  confusion_matrix: number[][] | null
  roc_curve: Curve
  pr_curve: PrecisionRecallCurve
  holdout_metrics: { metrics: Record<string, number> } | null
  runtime_seconds: number | null
  created_at: string
  feature_names: string[] | null
  test_data: Record<string, unknown> | null
  calibration: Calibration | null
  sweep_variable: string | null
  sweep_value: number | null
  pipeline_revision: string | null
  current_pipeline_revision: string
  superseded: boolean
}

export type Calibration = {
  brier_score: number
  bin_count: number
  bins: Array<{
    bin: number
    lower: number
    upper: number
    count: number
    predicted_probability: number | null
    observed_frequency: number | null
  }>
}

export type Explanation = {
  experiment_id: number
  model_type: string
  sample_count: number
  feature_names: string[]
  pipeline_revision: string | null
  current_pipeline_revision: string
  superseded: boolean
  method?: string
  base_values?: number | null
  top_features?: Array<{
    rank: number
    feature: string
    mean_absolute_contribution: number
  }>
  quantum?: {
    method: string
    epsilon: number
    base_probability: number
    feature_scores: Array<{
      rank: number
      feature: string
      mean_absolute_probability_change: number
      base_probability: number
    }>
  }
}

export type Circuit = {
  model_type: string
  quantum: boolean
  qubits?: number
  feature_map?: string
  ansatz?: string
  combined?: string
  message?: string
}

export type Health = {
  status: string
  datasets: number
  experiments: number
  hardware: {
    configured: boolean
    message: string
    token_present: boolean
    runtime_package_present: boolean
    execution_available: boolean
  }
}

export type SweepResponse = {
  variable: string
  values: number[]
  results: Array<{
    sweep_value: number
    experiment_id: number
    status: string
    metrics: Metrics
  }>
  errors: Array<{ sweep_value: number; error: string }>
}

export type ComparisonRow = {
  experiment_id: number
  dataset_id: number
  dataset_name: string
  model_type: string
  backend: string | null
  metrics: Metrics
  runtime_seconds: number | null
}

export type ComparisonResponse = {
  datasets: Array<{ id: number; name: string; source: string }>
  rows: ComparisonRow[]
  disclaimer: string
}
