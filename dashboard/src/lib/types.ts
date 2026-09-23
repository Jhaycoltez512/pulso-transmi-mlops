export interface Station {
  station_id: string;
  station_name: string;
  corridor: string;
  latitude: number;
  longitude: number;
}

export interface Observation {
  station_id: string;
  observed_at: string;
  demand: number;
}

export interface Prediction {
  station_id: string;
  target_at: string;
  horizon_minutes: number;
  predicted_demand: number;
  actual_demand: number | null;
  evaluated_at: string | null;
}

export interface ForecastRun {
  id: string;
  cycle_id: string | null;
  data_cutoff: string;
  started_at: string;
  finished_at: string | null;
  status: "running" | "succeeded" | "failed";
  trigger_reason: string | null;
  error_message: string | null;
  git_commit: string | null;
}

export interface IngestionRun {
  id: string;
  started_at: string;
  finished_at: string | null;
  status: "running" | "succeeded" | "failed";
  observation_rows_read: number;
  error_message: string | null;
  data_version: string | null;
}

export interface ModelVersion {
  id: string;
  name: string;
  version: string;
  algorithm: string;
  is_active: boolean;
  trained_at: string | null;
  training_data_end: string | null;
  git_commit: string | null;
  data_version: string | null;
}

export interface ModelMetric {
  training_run_id: string;
  station_id: string | null;
  split_name: string;
  metric_name: string;
  metric_value: number;
}

export interface DriftMeasurement {
  id: string;
  feature_name: string;
  drift_type: "data" | "concept" | "performance";
  method: string;
  value: number;
  threshold: number | null;
  triggered: boolean;
  calculated_at: string;
}

export interface LeaderboardRow {
  display_name: string;
  kind: string;
  eligible: boolean;
  accuracy: number;
  raw_wape: number;
  accuracy_at_20: number;
  coverage: number;
  rank: number;
  calculated_at: string;
}

export interface LeaderboardResponse {
  me: { participant_id: string; display_name: string; kind: string } | null;
  cumulative: LeaderboardRow[];
  rolling_24h: LeaderboardRow[];
  error?: string;
}
