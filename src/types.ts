export type ProviderCapabilities = {
  supports_logprobs: boolean;
  supports_seed: boolean;
  supports_streaming: boolean;
  supports_vision: boolean;
  supports_retrieval: boolean;
  supports_token_usage: boolean;
};

export type DatasetExample = {
  id: string; question: string; reference_answer?: string; acceptable_answers: string[];
  unanswerable: boolean; domain: string; difficulty: string; source: string;
  tags: string[]; metadata: Record<string, unknown>;
};
export type DatasetSummary = {
  name: string; version: string; description: string; size: number; demonstration: boolean;
  license: string; domains: string[]; difficulties: string[];
};
export type DatasetManifest = {
  schema_version: string; name: string; version: string; description: string;
  license: string; demonstration: boolean; examples: DatasetExample[];
};
export type SemanticCluster = {
  cluster_id: string; response_indices: number[]; responses: string[]; size: number;
  representative_answer: string; probability: number; evaluator_method: string;
};
export type ExampleResult = {
  example_id: string; question: string; reference_answer?: string; acceptable_answers: string[];
  domain: string; difficulty: string; source: string; tags: string[];
  raw_generation: {
    response: string; sampled_responses: string[]; token_logprobs?: number[];
    token_entropies?: number[]; token_usage: Record<string, number>; latency_ms: number;
    estimated_cost?: number; provider_metadata: Record<string, unknown>; error?: string;
  };
  semantic_clusters: SemanticCluster[];
  verification?: { verdict: string; confidence: number; reason: string; source: string };
  signals: Record<string, number | string | null | undefined>;
  predicted_risk?: number;
  risk_contributions: Record<string, number>;
  correctness: {
    correct?: boolean; score?: number; exact_match?: number; token_f1?: number;
    method: string; reason: string; error_type: string; automated_label?: boolean;
    human_label?: boolean; human_override_at?: string; human_note?: string;
  };
  failure_types: string[]; trace: Record<string, unknown>; error_state?: string;
};
export type AggregateMetrics = {
  total_examples: number; labelled_examples: number; failed_examples: number;
  accuracy?: number; exact_match?: number; token_f1?: number; auroc?: number; auprc?: number;
  ece?: number; brier?: number; negative_log_likelihood?: number;
  mean_latency_ms?: number; p50_latency_ms?: number; p95_latency_ms?: number;
  average_input_tokens?: number; average_output_tokens?: number; total_estimated_cost?: number;
  reliability_bins: {low:number;high:number;predicted:number;observed:number;count:number}[];
  risk_coverage: {coverage:number;risk_threshold:number;selective_accuracy:number;error_rate:number;review_rate:number;remaining_errors:number}[];
  signal_comparison: {signal:string;coverage:number;auroc?:number;auprc?:number;ece?:number;brier?:number}[];
  warnings: string[];
};
export type ExperimentSummary = {
  experiment_id: string; experiment_name: string; creation_time: string; state: string;
  dataset: Record<string, unknown>; model: Record<string, unknown>; aggregates?: AggregateMetrics;
  schema_version: string;
};
export type ExperimentRecord = {
  schema_version: string; experiment_id: string; experiment_name: string; creation_time: string;
  completed_time?: string; state: string; dataset: Record<string, unknown>;
  model: {provider:string;model:string;version?:string;mode:string;capabilities:ProviderCapabilities};
  config: Record<string, unknown>; results: ExampleResult[]; aggregates?: AggregateMetrics;
};

export type Analysis = {
  id: string; question: string; matchedDatasetQuestion?: string; answer: string;
  tokens: never[]; tokenSignal?: {available:boolean;reason:string};
  samples: {id:string;answer:string;cluster:number;latency:number;agreement:number}[];
  clusters: {id:string;label:string;sampleIds:string[];probability:number;representativeAnswer:string;evaluatorMethod:string}[];
  semanticEntropy?: number; normalisedSemanticEntropy?: number; pTrue?: number;
  verification?: {verdict:string;confidence:number;reason:string};
  score?: number; breakdown: Record<string,number>; mode:string; calibrationStatus:string;
  model:string;device:string;latency:number;error?:string;capabilities:ProviderCapabilities;
  correctness?: Record<string,unknown>; metadata: Record<string,unknown>;
};
