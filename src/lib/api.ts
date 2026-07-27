import type { Analysis, DatasetManifest, DatasetSummary, ExperimentRecord, ExperimentSummary, ProviderModel } from "../types";
const base = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    ...options,
    headers: options.body ? { "Content-Type": "application/json", ...options.headers } : options.headers,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.error?.message ?? payload?.detail ?? `Mirage API returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}
const post = <T>(path:string, body:unknown) => request<T>(path,{method:"POST",body:JSON.stringify(body)});

export const api = {
  health: () => request<{status:string;mode:string;schema_version:string}>("/api/v1/health"),
  system: () => request<Record<string,unknown>>("/api/v1/system"),
  models: () => request<{models:ProviderModel[]}>("/api/v1/models"),
  metrics: () => request<{metric_versions:Record<string,string>;metrics:Record<string,Record<string,unknown>>}>("/api/v1/metrics"),
  analyse: (question:string, reference_answer:string, sample_count:number, temperature:number) =>
    post<Analysis>("/api/v1/analyse",{question,reference_answer:reference_answer||null,sample_count,temperature,top_p:.9}),
  datasets: () => request<{datasets:DatasetSummary[]}>("/api/v1/datasets"),
  dataset: (name:string) => request<DatasetManifest>(`/api/v1/datasets/${name}`),
  validateDataset: (filename:string,content:string) => post<{
    valid:boolean;valid_count:number;invalid_count:number;errors:unknown[];preview:unknown[]
  }>("/api/v1/datasets/validate",{filename,content}),
  saveDataset: (filename:string,content:string) => post<{saved:boolean;name:string;version:string;size:number}>(
    "/api/v1/datasets",{filename,content}),
  experiments: () => request<{experiments:ExperimentSummary[]}>("/api/v1/experiments"),
  experiment: (id:string) => request<ExperimentRecord>(`/api/v1/experiments/${id}`),
  runExperiment: (body:unknown) => post<ExperimentRecord>("/api/v1/experiments",body),
  deleteExperiment: (id:string) => request<{deleted:boolean}>(`/api/v1/experiments/${id}`,{method:"DELETE"}),
  override: (experimentId:string,exampleId:string,human_label:boolean,note:string) =>
    post<ExperimentRecord>(`/api/v1/experiments/${experimentId}/examples/${exampleId}/override`,{human_label,note}),
  compare: (ids:string[]) => post<{experiments:Record<string,unknown>[]}>("/api/v1/experiments/compare",{experiment_ids:ids}),
  exportUrl: (id:string,format:string) => `${base}/api/v1/experiments/${id}/export?format=${format}`,
  findings: (group="research-v1") => request<Record<string,unknown>>(`/api/v1/findings/${group}`),
};
