export type TokenRisk = {
  id: number; text: string; logprob: number; entropy: number;
  normalisedUncertainty: number; importanceWeight: number; weightedRisk: number;
  category: "number" | "unit" | "entity" | "content" | "function" | "punctuation";
};
export type Sample = { id: string; answer: string; cluster: number; latency: number; agreement: number };
export type Cluster = { id: string; label: string; sampleIds: string[]; probability: number; representativeAnswer: string };
export type Analysis = {
  id: string; question: string; answer: string; tokens: TokenRisk[]; samples: Sample[]; clusters: Cluster[];
  semanticEntropy: number; normalisedSemanticEntropy: number; pTrue: number; verification: string;
  score: number; breakdown: { semantic: number; token: number; inversePTrue: number };
  mode: string; calibrationStatus: string; model: string; device: string; latency: number;
  metadata: Record<string, string | number>;
};
export type Bench = {
  id: string; dataset: string; count: number; incorrect: number; auroc: number | null;
  ece: number; brier: number; bins: { range: string; predicted: number; observed: number; count: number }[];
  records: { question: string; reference: string; correct: boolean; risk: number; method: string }[];
  timestamp: string;
};
