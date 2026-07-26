import type { Analysis, Bench } from "../types";
const base = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) throw new Error(`Mirage API returned ${response.status}`);
  return response.json() as Promise<T>;
}
export const api = {
  health: () => request<{status: string; mode: string}>("/api/v1/health"),
  system: () => request<Record<string, unknown>>("/api/v1/system"),
  analyse: (question: string, domain: string, sample_count: number, temperature: number) =>
    request<Analysis>("/api/v1/analyse", { question, domain, sample_count, temperature, top_p: .9 }),
  stress: (question: string, types: string[]) =>
    request<{original: Analysis; variants: StressVariant[]; stability: number; instability: number; summary: string}>(
      "/api/v1/stress", { question, types }),
  bench: (count: number) => request<Bench>("/api/v1/bench/runs", { count, seed: 42 }),
};
export type StressVariant = { id: number; type: string; question: string; answer: string; score: number; agreement: number; relation: string; adversarial: boolean };
