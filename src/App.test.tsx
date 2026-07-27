import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";

beforeEach(() => {
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const payload = url.includes("/health")
      ? { status: "ok", mode: "local_research", schema_version: "2.0" }
      : url.includes("/system")
        ? { mode: "local_research", token_uncertainty: "unavailable_without_provider_logprobs" }
        : url.includes("/models")
          ? { models: [{provider:"cached_demo",model:"mirage/cached-research-samples",mode:"cached_demo",available:true,capabilities:{}}] }
        : url.includes("/findings")
          ? { available:false,reason:"No genuine live experiment records exist for this group.",experiments:[] }
        : url.includes("/datasets")
          ? { datasets: [] }
          : { experiments: [] };
    return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
  }) as typeof fetch;
});

test("positions Mirage as an evaluation platform", async () => {
  render(<App />);
  expect(screen.getByText("Make model uncertainty")).toBeInTheDocument();
  expect(screen.getByText(/not a truth oracle/i)).toBeInTheDocument();
  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
});

test("exposes unsupported token log-probabilities honestly", async () => {
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "Playground" }));
  expect(screen.getByText(/Token log-probabilities unavailable/i)).toBeInTheDocument();
});

test("shows the empty experiment state without fake metrics", async () => {
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "Experiments" }));
  await waitFor(() => expect(screen.getByText("No experiments saved yet.")).toBeInTheDocument());
});

test("requires at least one enabled uncertainty signal", async () => {
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "Experiments" }));
  for (const name of ["Semantic entropy", "Response consistency", "Self-verification"]) {
    fireEvent.click(screen.getByRole("checkbox", { name }));
  }
  expect(screen.getByRole("button", { name: "Run evaluation" })).toBeDisabled();
  expect(screen.getByText("Enable at least one available signal.")).toBeInTheDocument();
});

test("credits both builders with profile links", () => {
  render(<App />);
  expect(screen.getByRole("heading", { name: "Ninad Naik" })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "Aashita Jolly" })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Ninad Naik on LinkedIn" })).toHaveAttribute(
    "href", "https://www.linkedin.com/in/ninad-naik-274883262",
  );
  expect(screen.getByRole("link", { name: "Aashita Jolly on GitHub" })).toHaveAttribute(
    "href", "https://github.com/aashita-46",
  );
});

test("findings page refuses to invent absent research results", async () => {
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "Findings" }));
  await waitFor(() => expect(screen.getByText("No official findings yet.")).toBeInTheDocument());
  expect(screen.getByText(/No genuine live experiment records/i)).toBeInTheDocument();
});
