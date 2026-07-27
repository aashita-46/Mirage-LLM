# MirageEval: A Local-First Platform for LLM Reliability, Uncertainty and Selective Prediction

## Abstract

This report studies whether uncertainty and consistency signals predict labelled
errors and support selective human review. Results are inserted only from stored,
traceable Mirage experiments.

<!-- GENERATED_FROM_EXPERIMENT -->

Experiment group: `live-smoke-v1`. These results must not be interpreted beyond the recorded sample size.
Generated: 2026-07-27T05:16:27.197713+00:00
Git commit: `d8647b45057c5802972a73f4afa1eaade3c7fe1c-dirty`

| Experiment | Model | N | Accuracy | AUROC | AUPRC | ECE | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| `exp_2beb036aeb04eac1` | phi:latest | 3 | 0.000 | N/A | 1.000 | 0.467 | 0.360 |

Dataset: `mirage-reliability-set-v1` v1.0
Schema: `2.0`
Metric versions: `{"calibration": "2.0", "classification": "2.1", "risk_coverage": "2.0", "semantic_entropy": "2.1"}`

Reproduce:

```bash
python scripts/evaluate.py --config config/live-smoke-v1.json --resume
```

<!-- END_GENERATED_FROM_EXPERIMENT -->

## Motivation

LLM uncertainty is not factual truth. MirageEval evaluates whether selected signals
correlate with errors on a defined, provenance-aware dataset.

## Research questions

Which uncertainty and consistency signals best separate correct and incorrect factual
answers across genuinely available language models, and how much review can selective
prediction save?

## Dataset

Mirage Reliability Set v1 contains source-gold-labelled examples assembled from
pinned TruthfulQA and SQuAD 2.0 records. “Verified” means a gold annotation exists in
the cited source; Mirage has not independently reverified every fact.

## Models

<!-- GENERATED_MODEL_CONFIGURATIONS -->

- `ollama/phi:latest` — experiment `exp_2beb036aeb04eac1`, temperature 0.7, samples 2

<!-- END_GENERATED_MODEL_CONFIGURATIONS -->

## Experimental setup

See `config/research-experiment-v1.json`.

## Uncertainty signals

Semantic entropy, response disagreement, exact consistency, answer switching,
entity disagreement, numerical variance, self-verification uncertainty, and the
experimental Mirage Risk Score are evaluated when available.

## Semantic clustering methods

Lexical Jaccard remains a labelled fallback. The primary research configuration uses
local `all-minilm` cosine similarity with a saved threshold and similarity matrix.

## Correctness evaluation

Deterministic matching is the primary evaluator. Automated and human labels are
preserved separately. Optional judges do not overwrite deterministic labels.

## Metrics

AUROC, AUPRC, ECE, Brier score, NLL, risk coverage, selective accuracy, and bootstrap
confidence intervals are reported when mathematically valid.

## Statistical analysis

Paired bootstrap differences compare aligned signals. Post-hoc Platt and isotonic
calibration use disjoint stratified train and evaluation indices.

## Results

See the generated block above.

## Model-level results

Generated only when live model experiments exist.

## Domain-level results

Generated only from stored per-example data.

## Selective prediction

Coverage at 90% and 95% selective accuracy is reported when achieved.

## Failure analysis

Confident errors, uncertain-correct answers, premise failures, unanswerable responses,
entity disagreement, numerical instability, and self-verification failures are
retained for inspection.

## Limitations

Results are conditional on this dataset, prompts, models, providers, sampling
configuration, correctness evaluator, and clustering method.

## Threats to validity

- Dataset size and domain composition may not represent deployment traffic.
- Source gold annotations and reference-answer quality may contain errors.
- Model selection is constrained by locally available hardware.
- Sampling results depend on temperature, top-p, seed support, and provider versions.
- Embedding clustering can merge contradictions or split paraphrases.
- LLM judges and self-verification are fallible model outputs.
- Calibration can be unstable with limited or imbalanced errors.
- Bootstrap intervals inherit the observed sample and are not population guarantees.
- Results are prompt-dependent and providers can drift across versions.
- Uncertainty cannot independently establish factual truth.

## Reproducibility

Run the commands documented in the generated block and repository README.

## Conclusion

The generated block currently contains a three-example live-provider smoke run. It
validates the genuine Ollama, embedding, persistence, and reporting path, but it is
far too small and one-class to answer the primary research question. No comparative
research conclusion is claimed.

## References

See the repository methodology references and the pinned dataset source URLs.
