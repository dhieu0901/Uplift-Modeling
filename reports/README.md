# Evidence Index

The [research paper](uplift_research_paper.md) is the canonical narrative.
This directory contains only the protocol, decision controls, and evidence
required to support or reproduce it.

## Research Package

| Artifact | Purpose |
|---|---|
| [Research paper](uplift_research_paper.md) | Research question, methods, results, decision, and limitations |
| [Methodology protocol](methodology_protocol.md) | Pre-specified estimand, split, selection, evaluation, and decision rules |
| [Claim ledger](claim_ledger.md) | Approved interpretation of each material claim |
| [Audit construction](audit_sample.md) | One-million-row sample and zero-overlap verification |
| [Visit audit](audit_visit_evaluation.md) | Locked S-learner versus response comparison |
| [Visit stability](visit_stability.md) | Ten repeated honest-split sensitivity runs |
| [Conversion development](rare_conversion_development.md) | Undersampling-family and factor selection |
| [Conversion audit](audit_conversion_evaluation.md) | Locked T-learner k=5 versus response comparison |
| [Conversion calibration](conversion_uplift_calibration.md) | Independent isotonic calibration analysis |
| [Known-CATE benchmark](semisynthetic_benchmark.md) | CATE error, exact policy value, and oracle regret |
| [Online experiment design](online_experiment_design.md) | Power, allocation, randomization, and analysis plan |

## Artifact Policy

- Markdown files contain the reviewed interpretation.
- CSV files in `tables/` contain the corresponding numerical evidence.
- PNG files in `figures/` are referenced by the paper or an evidence appendix.
- Ad hoc output belongs in the ignored `generated/` directory and is not
  decision evidence.
