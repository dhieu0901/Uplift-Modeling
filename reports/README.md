# Evidence Index

The [project report](project_report.md) is the canonical narrative.
This directory contains only the protocol, decision controls, and evidence
required to support or reproduce it.

## Project Documentation

| Artifact | Purpose |
|---|---|
| [Project report](project_report.md) | Goal, implementation, results, decision, and limitations |
| [Evaluation protocol](evaluation_protocol.md) | Locked estimand, split, selection, evaluation, and decision rules |
| [Decision log](decision_log.md) | Evidence status and approved interpretation of each material result |
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
- PNG files in `figures/` are referenced by the project report or an evidence appendix.
- Ad hoc output belongs in the ignored `generated/` directory and is not
  decision evidence.
