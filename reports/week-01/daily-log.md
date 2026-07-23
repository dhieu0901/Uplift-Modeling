# Week 1 Work Log

| Field | Value |
|---|---|
| Period | July 20–26, 2026 |
| Author | Nguyen Duong Hieu |
| Project | Uplift Modeling for Campaign Optimization |
| Team size | 1 |

## Objectives

- [x] Define the causal targeting problem and evaluation criteria.
- [x] Set up the repository, environment, and data pipeline.
- [x] Review the main uplift-modeling methods and references.
- [x] Implement baseline models and offline evaluation.
- [ ] Complete stability, calibration, and business analyses.

## Daily Log

Record only work that was actually completed. Add links to commits, code, reports, or figures where useful.

| Date | Work completed | Result or evidence | Next step |
|---|---|---|---|
| Mon, Jul 20 | Defined the difference between response and uplift targeting; reviewed the Criteo, Hillstrom, meta-learner, and uplift-modeling references. | [Project overview](../../README.md) | Prepare the environment and inspect the datasets. |
| Tue, Jul 21 | Set up the Python environment and repository structure; implemented the Criteo and Hillstrom loaders; checked treatment and outcome distributions. | [Data loaders](../../src/data), [Criteo EDA](../criteo_eda.md) | Build the response baseline and initial uplift models. |
| Wed, Jul 22 | Implemented the response baseline, S-learner, and T-learner; added fixed-budget uplift evaluation and basic tests. | [Models](../../src/models), [Evaluation code](../../src/evaluation), [Tests](../../tests) | Add methods suitable for imbalanced treatment groups. |
| Thu, Jul 23 | Added X-learner, CVT, and MOM; compared policies on Hillstrom and the Criteo sample; documented preliminary model results. | [Hillstrom report](../hillstrom_warmup.md), [Model evaluation](../model_evaluation.md) | Run multi-seed stability and bootstrap analysis. |
| Fri, Jul 24 | Planned: evaluate the main policies across multiple seeds and add bootstrap confidence intervals. | — | Calibrate the shortlisted uplift scores. |
| Weekend | Planned: clean up documentation and review calibration, cost-benefit, and online-experiment requirements. | — | Finalize the Week 1 summary and Week 2 backlog. |

## Weekly Summary

- **Completed:** Project scope, literature review, data loading, EDA, six candidate models, offline ranking metrics, and initial tests.
- **Key result:** The end-to-end offline workflow runs on both datasets, and MOM is a promising low-budget policy for the `visit` outcome.
- **Evidence:** [Source code](../../src), [reports](../README.md), and [test suite](../../tests).
- **Open issue:** Multi-seed stability, score calibration, and production validation still require further work.
- **Week 2 plan:** Finalize robustness checks, cost-benefit assumptions, and the randomized online-experiment design.
