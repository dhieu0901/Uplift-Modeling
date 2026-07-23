# Uplift Modeling for Campaign Optimization

This project uses causal machine learning to identify customers whose behavior is likely to change because of a campaign. It compares uplift and response targeting at the same contact budget.

## Project Status

The offline pipeline covers data preparation, modeling, policy evaluation, calibration, economics, and online-experiment design. Production impact remains unverified until a live randomized experiment is completed.

### Main result

On the 500,000-row Criteo sample, the Modified Outcome Method (MOM) performs best for `visit` at low budgets. Results are averaged across seeds `42`, `123`, and `2026`, each with 150,000 test users.

| Budget | MOM visits | Response visits | Gain | Improvement |
|---:|---:|---:|---:|---:|
| 5% | 715.40 | 430.13 | +285.27 | +66.32% |
| 10% | 907.78 | 700.76 | +207.03 | +29.54% |
| 20% | 1,073.50 | 1,026.33 | +47.16 | +4.60% |
| 30% | 1,241.09 | 1,222.94 | +18.14 | +1.48% |

Evidence is strongest at 5–10%. On the reference seed, the paired-bootstrap 95% intervals are `[180.03, 632.15]` at 5% and `[86.60, 589.11]` at 10%. Results at 20–30% are inconclusive.

The tested uplift policies do not outperform response targeting for the rarer `conversion` outcome. Conversion is therefore treated as a secondary KPI and guardrail.

## Problem Definition

A response model estimates who is likely to produce an outcome:

```text
P(Y = 1 | X)
```

That score can prioritize users who would act without the campaign. Uplift modeling instead estimates the change caused by treatment.

For each user, let `Y(1)` be the outcome under treatment and `Y(0)` the outcome under control. Only one is observed:

```text
Y = T * Y(1) + (1 - T) * Y(0)
```

Because an individual treatment effect cannot be observed directly, the model estimates the conditional average treatment effect:

```text
tau(x) = E[Y(1) - Y(0) | X = x]
```

The score separates four conceptual groups: persuadables, sure things, lost causes, and do-not-disturbs. These groups are useful for interpretation but are not observed labels.

Valid causal interpretation relies on:

- Random assignment or conditional unconfoundedness.
- Positivity: comparable users can appear in both treatment arms.
- Consistency between assigned treatment and observed outcome.
- No material interference between users.
- Pre-treatment features only.

### Policy objective

For a fixed budget, users are ranked by score and the top `k%` are targeted. Random, response, and uplift rankings are compared on the same test population.

If an incremental outcome is worth `V` and one contact costs `C`, the user-level expected net value is:

```text
net_value(x) = tau(x) * V - C
break_even_uplift = C / V
```

An absolute threshold is meaningful only after score calibration. Otherwise, a fixed top-k policy is safer.

## Data

### Criteo Uplift Prediction Dataset v2.1

| Attribute | Value |
|---|---:|
| Rows | 13,979,592 |
| Features | 12 (`f0`–`f11`) |
| Treatment rate | 85.00% |
| Visit rate | 4.70% |
| Conversion rate | 0.29% |

Two reservoir samples support development and rare-outcome analysis:

- `data/processed/criteo_sample_500k.parquet`: primary `visit` experiments.
- `data/processed/criteo_sample_2m.parquet`: `conversion` robustness experiments.

`exposure` is excluded because it is measured after treatment and would introduce leakage.

### Hillstrom Email Dataset

Hillstrom is a pipeline warm-up comparing `Mens E-Mail` with `No E-Mail` on `visit`. It validates the workflow but does not reproduce the paper's M/W models.

## Methods

| Method | Definition and role |
|---|---|
| Response model | Operational baseline trained on treated users and ranked by outcome probability. |
| S-learner | Fits one model `mu(x,t)` and predicts `mu(x,1) - mu(x,0)`. |
| T-learner | Fits separate treated and control outcome models, then subtracts their predictions. |
| X-learner | Imputes treatment effects and fits second-stage effect models for both arms. |
| CVT | Converts treatment and outcome into a classification target, with propensity correction for the 85/15 split. |
| MOM | Regresses `Y(T-e)/(e(1-e))` on features, where `e` is treatment propensity. |

LightGBM is the default nonlinear learner, with scikit-learn histogram gradient boosting as a fallback. MOM uses standardized features and Ridge regression.

## Evaluation

Experiments use joint treatment-outcome stratification and compare every policy on the same test population. Evaluation includes:

- Incremental outcomes at fixed budgets of 5%, 10%, 20%, and 30%.
- Cumulative uplift, Qini, and the Criteo separate relative AUUC.
- Three-seed stability analysis.
- Paired stratified bootstrap confidence intervals.
- Exact curve and learner validation against `scikit-uplift`.

For a selected group `S`, incremental outcomes are estimated from the randomized test data:

```text
gain(S) = |S| * (mean(Y | T=1, S) - mean(Y | T=0, S))
```

AUUC measures full-ranking quality. Incremental outcomes at the intended budget drive the policy decision.

## Model Decision

| Role | Choice | Reason |
|---|---|---|
| Primary visit policy | MOM, top 5% | Stable low-budget gain and fast fitting. |
| Visit challenger | S-learner | Competitive nonlinear ranking. |
| Conversion policy | Response model | Uplift models did not improve conversion. |
| Academic baseline | CVT | Retained for comparison. |

## Calibration and Economics

Calibration uses independent 60%/20%/20% train, calibration, and test partitions. Isotonic regression improves score magnitude while largely preserving ranking.

With illustrative values of `100` per incremental visit and `5` per contact, MOM at 5% has the highest estimated net value: `34,039.87`. Replace these inputs with actual unit economics before use.

## Online Validation Plan

The proposed randomized experiment compares:

| Arm | Policy | Target rate | Planned users |
|---|---|---:|---:|
| A | MOM ranking | 5% | 355,256 |
| B | Response ranking | 5% | 355,256 |
| H | No-campaign holdout | 0% | 147,273 |

The primary estimand is the intention-to-treat visit-rate difference between A and B. Checks cover allocation, missing outcomes, holdout contamination, uncertainty, and net value.

## Repository Structure

```text
.
├── data/                  # Raw and processed datasets
├── reports/               # Final reports, figures, tables, and weekly logs
├── scripts/               # Reproducible command-line workflows
├── src/
│   ├── data/              # Dataset loaders
│   ├── evaluation/        # Uplift metrics and bootstrap
│   ├── experiments/       # Calibration and online analysis
│   └── models/            # Baseline and uplift learners
├── tests/                 # Automated tests
├── README.md
└── requirements.txt
```

## Setup

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Reproduce the Pipeline

```powershell
# Prepare Criteo data and EDA
.\.venv\Scripts\python.exe scripts\prepare_criteo.py

# Validate the workflow on Hillstrom
.\.venv\Scripts\python.exe scripts\run_hillstrom.py

# Run the main Criteo experiment
.\.venv\Scripts\python.exe scripts\run_criteo.py

# Run stability and bootstrap analysis
.\.venv\Scripts\python.exe scripts\run_criteo_stability.py --policies transformed_outcome

# Validate against scikit-uplift
.\.venv\Scripts\python.exe scripts\validate_sklift.py

# Run calibration and business analyses
.\.venv\Scripts\python.exe scripts\analyze_uplift_calibration.py
.\.venv\Scripts\python.exe scripts\analyze_cost_benefit.py --outcome-value 100 --treatment-cost 5

# Design and dry-run the online experiment analyzer
.\.venv\Scripts\python.exe scripts\design_online_experiment.py
.\.venv\Scripts\python.exe scripts\simulate_online_experiment.py
.\.venv\Scripts\python.exe scripts\analyze_online_experiment.py `
  --input-path data/online_experiment_synthetic_results.csv `
  --report-path reports/generated/online_experiment_dry_run.md
```

Generated trial outputs are written to `reports/generated/`; reviewed results remain in the main `reports/` directory.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The suite covers data loading, uplift learners, ranking metrics, bootstrap, calibration, economic policies, and online-experiment analysis.

## Reports

See [reports/README.md](reports/README.md) for the report index. The main documents are:

- [Criteo EDA](reports/criteo_eda.md)
- [Hillstrom warm-up](reports/hillstrom_warmup.md)
- [Model evaluation](reports/model_evaluation.md)
- [scikit-uplift validation](reports/scikit_uplift_validation.md)
- [Uplift calibration](reports/uplift_calibration.md)
- [Cost-benefit analysis](reports/cost_benefit_policy.md)
- [Online-experiment design](reports/online_experiment_design.md)

## Limitations

- Results are offline estimates and require live randomized validation.
- The Criteo feature names are anonymized, limiting business interpretation.
- Calibration and monetary thresholds may change under population drift.
- The value and cost assumptions are illustrative.
- Strong ranking performance does not guarantee gains at every budget.

## References

1. Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). *Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning*. [PNAS](https://doi.org/10.1073/pnas.1804597116), [arXiv](https://arxiv.org/abs/1706.03461).
2. Gutierrez, P., & Gérardy, J.-Y. (2017). *Causal Inference and Uplift Modelling: A Review*. [PMLR](https://proceedings.mlr.press/v67/gutierrez17a.html).
3. Diemert, E., Betlei, A., Renaudin, C., & Amini, M. R. (2018). *A Large Scale Benchmark for Uplift Modeling*. [AdKDD](https://www.adkdd.org/papers/a-large-scale-benchmark-for-uplift-modeling/2018).
4. Diemert, E., Betlei, A., Renaudin, C., Amini, M. R., Gregoir, P., & Rahier, T. (2021). *A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling*. [arXiv](https://arxiv.org/abs/2111.10106), [code](https://github.com/criteo-research/large-scale-ITE-UM-benchmark).
5. Betlei, A., Diemert, E., & Amini, M. R. (2021). *Uplift Modeling with Generalization Guarantees*. [DOI](https://doi.org/10.1145/3447548.3467395).
6. Stochastic Solutions (2008). *Hillstrom Challenge: An Approach Using Uplift Modelling*. [PDF](https://www.stochasticsolutions.com/pdf/HillstromChallenge.pdf).
7. [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/).
8. [scikit-uplift documentation](https://www.uplift-modeling.com/).
