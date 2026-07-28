# Honest Uplift Modeling for Campaign Optimization

A reproducible decision project that tests whether a campaign should target
users by predicted response or by predicted incremental effect.

## The Original Problem

A campaign team has a limited contact budget and can target only 5% of eligible
users. The existing approach ranks users by their predicted probability of
visiting or converting after receiving the campaign:

```text
response_score(X) = P(Y = 1 | X, treatment = 1)
```

This score finds users who are likely to respond, but it does not establish
that the campaign caused the response. A high-ranked user may have visited
without any campaign. Spending budget on that user produces little incremental
value even when the response prediction is accurate.

The decision should instead prioritize users whose outcome probability changes
because of treatment:

```text
uplift(X) =
    P(Y = 1 | X, treatment = 1)
    - P(Y = 1 | X, treatment = 0)
```

The project therefore asks:

> At the same 5% campaign budget, does uplift targeting create more
> incremental outcomes than response targeting?

## Why This Is Difficult

The individual treatment effect is never directly observed: each user is
either treated or untreated, so the counterfactual outcome is missing.
Additional practical risks make a simple train/test benchmark unreliable:

- selecting a model and reporting its effect on the same holdout exaggerates
  performance;
- conversion is rare, making treatment-effect estimates unstable;
- a model can rank well globally but perform poorly within the operational top
  5%;
- subtracting two separately estimated policy values wastes the pairing
  information from users on whom the policies disagree;
- offline evidence from historical experiments does not prove future
  production impact.

## What the Project Builds

The repository implements an end-to-end comparison between the operational
response ranking and multiple uplift learners. The workflow:

1. loads randomized Criteo campaign data and removes the post-treatment
   `exposure` variable;
2. separates development data from a disjoint one-million-row audit sample;
3. produces out-of-fold predictions for model and policy selection;
4. evaluates fixed-budget policies with paired doubly robust contrasts;
5. stress-tests the result across repeated honest splits;
6. handles rare conversion with treatment-arm-specific undersampling and
   probability correction;
7. checks CATE recovery on a semi-synthetic benchmark with known ground truth;
8. converts the offline result into a powered online challenger-test design.

The candidate set contains response targeting plus S-, T-, X-, CVT-,
modified-outcome, R-, and doubly robust learners. These models are comparison
tools inside the project; the objective is a defensible campaign decision, not
to reproduce or extend a research paper.

## Evaluation Design

The source dataset contains 13,979,592 randomized observations. Three
development datasets serve different purposes:

| Dataset | Rows | Role |
|---|---:|---|
| Visit development sample | 500,000 | Repeated-split stability |
| Conversion development sample | 2,000,000 | Undersampling and calibration |
| Disjoint audit sample | 1,000,000 | Final model selection and locked evaluation |

The audit sample has an 84.59% treatment rate, 4.89% visit rate, and 0.31%
conversion rate. Complete-row hashes confirm zero overlap with both development
samples.

Each canonical experiment reserves 80% for development and 20% as a locked
test. Three joint treatment/outcome-stratified development folds create
out-of-fold candidate scores and nuisance predictions. The candidate with the
largest lower endpoint of the paired 95% confidence interval against response
targeting at the 5% budget is selected, refitted on all development data, and
evaluated once on the locked test.

For targeting policy `pi(X)`, incremental value relative to treating nobody is:

```text
V(pi) - V(0) = E[pi(X) * (Y(1) - Y(0))]
```

The doubly robust treatment-effect score is:

```text
phi =
    mu_1(X) - mu_0(X)
    + T / e * (Y - mu_1(X))
    - (1 - T) / (1 - e) * (Y - mu_0(X))
```

Policy value is the mean of `pi(X) * phi`. The primary paired contrast uses
`[pi_uplift(X) - pi_response(X)] * phi`, so uncertainty is driven by users on
whom the two policies disagree. AUUC remains a secondary ranking diagnostic.

## Reproduced Results

All results below are the checked-in snapshot from the commands in
[Reproduction](#reproduction). Generated tables, reports, and figures are
written to the git-ignored `outputs/` directory.

### Visit model selection

On 800,000 audit-development users, the 5% out-of-fold policy comparison
selected the S-learner. These values perform selection only; they are not the
final effect estimates.

| Candidate | Difference vs response | 95% CI |
|---|---:|---:|
| S-learner | +1,001.1 visits | [565.3, 1,436.9] |
| T-learner | +613.9 | [160.3, 1,067.5] |
| Modified outcome | +545.7 | [136.3, 955.1] |
| DR-learner | +465.1 | [27.5, 902.6] |
| X-learner | +346.1 | [-90.3, 782.4] |
| R-learner | +254.9 | [-182.0, 691.8] |
| CVT | -1,332.1 | [-1,764.2, -900.0] |

### Visit locked test

The selected S-learner and response baseline were refitted, then compared on
the untouched 200,000-row audit test:

| Budget | S-learner value | Response value | Paired difference | 95% CI |
|---:|---:|---:|---:|---:|
| 5% | 949.8 | 781.3 | +168.5 visits | [-53.0, 390.0] |
| 10% | 1,355.4 | 1,289.1 | +66.3 | [-114.8, 247.4] |
| 20% | 1,647.3 | 1,617.5 | +29.8 | [-76.6, 136.2] |
| 30% | 1,689.8 | 1,689.7 | +0.1 | [-83.2, 83.4] |

The pre-specified 5% point estimate favors the S-learner, but its interval
includes zero. Full-ranking relative AUUC is 0.010943 for the S-learner and
0.011249 for response targeting, illustrating why the local budget decision
need not match the global ranking metric.

### Visit stability

Ten complete train/selection/test repetitions on the 500,000-row sample
retrained the already selected S-learner:

| Runs | Mean difference per 100,000 users | Standard deviation | Range | Positive estimates | Wholly positive CIs |
|---:|---:|---:|---:|---:|---:|
| 10 | +93.6 visits | 62.7 | -37.7 to +183.1 | 9/10 | 1/10 |

The direction is reasonably stable, but interval evidence remains weak. The
splits overlap and are sensitivity analyses rather than independent
replications.

### Rare conversion

Treatment-stratified negative undersampling tested factors 1, 5, 10, 25, 50,
100, and 200 for T- and CVT-based learners. The conservative development rule
selected the T-learner with factor 5:

| Evaluation stage at 5% | T-learner k=5 minus response | 95% CI |
|---|---:|---:|
| Out-of-fold development | -35.7 conversions | [-131.8, 60.3] |
| Internal locked holdout | -2.5 | [-55.3, 50.3] |
| Disjoint audit test | -47.1 | [-104.6, 10.3] |

On the disjoint audit, the budget sensitivity was:

| Budget | T-learner k=5 minus response | 95% CI |
|---:|---:|---:|
| 5% | -47.1 conversions | [-104.6, 10.3] |
| 10% | -57.7 | [-106.1, -9.4] |
| 20% | -58.2 | [-100.6, -15.9] |
| 30% | -65.2 | [-106.8, -23.7] |

Response targeting remains the conversion policy.

### Conversion calibration

An independent calibration holdout improved score magnitude without materially
changing ranking:

| Score | EUCE | MUCE | Calibration intercept | Calibration slope | Relative AUUC |
|---|---:|---:|---:|---:|---:|
| Raw | 0.000629 | 0.003435 | -0.000380 | 1.576 | 0.001067 |
| Isotonic calibrated | 0.000298 | 0.001210 | -0.000048 | 1.024 | 0.001070 |

Calibration therefore improves absolute interpretation, not demonstrated
policy value.

### Semi-synthetic ground-truth check

The semi-synthetic benchmark uses real Criteo covariates with known nonlinear
potential outcomes and CATE. Exact policy value at the 5% budget was:

| Policy | PEHE | Spearman with true CATE | True incremental outcomes | Oracle value |
|---|---:|---:|---:|---:|
| Oracle | 0.0000 | 1.000 | 61.61 | 100.0% |
| Modified outcome | 0.0043 | 0.850 | 54.08 | 87.8% |
| S-learner | 0.0102 | 0.605 | 45.97 | 74.6% |
| X-learner | 0.0139 | 0.528 | 45.00 | 73.0% |
| T-learner | 0.0222 | 0.401 | 43.79 | 71.1% |
| CVT | 0.0461 | 0.393 | 42.57 | 69.1% |
| R-learner | 0.0218 | 0.455 | 41.08 | 66.7% |
| DR-learner | 0.0211 | 0.463 | 40.83 | 66.3% |
| Response | 0.0536 | 0.304 | 39.83 | 64.6% |

Observed out-of-fold AIPW selection chose CVT because it had the least-negative
lower confidence bound, while exact ground truth favored modified outcome.
This stress test shows that finite-sample policy selection can mis-rank models
even when the evaluation estimator is well motivated.

### Online challenger design

The proposed experiment retains 75% of the offline S-versus-response effect,
uses 80% power, two-sided 5% significance, and a 15% operational buffer:

| Arm | Policy | Target rate | Users | Expected offline visit rate |
|---|---|---:|---:|---:|
| A | S-learner | 5% | 1,837,713 | 0.042950 |
| B | Response targeting | 5% | 1,837,713 | 0.042108 |
| H | No-campaign holdout | 0% | 80,106 | 0.038201 |
| **Total** |  |  | **3,755,532** |  |

Users are randomized to complete policy arms before ranking. The primary
analysis is the intention-to-treat A-minus-B visit-rate difference across all
assigned users, not a comparison of the two targeted subsets.

## Final Decision

- **Visit:** advance the locked S-learner to a randomized online challenger;
  do not deploy from offline evidence alone.
- **Conversion:** keep response targeting; the selected uplift candidate is
  inferior on the audit.
- **Economics:** make no ROI claim until real outcome value and campaign cost
  are available.
- **Historical result:** the earlier 66.3% visit-improvement claim reused test
  evidence for selection and reporting and remains exploratory, not the current
  headline.

## Selected References

The project keeps three references that directly anchor its data, core model
family, and rare-outcome intervention:

1. [Diemert et al., *A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling*](https://arxiv.org/abs/2111.10106) — source of the randomized Criteo dataset and benchmark setting.
2. [Künzel et al., *Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning*](https://arxiv.org/abs/1706.03461) — basis for the S-, T-, and X-learner family used in the comparison.
3. [Nyberg, Kuśmierczyk, and Klami, *Uplift Modeling with High Class Imbalance*](https://proceedings.mlr.press/v157/nyberg21a.html) — basis for treatment-stratified negative undersampling and rare-outcome calibration.

These are implementation anchors, not an exhaustive literature review. The
audit construction, 5% decision rule, paired evaluation, semi-synthetic design,
online-test plan, and conclusions are project-specific.

## Repository

```text
.
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- data/                       # Local raw/processed data; git-ignored
|-- scripts/
|   |-- prepare_criteo.py
|   |-- prepare_audit_sample.py
|   |-- run_honest_criteo.py
|   |-- run_honest_stability.py
|   |-- analyze_uplift_calibration.py
|   |-- run_semisynthetic_benchmark.py
|   `-- design_online_experiment.py
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- criteo.py
|   |   |-- imbalance.py
|   |   `-- semisynthetic.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- base.py
|   |   |-- registry.py
|   |   |-- response_model.py
|   |   |-- s_learner.py
|   |   |-- t_learner.py
|   |   |-- x_learner.py
|   |   |-- cvt_learner.py
|   |   |-- modified_outcome.py
|   |   |-- r_learner.py
|   |   |-- dr_learner.py
|   |   |-- cross_fitting.py
|   |   |-- undersampled.py
|   |   `-- uplift_calibration.py
|   |-- evaluation/
|   |   |-- __init__.py
|   |   |-- uplift.py
|   |   |-- policy_value.py
|   |   |-- calibration.py
|   |   |-- ground_truth.py
|   |   `-- experiment_design.py
|   |-- experiments/
|   |   |-- __init__.py
|   |   |-- splitting.py
|   |   `-- honest_uplift.py
|   `-- reporting.py
|-- tests/
|   |-- test_criteo_loader.py
|   |-- test_direct_uplift_models.py
|   |-- test_experiment_design.py
|   |-- test_honest_experiment.py
|   |-- test_honest_splitting.py
|   |-- test_imbalance.py
|   |-- test_model_registry.py
|   |-- test_policy_value.py
|   |-- test_semisynthetic.py
|   |-- test_uplift_calibration.py
|   `-- test_uplift_evaluation.py
`-- outputs/                    # Generated by scripts; git-ignored
```

Raw data, processed data, and generated outputs are intentionally excluded from
version control.

## Reproduction

Python 3.11 or 3.12 is recommended.

### 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Prepare data

Place the Criteo Uplift Prediction Dataset v2.1 at
`data/criteo-uplift-v2.1.csv.gz`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_criteo.py `
  --sample-size 500000 `
  --sample-path data\processed\criteo_sample_500k.parquet `
  --random-state 42

.\.venv\Scripts\python.exe scripts\prepare_criteo.py `
  --sample-size 2000000 `
  --sample-path data\processed\criteo_sample_2m.parquet `
  --random-state 42

.\.venv\Scripts\python.exe scripts\prepare_audit_sample.py
```

### 3. Rebuild the evidence

Visit audit:

```powershell
.\.venv\Scripts\python.exe scripts\run_honest_criteo.py `
  --sample-path data\processed\criteo_audit_1m.parquet `
  --outcome visit --selection-folds 3 --random-state 777 `
  --report-path outputs\audit_visit_evaluation.md `
  --validation-path outputs\tables\audit_visit_selection.csv `
  --test-path outputs\tables\audit_visit_test.csv `
  --contrast-path outputs\tables\audit_visit_contrasts.csv `
  --figure-path outputs\figures\audit_visit_policy_value.png

.\.venv\Scripts\python.exe scripts\run_honest_stability.py
```

Conversion development, audit, and calibration:

```powershell
.\.venv\Scripts\python.exe scripts\run_honest_criteo.py `
  --sample-path data\processed\criteo_sample_2m.parquet `
  --outcome conversion --models response_model --selection-folds 3 `
  --undersampling-factors 1,5,10,25,50,100,200 `
  --report-path outputs\rare_conversion_development.md `
  --validation-path outputs\tables\rare_conversion_selection.csv `
  --test-path outputs\tables\rare_conversion_internal_holdout.csv `
  --contrast-path outputs\tables\rare_conversion_internal_contrasts.csv `
  --figure-path outputs\figures\rare_conversion_development.png

.\.venv\Scripts\python.exe scripts\run_honest_criteo.py `
  --sample-path data\processed\criteo_audit_1m.parquet `
  --outcome conversion --models response_model --selection-folds 3 `
  --undersampling-factors 5 --undersampling-families t --random-state 777 `
  --report-path outputs\audit_conversion_evaluation.md `
  --validation-path outputs\tables\audit_conversion_selection.csv `
  --test-path outputs\tables\audit_conversion_test.csv `
  --contrast-path outputs\tables\audit_conversion_contrasts.csv `
  --figure-path outputs\figures\audit_conversion_policy_value.png

.\.venv\Scripts\python.exe scripts\analyze_uplift_calibration.py
```

Ground-truth benchmark and online-test design:

```powershell
.\.venv\Scripts\python.exe scripts\run_semisynthetic_benchmark.py
.\.venv\Scripts\python.exe scripts\design_online_experiment.py
```

## Quality Gates

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\ruff.exe check src scripts tests
```

## Interpretation Boundaries

- The audit sample is disjoint but comes from the same source population.
- Confidence intervals condition on fitted policies.
- Repeated splits overlap and are not independent replications.
- Semi-synthetic findings depend on the chosen response surface.
- Production impact and economics require a live randomized experiment.
