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

## Decision Metric

For targeting policy `pi(X)`, the project estimates incremental value relative
to treating nobody:

```text
V(pi) - V(0) = E[pi(X) * (Y(1) - Y(0))]
```

AUUC remains a secondary ranking diagnostic. The primary decision compares the
uplift and response policies on the same users at the pre-specified 5% budget.

## Results and Decision

The visit workflow selected the S-learner using three-fold out-of-fold
development predictions. On the locked 200,000-row audit test:

| Policy contrast at 5% | Estimated difference | 95% CI | Decision |
|---|---:|---:|---|
| S-learner minus response targeting | +168.5 visits | [-53.0, 390.0] | Online challenger test |
| Undersampled T-learner minus response targeting | -47.1 conversions | [-104.6, 10.3] | Keep response targeting |

The visit estimate is directionally promising, but its interval includes zero.
Across ten repeated honest splits, 9/10 point estimates were positive and only
1/10 intervals were wholly positive. This is enough to justify a controlled
online challenger, not a production rollout or ROI claim.

For conversion, the uplift candidate was also significantly worse at budgets
from 10% through 30%. Response targeting therefore remains the current
conversion policy.

Read the [project report](reports/project_report.md), the
[locked evaluation protocol](reports/evaluation_protocol.md), and the
[evidence index](reports/README.md).

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
|-- reports/      # Project report, protocol, evidence, tables, and figures
|-- scripts/      # Reproducible command-line workflows
|-- src/          # Data, models, evaluation, and experiment code
`-- tests/        # Unit and integration tests
```

Raw and processed data are intentionally excluded from version control.

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
  --report-path reports\audit_visit_evaluation.md `
  --validation-path reports\tables\audit_visit_selection.csv `
  --test-path reports\tables\audit_visit_test.csv `
  --contrast-path reports\tables\audit_visit_contrasts.csv `
  --figure-path reports\figures\audit_visit_policy_value.png

.\.venv\Scripts\python.exe scripts\run_honest_stability.py
```

Conversion development, audit, and calibration:

```powershell
.\.venv\Scripts\python.exe scripts\run_honest_criteo.py `
  --sample-path data\processed\criteo_sample_2m.parquet `
  --outcome conversion --models response_model --selection-folds 3 `
  --undersampling-factors 1,5,10,25,50,100,200 `
  --report-path reports\rare_conversion_development.md `
  --validation-path reports\tables\rare_conversion_selection.csv `
  --test-path reports\tables\rare_conversion_internal_holdout.csv `
  --contrast-path reports\tables\rare_conversion_internal_contrasts.csv `
  --figure-path reports\figures\rare_conversion_development.png

.\.venv\Scripts\python.exe scripts\run_honest_criteo.py `
  --sample-path data\processed\criteo_audit_1m.parquet `
  --outcome conversion --models response_model --selection-folds 3 `
  --undersampling-factors 5 --undersampling-families t --random-state 777 `
  --report-path reports\audit_conversion_evaluation.md `
  --validation-path reports\tables\audit_conversion_selection.csv `
  --test-path reports\tables\audit_conversion_test.csv `
  --contrast-path reports\tables\audit_conversion_contrasts.csv `
  --figure-path reports\figures\audit_conversion_policy_value.png

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
