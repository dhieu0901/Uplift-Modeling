# Honest Uplift Modeling for Campaign Optimization

A reproducible project testing whether uplift ranking creates more incremental
outcomes than response targeting at a fixed campaign budget.

> **Decision:** test the S-learner as an online challenger for visits. Keep
> response targeting for conversion. The offline evidence does not support a
> production rollout or an ROI claim.

## Primary Result

The model-selection protocol used three-fold out-of-fold development
predictions and a disjoint one-million-row audit sample. At the pre-specified
5% visit budget, the locked 200,000-row test produced:

| Policy contrast | Incremental visits | 95% CI |
|---|---:|---:|
| S-learner minus response targeting | +168.5 | [-53.0, 390.0] |

The point estimate favors the S-learner, but the interval includes zero. Across
ten repeated honest splits, 9/10 point estimates were positive and only 1/10
confidence intervals was wholly positive. The result is promising, not
confirmatory.

For conversion, the selected undersampled T-learner was worse than response
targeting by 47.1 conversions at 5% and significantly worse at 10% through 30%.

Read the [project report](reports/project_report.md), the
[locked evaluation protocol](reports/evaluation_protocol.md), and the
[evidence index](reports/README.md).

## Project Design

- Joint treatment/outcome-stratified development and locked-test partitions.
- Three-fold out-of-fold model and policy selection.
- Paired augmented inverse-propensity weighted policy contrasts.
- A disjoint audit sample with zero row overlap with development samples.
- S-, T-, X-, CVT-, modified-outcome, R-, and doubly robust learners.
- Treatment-arm-specific undersampling and probability correction for rare
  conversion.
- A semi-synthetic benchmark with known CATE, PEHE, exact policy value, and
  oracle regret.

The primary estimand for targeting policy `pi(X)` is:

```text
V(pi) - V(0) = E[pi(X) * (Y(1) - Y(0))]
```

AUUC is a secondary ranking diagnostic. The operational decision uses the
paired policy-value contrast at the fixed 5% budget.

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
