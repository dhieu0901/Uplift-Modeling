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

## Implementation-to-Source Map

Each source below was checked against the implementation. The scope column
states exactly what is reused; it does not imply that this project reproduces
the source's experiments or results.

| Project component | Source | Verified use |
|---|---|---|
| Criteo data and benchmark metric | [Diemert et al., *A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling*](https://arxiv.org/abs/2111.10106) | Dataset schema, randomized benchmark setting, and the separate relative AUUC convention used in `src/data/criteo.py` and `src/evaluation/uplift.py`. |
| S-, T-, and X-learners | [Künzel et al., *Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning*](https://arxiv.org/abs/1706.03461) | Direct match for the one-model S-learner, arm-specific T-learner, and the X-learner's imputation, second-stage effect models, and propensity-weighted combination. |
| Class-variable transformation | [Jaśkowski and Jaroszewicz, *Uplift Modeling for Clinical Trial Data*](https://people.cs.pitt.edu/~milos/icml_clinicaldata_2012/Papers/Oral_Jaroszewitz_ICML_Clinical_2012.pdf) | Direct match for `Z = 1(Y = T)`, `uplift = 2P(Z=1\|X)-1`, and treatment-arm reweighting when assignment is unbalanced. |
| Modified outcome | [Athey and Imbens, *Recursive Partitioning for Heterogeneous Causal Effects*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4941430/) | Direct match for the transformed target `Y(T-e) / [e(1-e)]` used by `ModifiedOutcomeModel` and the calibration pseudo-outcome. |
| R-learner | [Nie and Wager, *Quasi-Oracle Estimation of Heterogeneous Treatment Effects*](https://arxiv.org/abs/1712.04912) | Direct match for cross-fitted nuisance estimates and R-loss minimization, implemented as weighted regression of `(Y-m(X))/(T-e)` with weights `(T-e)^2`. |
| DR-learner | [Kennedy, *Towards Optimal Doubly Robust Estimation of Heterogeneous Causal Effects*](https://arxiv.org/abs/2004.14497) | Direct match for the cross-fitted doubly robust pseudo-outcome and second-stage CATE regression. |
| Cross-fitting safeguard | [Chernozhukov et al., *Double/debiased Machine Learning for Treatment and Structural Parameters*](https://academic.oup.com/ectj/article/21/1/C1/5056401) | Source for the sample-splitting and out-of-fold nuisance-prediction pattern. The repository does not claim to implement the paper's complete DML estimator. |
| Fixed-policy evaluation | [Dudík, Langford, and Li, *Doubly Robust Policy Evaluation and Learning*](https://arxiv.org/abs/1103.4601) | The AIPW value in `src/evaluation/policy_value.py` is the binary-action, no-treatment-reference specialization of doubly robust policy value. |
| Rare-outcome handling | [Nyberg, Kuśmierczyk, and Klami, *Uplift Modeling with High Class Imbalance*](https://proceedings.mlr.press/v157/nyberg21a.html) | Direct match for treatment-stratified negative undersampling, the low-rate CVT factor correction, and transformed-outcome isotonic calibration. The T-learner's exact case-control probability inversion is an implementation-specific correction, not attributed to this paper. |

The disjoint audit construction, 5% selection rule, paired decision criterion,
semi-synthetic response surface, online-test design, and project conclusions
are project-specific choices rather than results taken from these sources.

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
