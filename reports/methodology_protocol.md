# Pre-Specified Honest Uplift Evaluation Protocol

## Research Question

At a fixed campaign budget, does an uplift ranking produce more incremental
outcomes than the operational response ranking on users not used for model or
policy selection?

The primary analysis uses `visit` and a 5% targeting budget. `Conversion` is a
secondary rare-outcome analysis and a business guardrail.

## Primary Estimand

For a deterministic targeting policy `pi(X)` and a no-campaign reference, the
incremental policy value is:

```text
V(pi) - V(0) = E[pi(X) * (Y(1) - Y(0))]
```

The reported count on an evaluation sample of size `n` is:

```text
incremental_outcomes = n * (V(pi) - V(0))
```

The primary contrast is the paired difference between the development-selected
uplift policy and response targeting at the same 5% budget.

## Data Partitioning

Every canonical experiment first uses a joint treatment/outcome-stratified
outer split:

| Partition | Share | Permitted use |
|---|---:|---|
| Development | 80% | Cross-validated candidate and policy selection. |
| Test | 20% | One final evaluation after all choices are fixed. |

The development partition uses three joint-stratified folds. Each candidate and
the AIPW nuisance models are trained outside a fold and predict that fold.
Scores are converted to within-fold percentiles before the out-of-fold
predictions are combined. The full development sample therefore selects the
model without in-sample policy evaluation.

The champion and response baseline are refit on all development data before the
locked test is opened. Test outcomes never influence candidate selection,
hyperparameters, outcome choice, or the primary budget. A 60%/20% explicit
train/validation split remains available for low-cost smoke tests only.

## Candidate Models

The standard benchmark includes:

- Response targeting trained on treated users.
- S-, T-, and X-learners.
- Class-variable transformation (CVT).
- Modified-outcome learner (implementation name: `transformed_outcome`).
- Cross-fitted R-learner.
- Cross-fitted doubly robust learner.

All candidates use the same pre-treatment feature set. `exposure` remains
excluded because it is measured after treatment.

## Model Selection

For every candidate, out-of-fold development policy value and its paired
contrast against response targeting are estimated at the pre-specified budgets.
The champion is
the uplift candidate with the largest lower 95% confidence bound for that
paired contrast at the primary 5% budget.

This conservative rule rewards both estimated value and precision. AUUC is
reported as a secondary full-ranking diagnostic and does not select the
deployment policy.

## Doubly Robust Policy Evaluation

Outcome nuisance models estimate:

```text
mu_1(X) = E[Y | X, T = 1]
mu_0(X) = E[Y | X, T = 0]
```

For randomized treatment propensity `e`, the AIPW treatment-effect score is:

```text
phi =
    mu_1(X) - mu_0(X)
    + T / e * (Y - mu_1(X))
    - (1 - T) / (1 - e) * (Y - mu_0(X))
```

Policy value uses the mean of `pi(X) * phi`. Policy contrasts use
`(pi_a(X) - pi_b(X)) * phi`, preserving the pairing between policies on the
same users. Influence-score standard errors produce the confidence intervals.

The intervals condition on the fitted policy. End-to-end repeated splits are
used separately to measure training and selection instability.

## Rare-Outcome Analysis

Conversion experiments address both treatment-arm imbalance and severe outcome
imbalance. Negative outcomes are undersampled separately within treated and
control arms using the same overall reduction factor. All positive outcomes
are retained, and validation selects the factor without accessing test
outcomes.

The candidate factors are declared before each run. Factor `1` is the
no-undersampling control. T-learner arm probabilities receive an exact
case-control prior correction using their realized negative keep rates. CVT
scores use the low-rate factor correction from the published method before
calibration or absolute interpretation.

## Semi-Synthetic Ground-Truth Benchmark

Real Criteo covariates are combined with nonlinear response surfaces containing
interactions and both positive and negative heterogeneous effects. Treatment is
randomized, and both `mu_0(X)` and `mu_1(X)` are known.

This benchmark reports:

- Precision in estimation of heterogeneous effect (PEHE).
- CATE MAE and bias.
- Pearson and Spearman association with true CATE.
- Exact policy value and regret against the oracle ranking.

Semi-synthetic results test CATE recovery; real-data policy value remains the
deployment-relevant analysis.

## Robustness

The full train/validation/test procedure is repeated over ten declared seeds.
These splits overlap and are correlated sensitivity analyses, not independent
experiments. The canonical seed remains the primary locked-test result.

## Decision Rule

An uplift policy is eligible for an online challenger test only when:

1. Its locked-test point estimate exceeds response targeting at 5%.
2. The paired confidence interval and repeated-split results do not reveal
   material instability that invalidates the expected gain.
3. Conversion and operational guardrails remain acceptable.
4. Estimated net value is positive under validated, non-illustrative business
   inputs.

Production impact requires a randomized online comparison. Offline results do
not authorize rollout by themselves.
