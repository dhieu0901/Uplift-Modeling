# Randomized Online-Experiment Design

## Objective

Compare MOM and response targeting at the same 5% budget. The primary estimand is the intention-to-treat difference in visit rate across all users assigned to the two policy arms.

## Planning Assumptions

| Item | Value |
|---|---:|
| No-campaign visit rate | 0.038201 |
| Offline MOM arm rate | 0.042970 |
| Offline response arm rate | 0.041069 |
| Offline A–B difference | 0.001902 |
| Retained online effect | 75% |
| Planning difference | 0.001426 |
| Significance level | 0.05, two-sided |
| Power | 80% |
| Buffer | 15% |

## Proposed Allocation

| Arm | Policy | Target rate | Planned users |
|---|---|---:|---:|
| A | MOM ranking | 5% | 355,256 |
| B | Response ranking | 5% | 355,256 |
| H | No-campaign holdout | 0% | 147,273 |

Total planned cohort: **857,785 users**.

## Effect-Size Sensitivity

| Offline effect retained | Difference to detect | Users per policy arm before buffer | Users per arm after buffer |
|---:|---:|---:|---:|
| 100% | 0.001902 | 174,712 | 200,919 |
| 75% | 0.001426 | 308,918 | 355,256 |
| 50% | 0.000951 | 691,284 | 794,977 |

Smaller online effects require substantially more traffic. The 75% case is the default design; the 50% case is safer when traffic permits.

## Randomization

1. Lock eligibility, observation window, and campaign exclusions.
2. Randomize users into A, B, and H before applying either ranking policy.
3. Score A and B independently and target the top 5% within each arm.
4. Keep channel, creative, timing, frequency cap, and costs equal across A and B.
5. Log assignment, score, targeting decision, delivery, outcome, and campaign cost.

Do not compare only targeted users: each policy selects a different population, so that comparison would break randomization.

## Analysis Plan

- Primary: A–B visit-rate difference with a 95% confidence interval, analyzed by intention to treat.
- Secondary: A–H and B–H incremental visits, conversion, and full-arm net value.
- Guardrails: unsubscribe, complaints, contact frequency, and total cost.
- Data quality: sample-ratio mismatch, missing outcomes, duplicate users, and holdout contamination.
- Reporting: absolute effect, relative lift, confidence interval, and net value—not the p-value alone.

Sample size and the observation window must be locked before launch. Synthetic data may test the analyzer but cannot support a rollout decision.

## Analyzer Input

The aggregate input contains one row per randomized arm.

| Column | Meaning |
|---|---|
| `arm` | Arm identifier. |
| `assigned_n` | Users assigned to the arm. |
| `outcome_observed_n` | Users with an observed outcome. |
| `outcomes` | Positive outcomes. |
| `targeted_n` | Users selected by the policy. |
| `treatment_received_n` | Users who received treatment. |
| `total_campaign_cost` | Campaign cost, if available. |

Assignment defines the primary analysis; delivery counts are used for compliance checks.

## Decision Rule

Roll out MOM only if A beats B on the primary KPI, net value is positive, and guardrails remain acceptable. If A–B is inconclusive, retain the current policy and collect more data.
