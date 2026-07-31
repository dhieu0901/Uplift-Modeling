# Randomized Experiment Design for the Targeting Policy

## Objective

Directly compare `s_learner` with `response_model` at the same
`5.0%` budget. The primary estimand is the
**intention-to-treat difference** in visit rate across all users assigned to
each policy arm.

## Sample-Size Assumptions

- Offline source: `outputs/tables/confirmatory_visit_test.csv` on a test population of 4,000,000 users.
- No-campaign visit rate: `0.038333`.
- Implied visit rates for arms A/B: `0.042906` / `0.041441`.
- Offline A-B difference: `0.001465`.
- Planning effect retains `75%` of the offline
  difference: `0.001099`.
- Two-sided test, alpha `0.05`, power `80%`.
- Buffer for attrition/logging loss: `15%`.

The unbuffered sample size for each policy arm is 522,949. The
holdout size is calculated conservatively from the response-policy versus
no-campaign comparison: 109,595.

## Proposed Allocation

| Arm | Policy              | Target rate | Users   | Offline visit rate |
| --- | ------------------- | ----------- | ------- | ------------------ |
| A   | s_learner           | 5%          | 601,392 | 0.042906           |
| B   | response_model      | 5%          | 601,392 | 0.041441           |
| H   | no_campaign_holdout | 0%          | 126,035 | 0.038333           |

Proposed total cohort: **1,328,819 users**. Each policy arm has
601,392 users and is expected to target approximately
30,070 / 30,070 users. The holdout contains
126,035 users who receive no campaign during the measurement window.

## Sensitivity Analysis by Online Effect Size

| Effect retained | Difference | Users per arm | Users per arm with buffer |
| --------------- | ---------- | ------------- | ------------------------- |
| 100%            | 0.001465   | 295,385       | 339,693                   |
| 75%             | 0.001099   | 522,949       | 601,392                   |
| 50%             | 0.000733   | 1,171,729     | 1,347,489                 |

The required sample size grows rapidly when the online effect is smaller than
the offline estimate. The default design assumes 75% effect retention; if traffic
allows, the 50% scenario is safer.

## Randomization Procedure

1. Finalize eligibility and the observation window; exclude users in
   conflicting campaigns.
2. Randomize deterministically by user ID into A, B, and H **before applying
   ranking policies**.
3. Score A and B independently, then target exactly the top
   5.0% within each arm.
4. Use the same channel, creative, send time, frequency cap, and treatment
   cost for A/B.
5. Keep assignment fixed; log assignment, score, treatment delivered, and
   outcome.

Do not compare only the two targeted subsets, because each policy selects a
different population and that comparison breaks randomization.

## Analysis Plan

- Primary: A-B visit-rate difference with a 95% confidence interval, analyzed
  by ITT.
- Secondary: incremental visits versus H, conversion rate, and net value for
  the full arm.
- Guardrails: unsubscribe/opt-out, complaints, contact frequency, and campaign
  cost.
- Report absolute difference, relative lift, and confidence interval, not only
  the p-value.
- Lock sample size and the measurement window before launch; do not stop early
  based on p-values.
- Check sample-ratio mismatch, contamination, and missing outcomes before
  interpreting lift.

## Decision Criteria

Roll out `s_learner` only when A beats B on the primary KPI, net value is
positive, guardrails do not deteriorate, and the result is not driven by a
small subgroup. If A-B is inconclusive but both beat H, retain the current
policy and collect more data instead of declaring the policies equivalent.

## Reproducible Output

- Allocation table: `outputs/tables/online_experiment_arms.csv`
