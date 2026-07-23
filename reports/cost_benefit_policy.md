# Targeting Policy Cost-Benefit Analysis

## Assumptions

- Incremental outcomes are averaged across three Criteo `visit` experiment seeds.
- One incremental visit is valued at `100` currency units.
- Targeting one user costs `5` currency units.
- The implied score break-even threshold is `5 / 100 = 0.05`.

These inputs illustrate the decision workflow and are not validated business assumptions.

## Policy Economics

| Policy | Budget | Incremental visits | Break-even cost per target | Net value at current inputs |
|---|---:|---:|---:|---:|
| MOM | 5% | 715.40 | 9.54 | 34,039.87 |
| MOM | 10% | 907.78 | 6.05 | 15,778.36 |
| MOM | 20% | 1,073.50 | 3.58 | -42,650.37 |
| MOM | 30% | 1,241.09 | 2.76 | -100,891.35 |
| Response | 5% | 430.13 | 5.74 | 5,513.15 |
| Response | 10% | 700.76 | 4.67 | -4,924.38 |
| Response | 20% | 1,026.33 | 3.42 | -47,366.87 |
| Response | 30% | 1,222.94 | 2.72 | -102,705.73 |

MOM at 5% maximizes the point estimate. The policy selector also includes `no_campaign` with zero value, preventing a negative-value campaign from being recommended.

## Gain over Response Targeting

| Budget | MOM net-value gain |
|---:|---:|
| 5% | 28,526.72 |
| 10% | 20,702.74 |
| 20% | 4,716.49 |
| 30% | 1,814.38 |

At equal budgets, campaign costs are identical; the difference comes from incremental visits.

## Decision

Use MOM top 5% as the candidate policy, subject to randomized validation. Before launch, replace the illustrative inputs with actual margin and campaign costs.
