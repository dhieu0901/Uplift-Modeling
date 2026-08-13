# What the Exposure Column Would Give

## Why This Exists

`exposure` records whether the ad actually rendered. The models exclude it, and
this report is the reason in numbers rather than in argument: it reports both
what the column appears to show and what it can legitimately identify.

The question the column invites — *does someone who sees the ad and then acts
show the campaign working?* — is a real one. It is answerable, just not by
comparing the exposed group against the control arm.

## Protocol

- Data: `data/processed/criteo_sample_500k.parquet` (500,000 rows). This is a
  development sample on purpose. What follows describes how the data was
  recorded rather than how a model performs, so it has no business spending the
  confirmatory sample, which is opened once.
- Confidence level: `95.0%`.
- Assignment is randomized and **no** control user can be exposed, so
  assignment is a valid instrument for exposure under one-sided noncompliance
  and the Wald ratio identifies the effect among users whose ad renders.
- Standard errors for that ratio come from its influence function, because the
  numerator and denominator are computed on the same users.

## Results

### visit

| group                       | users  | visit_pct |
| --------------------------- | ------ | --------- |
| not assigned                | 74948  | 3.942734  |
| assigned, ad never rendered | 409728 | 3.517211  |
| assigned, ad rendered       | 15324  | 41.966849 |

Reading the last two rows as a comparison gives **+38.02 pp**. Users who were assigned but never saw the ad sit at `3.517%`, **below** the `3.943%` of users never assigned at all. Sending an ad cannot make someone less likely to act, so that ordering is not an effect: it shows the split on a post-assignment column has selected on the outcome.

The mechanism is that an ad renders only while someone is already browsing, so
`exposure = 1` marks users who were already on their way to the site.

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| Effect of being assigned, everyone | `+0.9607 pp` | `[+0.8070, +1.1144]` |
| Share of assigned users whose ad rendered | `3.6052%` | — |
| **Effect among users whose ad renders** | **`+26.6466 pp`** | `[+22.3368, +30.9565]` |

Splitting the apparent gap:

| Component | Value |
|---|---:|
| Apparent gap, rendered minus not assigned | `+38.02 pp` |
| What those users would have done untreated | `15.320%` |
| Effect the ad actually caused among them | `+26.65 pp` |
| Selection rather than campaign | `+11.38 pp` |

Those users were already `3.9x` more likely to act than the population before any ad was shown.

Only `3.61%` of assigned users see the ad, so the effect among them is diluted by roughly `28x` before it reaches the number a campaign can act on. Both estimates point the same way here; they differ in which group they describe, not in direction.

### conversion

| group                       | users  | conversion_pct |
| --------------------------- | ------ | -------------- |
| not assigned                | 74948  | 0.210813       |
| assigned, ad never rendered | 409728 | 0.112270       |
| assigned, ad rendered       | 15324  | 5.579483       |

Reading the last two rows as a comparison gives **+5.37 pp**. Users who were assigned but never saw the ad sit at `0.112%`, **below** the `0.211%` of users never assigned at all. Sending an ad cannot make someone less likely to act, so that ordering is not an effect: it shows the split on a post-assignment column has selected on the outcome.

The mechanism is that an ad renders only while someone is already browsing, so
`exposure = 1` marks users who were already on their way to the site.

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| Effect of being assigned, everyone | `+0.0986 pp` | `[+0.0617, +0.1354]` |
| Share of assigned users whose ad rendered | `3.6052%` | — |
| **Effect among users whose ad renders** | **`+2.7339 pp`** | `[+1.7137, +3.7540]` |

Splitting the apparent gap:

| Component | Value |
|---|---:|
| Apparent gap, rendered minus not assigned | `+5.37 pp` |
| What those users would have done untreated | `2.846%` |
| Effect the ad actually caused among them | `+2.73 pp` |
| Selection rather than campaign | `+2.63 pp` |

Those users were already `13.5x` more likely to act than the population before any ad was shown.

Only `3.61%` of assigned users see the ad, so the effect among them is diluted by roughly `28x` before it reaches the number a campaign can act on. Both estimates point the same way here; they differ in which group they describe, not in direction.

## What This Licenses

- The campaign chooses **whom to send to**, not whose ad renders. The
  decision-relevant number stays the effect of being assigned.
- Nothing here can be targeted on. At the moment a user is selected it is not
  known whether their ad will render, so `exposure` cannot enter a model that
  has to rank users before the campaign is sent.
- The exclusion restriction is assumed: assignment moves the outcome only
  through the ad rendering. For display advertising that is reasonable, since
  an ad that never rendered leaves nothing for the user to respond to. It is an
  assumption rather than something this data can verify.
- The effect among users whose ad renders describes a group that cannot be
  identified in advance, so it explains the mechanism rather than supporting a
  targeting decision.

## Reproducible Outputs

- Summary table: `outputs/tables/exposure_iv.csv`
