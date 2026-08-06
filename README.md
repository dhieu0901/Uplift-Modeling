# Uplift Modeling for a Budget-Constrained Campaign

A campaign can contact 5% of eligible users. Should it target the users most
likely to respond, or the users whose behaviour actually *changes* because of
the campaign?

On four million randomized users never touched by training or model selection,
uplift targeting delivers **+5,861 incremental visits at a 5% budget
(95% CI [4,851, 6,871]), 47% above the response-targeting incumbent**.

Two limits come with that number. The advantage is gone by a 20% budget. And
for the rarer conversion outcome the same protocol reaches no conclusion at all.

Every figure below is backed by a tracked file in [outputs/](outputs/).

## Decisions

| Question | Decision | Evidence |
|---|---|---|
| Visit targeting | Adopt uplift targeting at 5%, then confirm live | +5,861 visits [4,851, 6,871] on 4M untouched users, z = 11.4 |
| Which model | Treat the S-learner as one acceptable instance, not the proven best | It wins 6 of 10 repeated splits; three other learners win the rest |
| Budget | Keep it tight | +47% at 5%, +15% at 10%, no measurable advantage at 20% or 30% |
| Conversion | Keep response targeting | No interval excludes zero at any budget — no case for switching either way |
| Economics | No ROI claim | Outcome value and contact cost were never available |
| Next step | Run the randomized test in [online_experiment_design.md](outputs/online_experiment_design.md) | Offline evidence bounds what a live campaign might do; it does not replace it |

## The Problem

The incumbent ranks by `P(Y = 1 | X, T = 1)` — who is likely to respond. That
never establishes the campaign *caused* the response. Treating everyone lifts
the visit rate from 3.820% to 4.854%, so roughly three-quarters of the
"responses" a response model learns to predict would have happened anyway.

The quantity the campaign cares about is the difference:

```text
uplift(X) = P(Y = 1 | X, T = 1) - P(Y = 1 | X, T = 0)
```

which is never observed for anyone: each user appears in one arm only, so there
is no label to train on and none to validate against.

## Data

Criteo Uplift v2.1 — **13,979,592 rows**, treatment randomized at 85%, visit
4.70%, conversion 0.29%, average treatment effect **+1.0342 pp** on visits.
Randomization is the load-bearing property: without it, every causal estimate
would rest on an untestable "all confounders controlled" assumption.

**One column is excluded.** `exposure` records whether the ad actually rendered,
which is decided *after* assignment. Visit rate is 41.5% among exposed users
against 3.5% among treated-but-unexposed — an ad renders only while someone is
already browsing, so the column identifies people already on their way to visit.
Conditioning on it is collider bias. The proof is that treated-unexposed (3.5%)
sits *below* control (3.8%), which cannot happen under intact randomization.
Only 3.6% of treated users were exposed, so the estimand is intention-to-treat:
the effect of choosing to reach someone, which is the only lever a campaign has.

Four disjoint samples, made disjoint by row identity rather than by value:

| Sample | Rows | Measured effect | Role |
|---|---:|---:|---|
| Development | 500,000 | +0.9607 pp | Repeated-split stability |
| Conversion development | 2,000,000 | +1.1257 pp | Rare-outcome experiments |
| Audit | 1,000,000 | +1.0827 pp | Model selection, one internal locked test |
| **Confirmatory** | **4,000,000** | **+1.0052 pp** | **Primary evidence, opened once** |

All four sit within about 1 SE of the population, which the identity rule is
what secures. The source holds 2,221,150 rows whose values duplicate another
row, and they are almost entirely inert (+0.002 pp effect, against +1.910 pp
among unique rows). Excluding by value would drop every copy as soon as one was
drawn, so later samples would lose inert rows and read high. Each row carries a
`row_id`, and `count_overlapping_rows` verifies disjointness as a separate query
before a sample is opened.

## Method

**Evaluation.** Policy value is estimated with the AIPW score

```text
phi = mu_1(X) - mu_0(X)
      + T / e * (Y - mu_1(X))
      - (1 - T) / (1 - e) * (Y - mu_0(X))
```

and the value of a targeting policy relative to treating nobody is
`E[pi(X) * phi]`. AIPW is doubly robust — unbiased if either the outcome models
or the propensity is right. In a randomized design the propensity is *known*, so
that condition holds by construction, and the estimate stays unbiased however
badly the outcome models fit. That is why one LightGBM configuration is fixed
across all candidates and never tuned: evaluation does not need the nuisance
models to be good, only to be out of fold.

**Paired contrast.** The primary comparison is
`[pi_uplift(X) - pi_response(X)] * phi`, not the difference of two separately
estimated values. Users both policies select cancel, leaving uncertainty only
where the policies disagree — necessary to resolve a difference inside the top 5%.

**Protocol.** 80% development, 20% locked test, stratified on the joint
treatment×outcome strata. Three cross-fitted folds produce out-of-fold candidate
scores and out-of-fold nuisances. The candidate with the largest lower bound
against response targeting at the 5% budget is selected, refit, and evaluated
once. After selection only the champion and two reference policies are scored on
test, so no result exists to change one's mind with.

Budget (5%), confidence (95%), and the selection rule were fixed before looking
at any data. Two policies are always evaluated but can never win:
`response_model` (the incumbent) and `random_targeting` (a floor that audits the
estimator — an uninformative policy must collect roughly 5% of the achievable
effect on a 5% budget; it reads 4.2%).

**Candidates.** Eight learners on one shared LightGBM configuration:
S-, T-, X-learner (outcome modelling); CVT, transformed-outcome, R-, DR-learner
(label transformation). CVT needed inverse-propensity reweighting, because its
classical form assumes a balanced design and this one is 85/15.

## Results — Visit

Selection on 800,000 out-of-fold development rows at the 5% budget:

| Candidate | vs response | 95% CI |
|---|---:|---:|
| **S-learner** | **+928.0** | **[491.5, 1,364.4]** |
| X-learner | +322.9 | [-113.7, 759.4] |
| Transformed outcome | +291.6 | [-131.7, 714.9] |
| DR-learner | +267.6 | [-168.7, 704.0] |
| R-learner | +136.8 | [-294.0, 567.6] |
| T-learner | +38.9 | [-408.7, 486.4] |
| CVT | -482.2 | [-906.0, -58.4] |
| _random targeting_ | _-2,678.1_ | _[-3,193.3, -2,162.8]_ |

The S-learner is the only candidate whose interval clears zero. Complexity did
not pay: R- and DR-learner cost ~50 s each on five-fold cross-fitting and land
mid-table, while a ridge regression at 0.29 s nearly matches them.

Confirmatory test on 4,000,000 untouched rows, opened once:

| Budget | S-learner minus response | 95% CI | z | Relative |
|---:|---:|---:|---:|---:|
| **5%** | **+5,861** | **[4,851, 6,871]** | **11.37** | **+47.1%** |
| 10% | +2,950 | [2,076, 3,825] | 6.61 | +14.8% |
| 20% | -421 | [-1,019, 176] | -1.38 | -1.6% |
| 30% | -432 | [-908, 45] | -1.77 | -1.6% |

![Confirmatory policy value](outputs/figures/confirmatory_visit_policy_value.png)

The internal 200,000-row test pointed the same way (+380.5 [163.6, 597.5],
z = 3.44) with a standard error over four times larger per user. The larger
sample bought precision, not a different answer.

Note that response targeting has the *better* global AUUC (0.009130 against
0.009037) while losing decisively at 5%. A whole-ranking metric can disagree
with the operational decision, which is why the primary criterion is a
budget-specific contrast.

## Results — Conversion

At a 0.29% base rate, negative undersampling was swept over factors 1 to 200 for
T- and CVT-based logistic learners, each with case-control prior correction.
Selection chose **factor 1** — no undersampling at all, which is itself the
answer to whether undersampling helps here.

On the disjoint audit test the selected candidate trails at every budget and no
interval excludes zero (at 5%: -22.3 conversions, [-64.9, 20.2]). Response
targeting stays because nothing justifies changing it, not because uplift was
shown to be worse. This sample cannot separate the two.

## Stability

Ten complete repetitions of the protocol on the 500,000-row sample:

| Winning learner | Splits won |
|---|---:|
| S-learner | 6 / 10 |
| Transformed outcome | 2 / 10 |
| X-learner | 1 / 10 |
| DR-learner | 1 / 10 |

Mean difference +69.2 per 100,000 users (SD 60.9, range -10.8 to +171.8);
8 of 10 point estimates positive, but only 1 of 10 intervals wholly positive.

A winner that changes across splits can mean the rule is unstable or that the
candidates are tied, so each run also records how close the call was. The median
gap between first and second place is 32.0 incremental outcomes, against a median
half-width of 163.7 on the winner's own selection interval — a fifth of the
uncertainty in the number being ranked. Two splits are decided by margins of 0.5
and 3.3. In 3 of 10 runs no candidate reaches a positive selection bound at all,
and the rule still names a winner, because it ranks candidates rather than
requiring one to clear a bar.

**The selection rule does not identify a single best learner**, and the margins
say why: this sample cannot separate the candidates. The S-learner places first
or second in 7 of 10 runs, which is a ranking tendency rather than a win. The
defensible claim is about the policy class: uplift targeting at a tight budget
beats response targeting, and the S-learner is the instance that was locked before
the confirmatory sample was drawn. These splits overlap, so they are a sensitivity
analysis rather than ten independent experiments.

## Exploratory Work

Three side studies, kept because they inform the limits above rather than the
headline. Each is a single script and a single report.

- **Ground truth** ([semisynthetic_benchmark.md](outputs/semisynthetic_benchmark.md)) —
  real covariates with a known CATE, so the selection rule can be graded instead
  of trusted. It picked the transformed-outcome learner, which the ground truth
  confirms as the best of the eight (91.7% of oracle value). One draw, so this
  says the rule is not systematically broken, not that it is reliable.
- **Calibration** ([conversion_uplift_calibration.md](outputs/conversion_uplift_calibration.md)) —
  isotonic calibration made magnitude errors on the selected model *worse*
  (slope 1.140 to 0.783). Fitting a monotone correction on ~1,160 conversions is
  fragile. Not recommended.
- **Online design** ([online_experiment_design.md](outputs/online_experiment_design.md)) —
  three arms, 1,328,819 users, 80% power, randomized before ranking so the
  analysis is intention-to-treat rather than a comparison of two selected subsets.

## Limits

- The confirmatory sample is disjoint but drawn from the same experiment and
  period. It measures generalization to new users, not a new market or season.
- Intervals condition on the fitted policy. They exclude the uncertainty of
  having selected that learner; the repeated splits address that separately and
  are the weakest part of the evidence.
- Repeated splits overlap and are a sensitivity analysis, not replication.
- The conversion result is absence of evidence, not evidence of absence.
- Semi-synthetic findings depend on the response surface chosen.
- No hyperparameter tuning, one dataset, one time window.
- Offline evidence is not production impact.

## Repository

```text
scripts/     prepare samples, run each experiment, write reports to outputs/
src/data     Criteo loading, undersampling, semi-synthetic generator
src/models   response baseline, 8 learners, random reference, calibrator
src/evaluation  AIPW policy value, uplift curves, calibration, experiment design
src/experiments honest splitting and the locked protocol
tests/       48 tests, no data download required
outputs/     tracked evidence: reports, tables, figures
docs/        reproducibility notes
```

## Reproduction

Python 3.11 or 3.12. Place the Criteo dataset at `data/criteo-uplift-v2.1.csv.gz`.

```powershell
python -m pip install -r requirements.lock.txt

# Build samples. The first call indexes the source file (~1 min) and every
# later sample is drawn from that index.
python scripts\prepare_criteo.py --sample-size 500000 `
  --sample-path data\processed\criteo_sample_500k.parquet --random-state 42
python scripts\prepare_criteo.py --sample-size 2000000 `
  --sample-path data\processed\criteo_sample_2m.parquet --random-state 42
python scripts\prepare_audit_sample.py
python scripts\prepare_audit_sample.py `
  --excluded-paths "data/processed/criteo_sample_500k.parquet,data/processed/criteo_sample_2m.parquet,data/processed/criteo_audit_1m.parquet" `
  --output-path data\processed\criteo_confirm_4m.parquet `
  --sample-size 4000000 --random-state 20260730 `
  --report-path outputs\confirmatory_sample.md

# Visit: select, then confirm
python scripts\run_honest_criteo.py `
  --sample-path data/processed/criteo_audit_1m.parquet `
  --outcome visit --selection-folds 3 --random-state 777 `
  --report-path outputs\audit_visit_evaluation.md `
  --validation-path outputs\tables\audit_visit_selection.csv `
  --test-path outputs\tables\audit_visit_test.csv `
  --contrast-path outputs\tables\audit_visit_contrasts.csv `
  --figure-path outputs\figures\audit_visit_policy_value.png
python scripts\run_confirmatory_test.py
python scripts\run_honest_stability.py `
  --models "response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner"
```

The conversion, calibration, semi-synthetic, and online-design commands follow
the same shape; each script's `--help` lists its arguments, and the protocol
block at the top of every report in `outputs/` records the exact settings used.

## Quality Gates

```powershell
ruff check src scripts tests
pytest tests
```

Both run in CI on Python 3.11 and 3.12. No test downloads Criteo data.

## References

The four papers the reported result depends on. Formulas for individual
candidate learners are attributed in the docstring of the file that implements
them rather than listed here.

| Paper | Taken from it | Used in |
|---|---|---|
| [Diemert et al., *A Large Scale Benchmark for ITE Prediction and Uplift Modeling*](https://arxiv.org/abs/2111.10106) | The randomized Criteo dataset, and the `auuc_sep_rel_prop1` definition of relative AUUC so figures compare to the published benchmark | `src/evaluation/uplift.py` |
| [Athey and Wager, *Policy Learning with Observational Data*](https://arxiv.org/abs/1702.02896), *Econometrica* 89(1), 2021 | The AIPW-score form of policy value, as opposed to the average-effect form | `src/evaluation/policy_value.py` |
| [Chernozhukov et al., *Double/Debiased Machine Learning*](https://arxiv.org/abs/1608.00060), *The Econometrics Journal* 21(1), 2018 | Why nuisance models must be fit out of fold, and the term *cross-fitting* | `src/models/cross_fitting.py` |
| [Künzel et al., *Metalearners for Estimating Heterogeneous Treatment Effects*](https://arxiv.org/abs/1706.03461) | S-, T-, and X-learner, including the X-learner's propensity weighting | `src/models/{s,t,x}_learner.py` |

Everything else is project-specific: the identity-based sample construction, the
5% decision rule, the paired contrast, the three-stage protocol, the
semi-synthetic benchmark, the online design, and every conclusion drawn here.
