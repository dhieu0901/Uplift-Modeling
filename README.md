# Honest Uplift Modeling for Campaign Optimization

A campaign can contact only 5% of eligible users. Should it target the users most
likely to respond, or the users whose behaviour actually changes because of the
campaign?

On four million randomized users never touched by training or model selection,
uplift targeting delivers **+5,979 incremental visits at the 5% budget
(95% CI [5,006, 6,952]), a 46.3% improvement over the response-targeting
incumbent**. The advantage is decisive at tight budgets and disappears entirely
by a 30% budget. For the rarer conversion outcome, the same protocol finds the
opposite: response targeting wins, and the uplift candidate is decisively worse.

The result that holds up is about the *approach*, not a favourite model. Across
ten repeated splits, four different uplift learners take turns winning selection,
so this project claims that uplift targeting beats response targeting at a tight
budget — not that any one learner is uniquely best.

Every number in this document is backed by a tracked file under
[outputs/](outputs/).

## The Decision

| Outcome | Decision | Evidence |
|---|---|---|
| **Visit** | Adopt uplift targeting at a 5% budget, deploying the locked S-learner, and confirm with a live test | +5,979 visits [5,006, 6,952] on 4M untouched users, z = 12.0 |
| **Model choice** | Treat the S-learner as one acceptable instance, not the proven optimum | It wins only 4 of 10 repeated splits; X-, R-, and modified-outcome learners win the rest |
| **Conversion** | Keep response targeting | Uplift candidate is -57.6 conversions [-105.1, -10.1]; interval excludes zero on the wrong side |
| **Budget** | Keep the budget tight | The advantage is +46.3% at 5%, +14.7% at 10%, and statistically absent at 20% |
| **Economics** | No ROI claim | Real outcome value and contact cost are not available; the monetary section of the calibration report is an explicitly labelled worked example |
| **Next step** | Run the 1.28M-user online test in [outputs/online_experiment_design.md](outputs/online_experiment_design.md) | Offline evidence bounds what a live campaign might do; it does not replace randomized production measurement |

## The Problem

The incumbent ranks users by their predicted probability of responding after
treatment:

```text
response_score(X) = P(Y = 1 | X, treatment = 1)
```

This finds users who are likely to respond, but it never establishes that the
campaign *caused* the response. A user who would have visited anyway ranks
highly and consumes budget while producing nothing incremental. The quantity the
campaign actually cares about is the change in outcome probability:

```text
uplift(X) = P(Y = 1 | X, treatment = 1) - P(Y = 1 | X, treatment = 0)
```

## Why This Is Hard

The individual treatment effect is never observed: each user is either treated or
not, so the counterfactual is missing and there is no label to validate against.
Four further traps make a naive train/test benchmark misleading:

- selecting a model and reporting its effect on the same holdout inflates the
  result;
- conversion is rare (0.30%), so treatment-effect estimates are unstable;
- a model can rank well globally and still lose inside the operational top 5%;
- subtracting two separately estimated policy values discards the pairing
  information from the users the two policies disagree about.

## Evidence Design

The source experiment has 13,979,592 randomized rows. Four disjoint samples serve
different purposes, and disjointness is verified by full-row hash after
construction rather than assumed from it:

| Sample | Rows | Role |
|---|---:|---|
| Visit development | 500,000 | Repeated-split stability |
| Conversion development | 2,000,000 | Undersampling sweep and calibration |
| Audit | 1,000,000 | Model selection, then one internal locked test |
| **Confirmatory** | **4,000,000** | **Primary evidence, opened once** |

The confirmatory sample has an 84.38% treatment rate, a 4.96% visit rate, and a
0.30% conversion rate. Its control arm visits at 3.846% and its treated arm at
5.166%, so treating every user would produce about 52,795 incremental visits.
That total is the denominator worth judging a 5% budget against.

`count_overlapping_rows` reports **0 overlapping rows** between the confirmatory
sample and each of the three earlier samples
([outputs/confirmatory_sample.md](outputs/confirmatory_sample.md)).

### Protocol

Each experiment reserves 80% for development and 20% as a locked test. Three
joint treatment/outcome-stratified folds produce out-of-fold candidate scores and
nuisance predictions. The candidate with the largest lower bound of the paired
95% interval against response targeting at the 5% budget is selected, refit, and
evaluated once.

For a targeting policy `pi(X)`, value relative to treating nobody is:

```text
V(pi) - V(0) = E[pi(X) * (Y(1) - Y(0))]
```

estimated with the doubly robust score

```text
phi = mu_1(X) - mu_0(X)
      + T / e * (Y - mu_1(X))
      - (1 - T) / (1 - e) * (Y - mu_0(X))
```

The primary contrast is paired: `[pi_uplift(X) - pi_response(X)] * phi`, so
uncertainty comes only from users the two policies treat differently. Two policies
are always evaluated for context but can never win selection — `response_model`
(the incumbent to beat) and `random_targeting` (the floor that separates ranking
skill from the mere act of treating users).

## Results

### Visit: model selection

On 800,000 audit-development users, out-of-fold policy comparison at the 5%
budget selected the **S-learner**. These values perform selection only.

| Candidate | Difference vs response | 95% CI |
|---|---:|---:|
| **S-learner** | **+903.8** | **[469.5, 1,338.0]** |
| T-learner | +651.0 | [197.7, 1,104.2] |
| Modified outcome | +575.4 | [167.1, 983.6] |
| X-learner | +418.2 | [-25.9, 862.4] |
| R-learner | +314.6 | [-119.7, 748.8] |
| DR-learner | +301.7 | [-136.2, 739.6] |
| CVT | -1,152.4 | [-1,581.6, -723.2] |
| _random targeting_ | _-2,546.0_ | _[-3,058.4, -2,033.6]_ |

### Visit: the 200,000-row locked test was too small to decide

The selected S-learner was refit and compared on the untouched 200,000-row audit
test. It produced **+266.5 visits [43.8, 489.2]** at the 5% budget.

That interval barely clears zero, and a previous run of the identical protocol on
the identical partition produced **+168.5 [-53.0, 390.0]**, which does not. The
only difference between the two runs was how per-model random seeds were derived
(see [docs/determinism.md](docs/determinism.md)); the data, the split, and the
estimator were unchanged.

A conclusion that flips on model-fitting seeds is not a conclusion. Rather than
argue over which run to publish, the project spent unused rows on evaluation.

### Visit: confirmatory test on 4,000,000 untouched users

The locked S-learner and the two reference policies were refit on the full
audit sample and scored once on the confirmatory sample. Nothing was selected at
this stage.

| Policy at 5% budget | Incremental visits | 95% CI | Share of achievable effect |
|---|---:|---:|---:|
| **S-learner** | **18,884** | [17,930, 19,839] | 35.8% |
| Response targeting | 12,905 | [11,794, 14,016] | 24.4% |
| Random targeting | 1,697 | [1,295, 2,098] | 3.2% |

The paired contrast against response targeting:

| Budget | Difference | 95% CI | z | Relative gain |
|---:|---:|---:|---:|---:|
| **5%** | **+5,979** | **[5,006, 6,952]** | **12.04** | **+46.3%** |
| 10% | +3,012 | [2,193, 3,831] | 7.21 | +14.7% |
| 20% | +391 | [-96, 878] | 1.57 | +1.4% |
| 30% | -235 | [-644, 174] | -1.13 | -0.8% |

![Confirmatory policy value](outputs/figures/confirmatory_visit_policy_value.png)

Two things are worth noting. First, the per-user effect agrees across the two
independent evaluations — 133 visits per 100,000 users on the audit test and 149
per 100,000 on the confirmatory sample — so the twentyfold increase in sample size
bought precision, not a different answer. Second, the advantage decays with
budget exactly as theory predicts: an uplift ranking pays off when the budget
forces a genuine choice, and converges to the incumbent once most users are
treated anyway.

### Visit: the ranking metric disagrees with the decision

| Policy | Relative AUUC (4M confirmatory) |
|---|---:|
| Response targeting | 0.011798 |
| S-learner | 0.011639 |
| Random targeting | 0.006630 |

Response targeting has the *better* global AUUC while losing decisively at the 5%
budget. This is not noise at four million rows. It is a concrete demonstration
that a whole-ranking metric can rank policies differently from the operational
decision, and it is why the primary criterion here is a budget-specific policy
contrast rather than AUUC.

### Stability: the direction holds, the winning learner does not

Ten complete repetitions of the entire protocol — split, select from all eight
candidates, refit, open the test — were run on the 500,000-row development sample
with ten different seeds. Letting every candidate compete in every repetition is
what makes this informative, and it produced the single most important caveat in
the project.

| Winning learner | Splits won | Mean difference vs response |
|---|---:|---:|
| S-learner | 4 / 10 | +112.4 |
| X-learner | 2 / 10 | +95.1 |
| R-learner | 2 / 10 | +68.1 |
| Modified outcome | 2 / 10 | +33.7 |

**The selection rule does not identify a single best learner.** Four different
families win depending only on the split. Any claim that "the S-learner is the
best uplift model for this problem" is unsupported.

What *is* stable is the direction of the uplift-versus-response advantage:

| Runs | Mean difference per 100,000 users | SD | Range | Positive | Wholly positive CI |
|---:|---:|---:|---:|---:|---:|
| 10 | +84.3 visits | 68.6 | -52.9 to +217.0 | 9 / 10 | 1 / 10 |

![Repeated-split stability](outputs/figures/visit_stability.png)

Only one repetition in ten produced an interval that excluded zero, which is the
same lesson the seed-sensitivity result taught: a 100,000-row test cannot resolve
an effect this size. Relative AUUC favoured response targeting in 5 of the 10
runs — a coin flip — while the budget-specific contrast was positive in 9.

The per-user effect here (+84 per 100,000) is smaller than the confirmatory
sample's (+149 per 100,000). The most likely reason is training size: these models
see 300,000 rows, against 1,000,000 for the confirmatory refit. The two figures
answer different questions and should not be averaged.

Taken with the confirmatory test, the defensible claim is about the *policy
class*, not the model: **uplift targeting at a tight budget beats response
targeting, and the S-learner is the pre-registered instance of it that was locked
before the confirmatory sample was drawn.** Several other learners would very
likely have performed comparably.

### Conversion: response targeting wins

Treatment-stratified negative undersampling was swept over factors 1, 5, 10, 25,
50, 100, and 200 for T- and CVT-based logistic learners. Selection chose the
T-learner at **factor 1** — that is, no undersampling at all, which is itself the
answer to whether undersampling helps here.

On the disjoint audit test the selected candidate loses at every budget:

| Budget | T-learner k=1 minus response | 95% CI |
|---:|---:|---:|
| 5% | -57.6 conversions | [-105.1, -10.1] |
| 10% | -50.1 | [-90.2, -9.9] |
| 20% | -23.6 | [-63.5, 16.2] |
| 30% | -39.1 | [-73.7, -4.5] |

Three of the four intervals exclude zero on the negative side. Response targeting
stays as the conversion policy, and this is a decisive answer rather than an
inconclusive one.

### Conversion: calibration fixes magnitude, not ranking

An independent calibration holdout, comparing no undersampling against factor 5:

| Score | EUCE | MUCE | Intercept | Slope | Relative AUUC |
|---|---:|---:|---:|---:|---:|
| k=1 raw | 0.000229 | 0.000616 | -0.000029 | 1.004 | 0.001069 |
| k=1 calibrated | 0.000219 | 0.000537 | -0.000008 | 0.983 | 0.001058 |
| k=5 raw | 0.000318 | 0.001015 | 0.000000 | 0.889 | 0.001059 |
| k=5 calibrated | 0.000196 | 0.000469 | 0.000004 | 0.971 | 0.001085 |

Undersampling at factor 5 distorts magnitude — its raw slope is 0.889, against
1.004 for the un-undersampled model — and isotonic calibration repairs most of
that. Ranking is essentially untouched in every case. Calibration therefore buys
absolute interpretability, not policy value, which is why the fixed-budget rule
remains the operating decision.

### Ground truth: the selection rule can still pick the wrong model

Real Criteo covariates were combined with a known nonlinear response surface, so
the true CATE and the exact policy value are available. This is the one place the
protocol can be graded rather than trusted.

| Policy at 5% budget | True incremental outcomes | Share of oracle | PEHE | Spearman vs true CATE |
|---|---:|---:|---:|---:|
| Oracle | 61.61 | 100.0% | 0.0000 | 1.000 |
| **Modified outcome** | **54.08** | **87.8%** | 0.0043 | 0.850 |
| S-learner | 46.04 | 74.7% | 0.0099 | 0.606 |
| X-learner | 44.85 | 72.8% | 0.0141 | 0.516 |
| _T-learner (selected)_ | _43.54_ | _70.7%_ | 0.0226 | 0.391 |
| CVT | 42.64 | 69.2% | 0.0469 | 0.400 |
| DR-learner | 41.87 | 68.0% | 0.0209 | 0.466 |
| R-learner | 40.92 | 66.4% | 0.0210 | 0.460 |
| Response targeting | 40.12 | 65.1% | 0.0534 | 0.300 |
| Random targeting | 29.79 | 48.4% | — | — |

The out-of-fold AIPW rule selected the **T-learner** at 70.7% of oracle value,
while the truth favoured **modified outcome** at 87.8%. All seven candidates sat
inside each other's confidence intervals, so the rule picked the least-negative
lower bound among statistically indistinguishable options and lost roughly 17
points of achievable value.

This is the honest limit of the whole approach: a well-motivated estimator can
still mis-rank models in finite samples. It is the strongest argument for the
online test, and the reason model selection is reported separately from the
locked evaluation rather than blended into one number. Note also that random
targeting reaches 48.4% of oracle value here — on a response surface where most
effects are positive, spending the budget at all does much of the work, which is
exactly why the random reference belongs in every table.

### Online challenger design

Because the offline effect estimate is now larger and tighter, the experiment
needed to confirm it is substantially cheaper. The design retains 75% of the
offline difference, uses 80% power, two-sided 5% significance, and a 15%
operational buffer:

| Arm | Policy | Target rate | Users | Expected visit rate |
|---|---|---:|---:|---:|
| A | S-learner | 5% | 581,201 | 0.043182 |
| B | Response targeting | 5% | 581,201 | 0.041687 |
| H | No-campaign holdout | 0% | 117,461 | 0.038461 |
| **Total** | | | **1,279,863** | |

An earlier design built on the weaker offline estimate required 3,755,532 users.
Sharpening the offline evidence cut the online test to about a third of its
former size, which is the practical payoff of spending idle rows on evaluation.

Users are randomized to complete policy arms *before* ranking. The primary
analysis is the intention-to-treat A-minus-B visit-rate difference across all
assigned users, not a comparison of the two targeted subsets.

## What Changed In This Revision

The project previously reported an inconclusive headline. The change came from
methodology and engineering, not from a new model:

| Change | Effect |
|---|---|
| Confirmatory test on 4M unused rows | z went from 2.35 to 12.04; the decision is no longer seed-dependent |
| Name-derived model seeds | Adding a candidate no longer reseeds the others, so runs are comparable |
| `random_targeting` reference policy | Reveals that response targeting is 7.6x random and the S-learner 11.1x |
| Stability run opened to all 8 candidates | Exposed that the winning learner changes across splits; the previous run fixed a single candidate and could not detect this |
| `n_jobs=-1` after proving bit-exact determinism | 4.6x faster fits, identical numbers, so the full evidence set is rebuildable |
| `evaluate_locked_policies` extracted | One code path serves the internal locked test and the confirmatory test |
| Disjointness re-verified at evaluation time | The confirmatory script refuses to run on overlapping rows |
| `outputs/` now tracked in git | Every published number points at a file instead of asking for trust |
| Removed a fake correction in `UndersampledCVTLearner` | It divided by the undersampling factor, which changes no ranking and implied a calibration that never happened |
| Stricter ruff rule set, pinned lockfile, CI | Quality gates run on every push instead of by hand |
| Tests: 34 to 45 | Added 14 covering reporting, reference policies, seed stability, overlap detection, and locked evaluation; dropped 3 along with the dead code they exercised |
| Deleted unused code | `auuc`, `cumulative_uplift_curve`, `uplift_by_quantile`, `budget_policy_table`, and `select_best_campaign` were superseded by the AIPW path and called by nothing in the pipeline |

The earlier claim of a 66.3% visit improvement remains retracted: it reused test
evidence for both selection and reporting. It is kept in this history as a record
of the mistake, not as a result.

## Repository

```text
.
|-- .github/workflows/ci.yml     # ruff + pytest on Python 3.11 and 3.12
|-- README.md
|-- pyproject.toml               # ruff, pytest, and type-checker configuration
|-- requirements.txt             # minimum versions; CI installs this so that a
|                                # version drift breaks the build early
|-- requirements.lock.txt        # exact versions behind the published numbers;
|                                # install this to reproduce them digit for digit
|-- docs/determinism.md          # measured reproducibility guarantees
|-- data/                        # git-ignored; rebuilt by scripts
|-- outputs/                     # tracked evidence: reports, tables, figures
|-- scripts/
|   |-- prepare_criteo.py
|   |-- prepare_audit_sample.py
|   |-- run_honest_criteo.py         # selection + internal locked test
|   |-- run_confirmatory_test.py     # locked policies vs a fresh large sample
|   |-- run_honest_stability.py
|   |-- analyze_uplift_calibration.py
|   |-- run_semisynthetic_benchmark.py
|   `-- design_online_experiment.py
|-- src/
|   |-- data/                    # criteo loading, imbalance, semi-synthetic
|   |-- models/                  # response, S/T/X/CVT/MO/R/DR, undersampled,
|   |                            # random reference, isotonic calibrator
|   |-- evaluation/              # AIPW policy value, uplift curves,
|   |                            # calibration, ground truth, experiment design
|   |-- experiments/             # honest splitting and the locked protocol
|   `-- reporting.py             # markdown tables and policy-value plots
`-- tests/                       # 45 tests, no data download required
```

## Reproduction

Python 3.11 or 3.12.

### 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```

### 2. Build the samples

Place the Criteo Uplift Prediction Dataset v2.1 at
`data/criteo-uplift-v2.1.csv.gz`, then:

```powershell
python scripts\prepare_criteo.py --sample-size 500000 `
  --sample-path data\processed\criteo_sample_500k.parquet --random-state 42
python scripts\prepare_criteo.py --sample-size 2000000 `
  --sample-path data\processed\criteo_sample_2m.parquet --random-state 42
python scripts\prepare_audit_sample.py

python scripts\prepare_audit_sample.py `
  --excluded-paths "data\processed\criteo_sample_500k.parquet,data\processed\criteo_sample_2m.parquet,data\processed\criteo_audit_1m.parquet" `
  --output-path data\processed\criteo_confirm_4m.parquet `
  --sample-size 4000000 --random-state 20260730 `
  --report-path outputs\confirmatory_sample.md
```

### 3. Rebuild the evidence

```powershell
# Visit: selection, internal locked test, then the confirmatory test
python scripts\run_honest_criteo.py `
  --sample-path data\processed\criteo_audit_1m.parquet `
  --outcome visit --selection-folds 3 --random-state 777 `
  --report-path outputs\audit_visit_evaluation.md `
  --validation-path outputs\tables\audit_visit_selection.csv `
  --test-path outputs\tables\audit_visit_test.csv `
  --contrast-path outputs\tables\audit_visit_contrasts.csv `
  --figure-path outputs\figures\audit_visit_policy_value.png

python scripts\run_confirmatory_test.py

python scripts\run_honest_stability.py `
  --models "response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner" `
  --report-path outputs\visit_stability.md `
  --results-path outputs\tables\visit_stability.csv `
  --figure-path outputs\figures\visit_stability.png

# Conversion: undersampling sweep, audit confirmation, calibration
python scripts\run_honest_criteo.py `
  --sample-path data\processed\criteo_sample_2m.parquet `
  --outcome conversion --models response_model --selection-folds 3 `
  --undersampling-factors 1,5,10,25,50,100,200 `
  --report-path outputs\rare_conversion_development.md `
  --validation-path outputs\tables\rare_conversion_selection.csv `
  --test-path outputs\tables\rare_conversion_internal_holdout.csv `
  --contrast-path outputs\tables\rare_conversion_internal_contrasts.csv `
  --figure-path outputs\figures\rare_conversion_development.png

python scripts\run_honest_criteo.py `
  --sample-path data\processed\criteo_audit_1m.parquet `
  --outcome conversion --models response_model --selection-folds 3 `
  --undersampling-factors 1 --undersampling-families t --random-state 777 `
  --report-path outputs\audit_conversion_evaluation.md `
  --validation-path outputs\tables\audit_conversion_selection.csv `
  --test-path outputs\tables\audit_conversion_test.csv `
  --contrast-path outputs\tables\audit_conversion_contrasts.csv `
  --figure-path outputs\figures\audit_conversion_policy_value.png

python scripts\analyze_uplift_calibration.py `
  --models "undersampled_t_lr_k1,undersampled_t_lr_k5" `
  --undersampling-factors "1,5" --undersampling-families t

# Ground truth and the online design
python scripts\run_semisynthetic_benchmark.py

python scripts\design_online_experiment.py `
  --input-path outputs\tables\confirmatory_visit_test.csv `
  --policy-a s_learner --policy-b response_model `
  --budget-pct 5.0 --no-campaign-rate 0.038461
```

## Quality Gates

```powershell
ruff check src scripts tests
pytest tests
```

Both run in CI on every push. The test suite never downloads Criteo data: each
test builds its own fixture or uses the semi-synthetic generator.

## Interpretation Boundaries

- The confirmatory sample is disjoint from every earlier sample but comes from
  the same source experiment and time period. It measures generalization to new
  users, not to a new market or season.
- Confidence intervals condition on the fitted policies. They do not include the
  uncertainty of having selected the S-learner; that is what the repeated-split
  experiment addresses separately.
- AIPW treats the treatment propensity as known, which is standard for a
  randomized design. The propensity is estimated from the refit sample, and that
  estimation variance is not propagated.
- Repeated splits overlap and are sensitivity analyses, not independent
  replications.
- Semi-synthetic findings depend on the chosen response surface.
- Offline evidence, however tight, is not production impact. The online test in
  [outputs/online_experiment_design.md](outputs/online_experiment_design.md) is
  the step that would justify a permanent rollout.

## Selected References

Three references anchor the data, the core model family, and the rare-outcome
intervention:

1. [Diemert et al., *A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling*](https://arxiv.org/abs/2111.10106) — source of the randomized Criteo dataset and the relative AUUC definition used here.
2. [Künzel et al., *Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning*](https://arxiv.org/abs/1706.03461) — basis for the S-, T-, and X-learner family.
3. [Nyberg, Kuśmierczyk, and Klami, *Uplift Modeling with High Class Imbalance*](https://proceedings.mlr.press/v157/nyberg21a.html) — basis for treatment-stratified negative undersampling and rare-outcome calibration.

These are implementation anchors, not a literature review. The audit and
confirmatory sample construction, the 5% decision rule, the paired evaluation,
the semi-synthetic design, the online-test plan, and all conclusions are
project-specific.
