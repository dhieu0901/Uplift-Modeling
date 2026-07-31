# Honest Uplift Modeling for Campaign Optimization

A campaign can contact only 5% of eligible users. Should it target the users most
likely to respond, or the users whose behaviour actually changes because of the
campaign?

On four million randomized users never touched by training or model selection,
uplift targeting delivers **+5,861 incremental visits at the 5% budget
(95% CI [4,851, 6,871]), a 47.1% improvement over the response-targeting
incumbent**. The advantage is decisive at tight budgets and gone by a 20% budget.

Two qualifications come with that headline, and both are load-bearing. The
result is about the *approach*, not a favourite model: across ten repeated splits
four different uplift learners take turns winning selection. And for the rarer
conversion outcome the same protocol reaches no conclusion at all.

Every number here is backed by a tracked file under [outputs/](outputs/).

## The Decision

| Question | Decision | Evidence |
|---|---|---|
| **Visit** | Adopt uplift targeting at a 5% budget, deploying the locked S-learner, then confirm live | +5,861 visits [4,851, 6,871] on 4M untouched users, z = 11.4 |
| **Model choice** | Treat the S-learner as one acceptable instance, not the proven optimum | It wins 6 of 10 repeated splits; three other learners win the rest |
| **Budget** | Keep the budget tight | +47.1% at 5%, +14.8% at 10%, and no measurable advantage at 20% or 30% |
| **Conversion** | Keep response targeting | The uplift candidate is behind at every budget but no interval excludes zero; there is no positive case for switching |
| **Calibration** | Do not apply it | On the selected model isotonic calibration made magnitude errors worse, not better |
| **Economics** | No ROI claim | Real outcome value and contact cost are unavailable; the currency figures in the calibration report are a labelled worked example |
| **Next step** | Run the 1.33M-user online test in [outputs/online_experiment_design.md](outputs/online_experiment_design.md) | Offline evidence bounds what a live campaign might do; it does not replace randomized production measurement |

## The Problem

The incumbent ranks users by their predicted probability of responding after
treatment:

```text
response_score(X) = P(Y = 1 | X, treatment = 1)
```

This finds users likely to respond but never establishes that the campaign
*caused* the response. The scale of the problem is visible in the raw data: among
users who visit after being treated, most would have visited anyway. Treating the
whole population lifts the visit rate from 3.820% to 4.854% — so roughly
three-quarters of the "responses" a response model is trained to predict would
have happened with no campaign at all.

The quantity the campaign actually cares about is the change in outcome
probability:

```text
uplift(X) = P(Y = 1 | X, treatment = 1) - P(Y = 1 | X, treatment = 0)
```

## Why This Is Hard

The individual treatment effect is never observed: each user is either treated or
not, so the counterfactual is missing and there is no label to validate against.
Four further traps make a naive train/test benchmark misleading:

- selecting a model and reporting its effect on the same holdout inflates the
  result;
- conversion is rare (0.29%), so treatment-effect estimates are unstable;
- a model can rank well globally and still lose inside the operational top 5%;
- subtracting two separately estimated policy values discards the pairing
  information from the users the two policies disagree about.

## Evidence Design

The source experiment has 13,979,592 randomized rows and a population treatment
effect of **+1.0342 pp** on visits. Four disjoint samples serve different
purposes:

| Sample | Rows | Measured effect | Role |
|---|---:|---:|---|
| Visit development | 500,000 | +0.9607 pp | Repeated-split stability |
| Conversion development | 2,000,000 | +1.1257 pp | Undersampling sweep and calibration |
| Audit | 1,000,000 | +1.0827 pp | Model selection, then one internal locked test |
| **Confirmatory** | **4,000,000** | **+1.0052 pp** | **Primary evidence, opened once** |

Every sample is statistically indistinguishable from the population, which is a
property the project had to work for — see the next section.

### Samples are made disjoint by identity, not by value

`prepare_criteo_index` writes the source file once with a `row_id` equal to the
row's position, and later samples exclude the `row_id`s already spent.

An earlier version excluded rows whose hash of *all columns* matched a used row.
That sounds equivalent and is not, because the source contains duplicate rows,
and those duplicates are not neutral:

| Group | Rows | Treated | Treatment effect |
|---|---:|---:|---:|
| Values duplicated elsewhere | 2,221,150 | 95.71% | +0.002 pp |
| Values unique | 11,758,442 | 82.98% | +1.910 pp |

The duplicated rows are inert — no visits, no conversions, overwhelmingly in the
treated arm. Hash exclusion drops *every* copy once one is drawn, so each
successive sample shed more inert rows and its measured effect drifted upward:
the audit sample read +1.2241 pp (3.5 SE above the population) and the
confirmatory sample +1.3199 pp (10.5 SE above). **The headline was inflated by
about 28% before this was found.**

The 500,000-row sample excludes nothing and was unaffected either way; the gap
between it and the confirmatory sample is what exposed the drift. Under identity
exclusion all four samples land within about 1 SE of the population, and a
sanity check falls into place that had previously failed: random targeting now
captures 4.2% of the achievable effect at a 5% budget, against the 5% that a
policy carrying no information should collect.

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
uncertainty comes only from users the two policies treat differently. Two
policies are always evaluated for context but can never win selection —
`response_model` (the incumbent) and `random_targeting` (the floor that separates
ranking skill from the mere act of treating users).

## Results

### Visit: model selection

On 800,000 audit-development users, out-of-fold comparison at the 5% budget
selected the **S-learner**. These values perform selection only.

| Candidate | Difference vs response | 95% CI |
|---|---:|---:|
| **S-learner** | **+928.0** | **[491.5, 1,364.4]** |
| X-learner | +322.9 | [-113.7, 759.4] |
| Modified outcome | +291.6 | [-131.7, 714.9] |
| DR-learner | +267.6 | [-168.7, 704.0] |
| R-learner | +136.8 | [-294.0, 567.6] |
| T-learner | +38.9 | [-408.7, 486.4] |
| CVT | -482.2 | [-906.0, -58.4] |
| _random targeting_ | _-2,678.1_ | _[-3,193.3, -2,162.8]_ |

### Visit: confirmatory test on 4,000,000 untouched users

The locked S-learner and the two reference policies were refit on the full audit
sample and scored once on the confirmatory sample. Nothing was selected here.

| Policy at 5% budget | Incremental visits | 95% CI | Share of achievable effect |
|---|---:|---:|---:|
| **S-learner** | **18,293** | [17,329, 19,257] | 45.5% |
| Response targeting | 12,433 | [11,293, 13,572] | 30.9% |
| Random targeting | 1,698 | [1,293, 2,102] | 4.2% |

Treating all four million users would produce about 40,210 incremental visits.
The S-learner collects 45.5% of that while contacting 5% of people.

The paired contrast against response targeting:

| Budget | Difference | 95% CI | z | Relative gain |
|---:|---:|---:|---:|---:|
| **5%** | **+5,861** | **[4,851, 6,871]** | **11.37** | **+47.1%** |
| 10% | +2,950 | [2,076, 3,825] | 6.61 | +14.8% |
| 20% | -421 | [-1,019, 176] | -1.38 | -1.6% |
| 30% | -432 | [-908, 45] | -1.77 | -1.6% |

![Confirmatory policy value](outputs/figures/confirmatory_visit_policy_value.png)

The advantage decays with budget exactly as theory predicts: an uplift ranking
pays off when the budget forces a genuine choice and is worth nothing once most
users are treated anyway. The internal 200,000-row locked test on the audit
sample pointed the same way (+380.5 visits [163.6, 597.5], z = 3.44) but with a
standard error over four times larger per user — which is why the confirmatory
sample was drawn rather than the smaller result being published.

### Visit: the ranking metric disagrees with the decision

| Policy | Relative AUUC (4M confirmatory) |
|---|---:|
| Response targeting | 0.009130 |
| S-learner | 0.009037 |
| Random targeting | 0.004958 |

Response targeting has the *better* global AUUC while losing decisively at the 5%
budget, and the same inversion appears in 4 of the 10 repeated splits. A
whole-ranking metric can rank policies differently from the operational decision,
which is why the primary criterion is a budget-specific policy contrast.

### Stability: the direction holds, the winning learner does not

Ten complete repetitions of the entire protocol — split, select from all eight
candidates, refit, open the test — on the 500,000-row sample:

| Winning learner | Splits won |
|---|---:|
| S-learner | 6 / 10 |
| Modified outcome | 2 / 10 |
| X-learner | 1 / 10 |
| DR-learner | 1 / 10 |

**The selection rule does not identify a single best learner.** Any claim that
the S-learner is *the* right model for this problem is unsupported.

| Runs | Mean difference per 100,000 users | SD | Range | Positive | Wholly positive CI |
|---:|---:|---:|---:|---:|---:|
| 10 | +69.2 visits | 60.9 | -10.8 to +171.8 | 8 / 10 | 1 / 10 |

![Repeated-split stability](outputs/figures/visit_stability.png)

Only one repetition in ten produced an interval excluding zero: a 100,000-row
test cannot resolve an effect this size, which is the same lesson the
confirmatory sample was built to answer. The per-user effect here (+69 per
100,000) is below the confirmatory sample's (+146.5 per 100,000); these models
see 300,000 training rows against 1,000,000, and the two figures answer
different questions.

The defensible claim is about the *policy class*: **uplift targeting at a tight
budget beats response targeting, and the S-learner is the pre-registered instance
of it that was locked before the confirmatory sample was drawn.**

### Conversion: no case for switching

Treatment-stratified negative undersampling was swept over factors 1, 5, 10, 25,
50, 100, and 200 for T- and CVT-based logistic learners. Selection chose the
T-learner at **factor 1** — no undersampling at all, which is itself the answer
to whether undersampling helps here.

On the disjoint audit test the selected candidate trails at every budget, but no
interval excludes zero:

| Budget | T-learner k=1 minus response | 95% CI |
|---:|---:|---:|
| 5% | -22.3 conversions | [-64.9, 20.2] |
| 10% | -15.8 | [-58.0, 26.3] |
| 20% | -8.8 | [-44.2, 26.7] |
| 30% | -8.1 | [-43.4, 27.2] |

At the 5% budget response targeting delivers 192.6 incremental conversions
[86.5, 298.6] against the candidate's 170.3 [70.1, 270.4]. Response targeting
stays, on the grounds that nothing here justifies changing it — not on the
grounds that uplift was proven worse. With 0.29% of users converting, this
sample cannot separate the two.

### Conversion: calibration did not help

An independent calibration holdout, comparing no undersampling against factor 5:

| Score | EUCE | MUCE | Intercept | Slope | Relative AUUC |
|---|---:|---:|---:|---:|---:|
| k=1 raw | 0.000224 | 0.000795 | -0.000104 | 1.140 | 0.001021 |
| k=1 calibrated | 0.000375 | 0.001909 | 0.000052 | 0.783 | 0.001023 |
| k=5 raw | 0.000369 | 0.001696 | 0.000084 | 0.784 | 0.000961 |
| k=5 calibrated | 0.000361 | 0.001690 | 0.000109 | 0.780 | 0.000978 |

The selected model is already close to calibrated when raw (slope 1.140), and
isotonic calibration overcorrects it to 0.783, roughly doubling both error
measures. For factor 5 the improvement is negligible. Ranking is essentially
untouched throughout, which is consistent with the earlier finding that
calibration moves magnitude rather than policy value — but here it moves it the
wrong way. Fitting a monotone correction on roughly 1,160 conversions is fragile,
and the recommendation is not to apply it.

### Ground truth: the selection rule got this one right

Real Criteo covariates were combined with a known nonlinear response surface, so
the true CATE and exact policy value are available. This is the one place the
protocol can be graded rather than trusted.

| Policy at 5% budget | True incremental outcomes | Share of oracle | PEHE | Spearman vs true CATE |
|---|---:|---:|---:|---:|
| Oracle | 61.44 | 100.0% | 0.0000 | 1.000 |
| **Modified outcome (selected)** | **56.35** | **91.7%** | 0.0034 | 0.889 |
| S-learner | 47.26 | 76.9% | 0.0092 | 0.641 |
| T-learner | 44.35 | 72.2% | 0.0220 | 0.445 |
| X-learner | 43.44 | 70.7% | 0.0143 | 0.558 |
| DR-learner | 42.67 | 69.5% | 0.0204 | 0.506 |
| R-learner | 42.64 | 69.4% | 0.0205 | 0.520 |
| CVT | 41.67 | 67.8% | 0.0461 | 0.293 |
| Response targeting | 38.47 | 62.6% | 0.0524 | 0.353 |
| Random targeting | 29.85 | 48.6% | — | — |

Here the out-of-fold AIPW rule selected the modified-outcome learner, which the
ground truth confirms as the best of the eight at 91.7% of oracle value. That is
reassuring but not a guarantee: it is one draw, and the repeated-split experiment
shows the winner moving around. Read together, the two say the selection rule is
not systematically broken but is noisy. Note also that random targeting reaches
48.6% of oracle value on this response surface — where most effects are positive,
spending the budget at all does much of the work, which is exactly why the random
reference belongs in every table.

### Online challenger design

The design retains 75% of the offline difference, uses 80% power, two-sided 5%
significance, and a 15% operational buffer:

| Arm | Policy | Target rate | Users | Expected visit rate |
|---|---|---:|---:|---:|
| A | S-learner | 5% | 601,392 | 0.042906 |
| B | Response targeting | 5% | 601,392 | 0.041441 |
| H | No-campaign holdout | 0% | 126,035 | 0.038333 |
| **Total** | | | **1,328,819** | |

Users are randomized to complete policy arms *before* ranking. The primary
analysis is the intention-to-treat A-minus-B visit-rate difference across all
assigned users, not a comparison of the two targeted subsets.

## What This Revision Changed

| Change | Effect |
|---|---|
| Samples made disjoint by row identity, not by value hash | Removed a 28% inflation of the measured treatment effect; all four samples now match the population |
| Confirmatory test on 4M unused rows | z of 11.4 against 3.4 on the internal 200,000-row test |
| `random_targeting` reference policy | Supplies the sanity check that caught the sampling bias, and shows response targeting at 7.3x random |
| Stability run opened to all 8 candidates | Exposed that the winning learner changes across splits |
| Name-derived model seeds | Adding a candidate no longer reseeds the others, so runs stay comparable |
| `n_jobs=-1` after proving bit-exact determinism | 4.6x faster fits, identical numbers |
| `outputs/` tracked in git | Every published number points at a file instead of asking for trust |
| Removed a fake correction in `UndersampledCVTLearner` | It divided by the undersampling factor, which changes no ranking and implied a calibration that never happened |
| Deleted five unused public functions | Superseded by the AIPW path and called by nothing in the pipeline |
| Stricter ruff rules, pinned lockfile, CI on 3.11 and 3.12 | Quality gates run on every push; CI caught two defects a local run had hidden |

An earlier claim of a 66.3% visit improvement remains retracted: it reused test
evidence for both selection and reporting.

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
|   |-- prepare_criteo.py            # index the source, then draw a sample
|   |-- prepare_audit_sample.py      # draw a sample disjoint by row identity
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
`-- tests/                       # 48 tests, no data download required
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
`data/criteo-uplift-v2.1.csv.gz`. The first call indexes the source file, which
takes about a minute and is reused by every later sample.

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
  --budget-pct 5.0 --no-campaign-rate 0.038333
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
- Confidence intervals condition on the fitted policies. They do not carry the
  uncertainty of having selected the S-learner; the repeated-split experiment
  addresses that separately, and it is the weakest part of the evidence.
- AIPW treats the treatment propensity as known, standard for a randomized
  design. It is estimated from the refit sample and that variance is not
  propagated.
- Repeated splits overlap and are sensitivity analyses, not independent
  replications.
- Semi-synthetic findings depend on the chosen response surface.
- The conversion result is an absence of evidence, not evidence of absence.
- Offline evidence, however tight, is not production impact. The online test is
  the step that would justify a permanent rollout.

## Selected References

1. [Diemert et al., *A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling*](https://arxiv.org/abs/2111.10106) — source of the randomized Criteo dataset and the relative AUUC definition used here.
2. [Künzel et al., *Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning*](https://arxiv.org/abs/1706.03461) — basis for the S-, T-, and X-learner family.
3. [Nyberg, Kuśmierczyk, and Klami, *Uplift Modeling with High Class Imbalance*](https://proceedings.mlr.press/v157/nyberg21a.html) — basis for treatment-stratified negative undersampling and rare-outcome calibration.
4. [Robins, Rotnitzky, and Zhao, *Estimation of Regression Coefficients When Some Regressors Are Not Always Observed*](https://www.semanticscholar.org/paper/46c56845fbb9e9452a318d736356949bd24fa012) — *JASA* 89(427):846–866, 1994. Origin of the augmented inverse-probability-weighted estimator this project uses to score policies.

These are implementation anchors, not a literature review. The sample
construction, the 5% decision rule, the paired evaluation, the semi-synthetic
design, the online-test plan, and all conclusions are project-specific.
