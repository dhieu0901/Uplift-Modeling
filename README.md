# Uplift Modeling for a Budget-Constrained Campaign

A campaign can contact 5% of eligible users. Should it target the users most
likely to respond, or the users whose behaviour actually *changes* because of
the campaign?

On four million randomized users never touched by training or model selection,
uplift targeting delivers **+5,861 incremental visits at a 5% budget
(95% CI [4,851, 6,871]), 47% above the response-targeting incumbent**.

Two limits come with that number. The advantage is gone by a 20% budget. And
for the rarer conversion outcome the same protocol reaches no conclusion at all.

Every figure below is backed by a tracked file: a report in
[outputs/](outputs/), or the [notebook](notebooks/) whose executed cell
produced it.

## Decisions

| Question | Decision | Evidence |
|---|---|---|
| Visit targeting | Adopt uplift targeting at 5%, then confirm live | +5,861 visits [4,851, 6,871] on 4M untouched users, z = 11.4 |
| Which model | Treat the S-learner as one acceptable instance, not the proven best | Two independent tests agree. Varying the data, it wins 8 of 10 repeated splits and never leaves the top two. Varying the estimator underneath it, three base learners crown three different winners and it falls to 6th of 7 on a linear base. Neither test separates a winner from its runner-up |
| Budget | Keep it tight | +47% at 5%, +15% at 10%, no measurable advantage at 20% or 30% |
| Conversion | Keep response targeting | No interval excludes zero at any budget - no case for switching either way |
| Next step | Run the randomized test in [online_experiment_design.md](outputs/online_experiment_design.md) | Offline evidence bounds what a live campaign might do; it does not replace it |

## The Problem

The incumbent ranks by `P(Y = 1 | X, T = 1)` - who is likely to respond. That
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

Criteo Uplift v2.1 - **13,979,592 rows**, treatment randomized at 85%, visit
4.70%, conversion 0.29%, average treatment effect **+1.0342 pp** on visits, all
measured in [sample_provenance.md](outputs/sample_provenance.md).
Randomization is the load-bearing property: without it, every causal estimate
would rest on an untestable "all confounders controlled" assumption.

**The feature set is the twelve columns as they arrive.** `f0` to `f11` are
anonymised floats, so there is no domain meaning to build on and none is
assumed. Nothing is derived, binned, or dropped. That is a decision, not an
omission: the candidates are trees, which are invariant to any monotone
rescaling, and inventing interactions for columns whose meaning is unknown adds
degrees of freedom without adding information. The distributions are far from
tidy - on the development sample `f1` takes 23 distinct values and `f11` has a
skew of -14.8 - which is an argument for trees rather than an argument for
cleaning. Where scaling does matter the estimator handles it: the linear base
learner standardises inside its own pipeline, so the transform is fitted per
fold and cannot leak. All twelve are pre-treatment, and the balance check
confirms it - the largest absolute standardised mean difference between arms is
0.0451, against the 0.1 that is usually treated as meaningful
([01_eda.ipynb](notebooks/01_eda.ipynb)).

**One column is excluded.** `exposure` records whether the ad actually rendered,
which is decided *after* assignment. On the development sample, visit rate is
41.97% among exposed users against 3.52% among treated-but-unexposed - an ad
renders only while someone is already browsing, so the column identifies people
already on their way to visit. Conditioning on it is collider bias. The proof is
that treated-unexposed (3.52%) sits *below* control (3.94%), which cannot happen
under intact randomization.
Only 3.6% of treated users were exposed, so the estimand is intention-to-treat:
the effect of choosing to reach someone, which is the only lever a campaign has.

The question the column invites is still answerable, just not by that comparison.
No control user can be exposed, so assignment is a one-sided instrument for it
and the Wald ratio identifies the effect among users whose ad renders:
**+26.65 pp on visits [22.34, 30.96]**, against the +38.02 pp the naive reading
gives. The 11.38 pp difference is selection, not campaign; those users were
already 3.9x more likely to visit before any ad was shown.
[exposure_iv.md](outputs/exposure_iv.md) reports both outcomes. Nothing there
can be targeted on, because at selection time it is not known whose ad will
render.

Four samples, separated by row identity rather than by value
([sample_provenance.md](outputs/sample_provenance.md)):

| Sample | Rows | Measured effect | From population | Role |
|---|---:|---:|---:|---|
| Development | 500,000 | +0.9607 pp | -0.94 SE | Exposure diagnostics, semi-synthetic covariates |
| Conversion development | 2,000,000 | +1.0200 pp | -0.37 SE | Rare-outcome sweep and calibration |
| Audit | 1,000,000 | +1.0827 pp | +0.89 SE | Model selection, stability, one internal locked test |
| **Confirmatory** | **4,000,000** | **+1.0052 pp** | **-1.06 SE** | **Primary evidence, opened once** |

All four land within about one standard error of the population, and all six
pairs share no rows. Each sample is drawn from the indexed source with every row
already spent by an earlier sample removed first, so separation is a property of
how the samples were built rather than of how they happened to land. The
distance column is printed rather than summarised because it is the check that
would show a draw going wrong.

Identity is what makes that verifiable. The source holds 2,221,150 rows whose
values duplicate another row, and they are almost entirely inert (+0.002 pp
effect, against +1.910 pp among unique rows). Excluding by value would drop every
copy as soon as one was drawn, so later samples would lose inert rows and read
high. Each row carries a `row_id`, and `count_overlapping_rows` verifies
separation as a query over every pair rather than only the pairs a result
happens to rest on.

## Method

**Evaluation.** Policy value is estimated with the AIPW score

```text
phi = mu_1(X) - mu_0(X)
      + T / e * (Y - mu_1(X))
      - (1 - T) / (1 - e) * (Y - mu_0(X))
```

and the value of a targeting policy relative to treating nobody is
`E[pi(X) * phi]`. AIPW is doubly robust - unbiased if either the outcome models
or the propensity is right. In a randomized design the propensity is *known*, so
that condition holds by construction, and the estimate stays unbiased however
badly the outcome models fit. That is why the nuisance models are one fixed
LightGBM configuration and are never tuned: evaluation does not need them to be
good, only to be out of fold. The candidates are a separate matter. Their fixed
configuration is a limit, not a consequence of this argument, and
[base_learner_comparison.md](outputs/base_learner_comparison.md) measures what
it costs.

**Paired contrast.** The primary comparison is
`[pi_uplift(X) - pi_response(X)] * phi`, not the difference of two separately
estimated values. Users both policies select cancel, leaving uncertainty only
where the policies disagree - necessary to resolve a difference inside the top 5%.

**Protocol.** 80% development, 20% locked test, stratified on the joint
treatment×outcome strata. Three cross-fitted folds produce out-of-fold candidate
scores and out-of-fold nuisances. The candidate with the largest lower bound
against response targeting at the 5% budget is selected, refit, and evaluated
once. After selection only the champion and two reference policies are scored on
test, so no result exists to change one's mind with.

Budget (5%), confidence (95%), and the selection rule were fixed before looking
at any data. Two policies are always evaluated but can never win:
`response_model` (the incumbent) and `random_targeting` (a floor that audits the
estimator - an uninformative policy must collect roughly 5% of the achievable
effect on a 5% budget; it reads 4.2%).

**Candidates.** Seven uplift learners: S-, T-, X-learner (outcome modelling);
CVT, transformed-outcome, R-, DR-learner (label transformation). Six of them
share one LightGBM configuration. The seventh, transformed-outcome, uses ridge,
because at `e = 0.85` its regression target takes only three values and a
flexible learner fits those spikes. CVT needed inverse-propensity reweighting,
because its classical form assumes a balanced design and this one is 85/15.

## Results - Visit

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
not pay: in the runtime table of that report, R- and DR-learner cost 32 to 57
seconds per selection fold on five-fold cross-fitting and land mid-table, while
a ridge regression at 0.23 s nearly matches them.

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

## Results - Conversion

At a 0.29% base rate, negative undersampling was swept over factors 1 to 200 for
T- and CVT-based logistic learners, each with case-control prior correction
([rare_conversion_development.md](outputs/rare_conversion_development.md)).
Selection chose **factor 5**. The top four are the T family at factors 5, 10, 1,
and 25, their intervals all span zero and overlap each other heavily, and every
candidate that undersamples hard lands near the bottom in both families. So the
sweep says light undersampling beats heavy and says nothing about which light
factor to prefer. No candidate reaches a positive lower bound against response
targeting on this sample.

On the separate audit test
([audit_conversion_evaluation.md](outputs/audit_conversion_evaluation.md)) the
selected candidate is level with response targeting at every budget and no
interval excludes zero (at 5%: +0.2 conversions, [-51.5, 51.9]). Response
targeting stays because nothing justifies changing it, not because uplift was
shown to be worse. This sample cannot separate the two.

## Stability

Ten complete repetitions of the protocol, on the same 1,000,000-row sample and
with the same 3-fold out-of-fold selection that chose the champion. The
selection stage is what is being measured, so it has to match: a smaller
selection sample widens every candidate's interval and would show up here as
instability belonging to the sample size rather than to the rule.

| Winning learner | Splits won |
|---|---:|
| S-learner | 8 / 10 |
| DR-learner | 2 / 10 |

Mean difference +163.4 per 200,000 evaluated users (SD 90.3, range +44.2 to
+303.8). All ten point estimates are positive; 3 of 10 intervals are wholly
positive, which is the resolution a fifth of the audit sample buys.

A winner that changes across splits can mean the rule is unstable or that the
candidates are tied, so each run also records how close the call was. The median
gap between first and second place is 99.8 incremental outcomes, against a median
half-width of 443.4 on the winner's own selection interval, roughly a fifth of
the uncertainty in the number being ranked. Every run had at least one candidate
clear zero, a median of six.

**The selection rule does not identify a single best learner**, and the margins
say why: no candidate is measurably better than the one below it. The ordering is
not arbitrary either, since the S-learner never leaves the top two. Gaps inside
the noise and a stable leader are consistent with each other, and together they
say the sample can rank these candidates without separating them. The defensible
claim is therefore about the policy class: uplift targeting at a tight budget
beats response targeting, and the S-learner is the instance that was locked before
the confirmatory sample was drawn. These splits overlap, so they are a sensitivity
analysis rather than ten independent experiments, and they hold the estimator
fixed while they vary the data. The section below varies the estimator instead
and arrives at the same conclusion, which is why the two are read together.

## Base Learners

The stability experiment above varies the data and holds the estimator fixed. It
cannot say whether the champion is a property of the recipe or of the pairing,
because every published run gave all seven candidates the same boosted trees.
[base_learner_comparison.md](outputs/base_learner_comparison.md) varies the
estimator instead, on the same 1,000,000-row sample, the same 3 folds, the same
seed, and the same 5% budget as the run that chose the locked champion.

| Base learner | Winner | Its selection bound | Rank of S-learner | Winner's margin / its own half-width |
|---|---|---:|---:|---:|
| Boosted trees | S-learner | +491.5 | 1 / 7 | 0.57 |
| Linear | T-learner | +386.9 | 6 / 7 | 0.73 |
| Random forest | X-learner | +117.3 | 3 / 7 | 0.31 |

**Three estimators, three different winners**, and the S-learner falls to sixth
of seven under a linear base. That reads worse than it is. In every column the
gap between first and second place is smaller than the half-width of the
winner's own interval, so no column separates its winner from its runner-up, and
the reordering between columns is movement inside the noise rather than evidence
that one estimator suits one recipe. This is the same conclusion the repeated
splits reach along the data axis, arrived at independently along the estimator
axis.

What survives is the policy-class claim: every column's winner clears zero, so
the case for uplift targeting at a tight budget does not rest on having picked
the right learner. What does not survive is any claim that the S-learner is best
of the seven, which was never demonstrated. One ordering is stable across all
three columns, and it is negative: `cvt` finishes last in every one.

The locked result is untouched. This is a development-stage study on the audit
sample, and it opens no confirmatory sample: reference policies and the AIPW
nuisance models stay on boosted trees in every column, so the bar being measured
against and the ruler doing the measuring are the same in all three.

`forest` is the one column that does not reproduce bit for bit - see
[determinism.md](docs/determinism.md) for the measurement and why its ranks are
stable anyway.

## Exploratory Work

Three side studies, kept because they inform the limits above rather than the
headline. Each is a single script and a single report.

- **Ground truth** ([semisynthetic_benchmark.md](outputs/semisynthetic_benchmark.md)) -
  real covariates with a known CATE, so the selection rule can be graded instead
  of trusted. It picked the transformed-outcome learner, which the ground truth
  confirms as the best of the seven (91.7% of oracle value). One draw, so this
  says the rule is not systematically broken, not that it is reliable.
- **Calibration** ([conversion_uplift_calibration.md](outputs/conversion_uplift_calibration.md)) -
  isotonic calibration pulls the selected model's slope toward 1 (0.794 to 0.828)
  and cuts its worst-bin error, but raises mean absolute error slightly. On the
  runner-up it makes every metric worse, turning a slope of 1.015 into 0.800.
  Fitting a monotone correction on ~1,160 conversions gives a result that depends
  on which model it is fitted to, which is the reason not to rely on it.
- **Online design** ([online_experiment_design.md](outputs/online_experiment_design.md)) -
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
- No hyperparameter tuning, one dataset, one time window. Every learner runs
  at one fixed configuration of its estimator, so the comparison is between
  these learners *at those settings* rather than between the learners as
  such. The estimator itself is no longer part of this limit - the base
  learner comparison varies it across three families - but the settings
  inside each family still are.

  What tuning would cost is worth separating into two parts, because they are
  not the same size. The visible part is multiplicity: the selection rule keeps
  the largest lower bound over its candidates, so more candidates means a wider
  Bonferroni correction. That part is cheap. Applying the adjustment this
  repository already reports to the champion's selection figures (+928.0 with a
  standard error of 222.7), the bound holds at +328.9 over 7 candidates, +163.3
  over a 12-point grid for each, and +83.3 over a 48-point grid for each. It
  takes roughly 1,600 candidates before the bound reaches zero.

  The expensive part is the one no correction repairs. Tuning is a *selection*
  step, so it has to sit inside each selection fold, choosing on the fold's
  training rows and never on the rows the fold is scored with. Tuning once over
  the whole development set and only then splitting produces a bound that is
  optimistic no matter how it is adjusted afterwards. That nesting, not the
  correction, is what makes tuning a larger change than it looks.
- The exposure estimate assumes assignment moves the outcome only through the
  ad rendering. Reasonable for display advertising, but not testable here.

## Repository

```text
notebooks/   the analysis end to end: exploration, modelling, evaluation
scripts/     prepare samples, run each experiment, write reports to outputs/
src/data     Criteo loading, sample provenance, undersampling, semi-synthetic
src/models   response baseline, 7 uplift learners, base learner families
src/evaluation  AIPW policy value, uplift curves, calibration, exposure IV
src/experiments honest splitting and the locked protocol
src/serving  the locked policy, saved with what it was measured to do
app.py       Streamlit page over that policy
tests/       114 tests, no data download required
outputs/     tracked evidence: reports, tables, figures
docs/        reproducibility notes
```

## Notebooks

The reasoning behind every decision, in the order it was made, with the charts
and the intermediate numbers visible.

| Notebook | Covers |
|---|---|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | Data integrity, the randomization check, why response probability is the wrong objective, the post-treatment column that has to be excluded, and the duplicate rows that dictate how samples are drawn |
| [02_modeling.ipynb](notebooks/02_modeling.ipynb) | What can be trained when the label does not exist, the three-stage split, cross-fitting against cross-validation, seven learners from two families, and the selection rule applied end to end |
| [03_evaluation.ipynb](notebooks/03_evaluation.ipynb) | Why accuracy and AUC do not apply, AIPW checked against a known answer, uplift curves, the confirmatory result, stability across ten repeated splits, the multiplicity adjustment, and the instrumental-variable answer to the exposure question |
| [04_base_learners.ipynb](notebooks/04_base_learners.ipynb) | A meta-learner is a recipe and the estimator under it is a separate choice, three base learner families, what has to stay fixed for the columns to be comparable, and whether the champion survives changing the estimator |

They import from `src/` rather than restating it, so a notebook and the locked
pipeline cannot drift apart. Runs are on development samples so the notebooks
execute in minutes; every headline figure is read back from `outputs/` and
labelled as such.

```powershell
python -m ipykernel install --user --name vinsmart --display-name "Python (vinsmart)"
python -m jupyter lab notebooks
```

## Using the Locked Policy

A score ranks users but carries no units, so the saved policy travels with the
rates the confirmatory test measured. That is what lets a target list arrive
with a number attached, and keeps the number traceable to the sample it came
from.

```powershell
python scripts\fit_campaign_policy.py          # fit once, save to artifacts/
python scripts\score_campaign.py --budget 0.05 # rank users, write the list
python -m streamlit run app.py                 # the same thing with a budget slider
```

Two things it will not do. It does not measure lift on the users passed in: a
list awaiting contact has no outcomes, so the projection is the locked rate
carried across and is labelled as such. And it refuses budgets the test never
evaluated, because interpolating between them would invent a confidence interval
that no sample supports.

## Reproduction

Python 3.11 or 3.12. The source is the Criteo Uplift Prediction Dataset v2.1,
released by Criteo AI Lab and available from
[its dataset page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
under the terms stated there. It is not redistributed here. Place the download
at `data/criteo-uplift-v2.1.csv.gz`; the file the numbers below were produced
from is 311,422,618 bytes with SHA-256
`2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`, so a reader
who gets a different digest is reading a different file and should expect
different numbers.

```powershell
python -m pip install -r requirements.lock.txt

# Build samples, in this order. The first call indexes the source file (~1 min)
# and every later sample is drawn from that index, excluding the rows every
# earlier sample already spent.
python scripts\prepare_criteo.py --sample-size 500000 `
  --sample-path data\processed\criteo_sample_500k.parquet --random-state 42

# Reserves two million rows so the audit and confirmatory samples are drawn
# around them. Nothing is fitted on this file; the conversion development sample
# is drawn last, from what is left, and so cannot be reserved against itself.
python scripts\prepare_criteo.py --sample-size 2000000 `
  --sample-path data\processed\criteo_reserved_2m.parquet --random-state 42

python scripts\prepare_audit_sample.py
python scripts\prepare_audit_sample.py `
  --excluded-paths "data/processed/criteo_sample_500k.parquet,data/processed/criteo_reserved_2m.parquet,data/processed/criteo_audit_1m.parquet" `
  --output-path data\processed\criteo_confirm_4m.parquet `
  --sample-size 4000000 --random-state 20260730 `
  --report-path outputs\confirmatory_sample.md

python scripts\prepare_audit_sample.py `
  --excluded-paths "data/processed/criteo_sample_500k.parquet,data/processed/criteo_audit_1m.parquet,data/processed/criteo_confirm_4m.parquet" `
  --output-path data\processed\criteo_sample_2m.parquet `
  --sample-size 2000000 --random-state 42 `
  --report-path outputs\conversion_development_sample.md

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

# Repeat the whole protocol ten times. The sample and the selection folds match
# the run above on purpose: a cheaper selection stage would measure sensitivity
# to sample size rather than to the split.
python scripts\run_honest_stability.py `
  --sample-path data/processed/criteo_audit_1m.parquet --selection-folds 3 `
  --models "response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner"

# Rebuild every candidate on three different estimators. Sample, folds, and
# seed match the selection run above. Before comparing anything the script
# reruns the locked configuration and checks it row for row against the table
# the run above wrote, so a column can only be read once that anchor matches.
python scripts\run_base_learner_comparison.py

python scripts\run_exposure_iv.py

# Measure the population and every sample, including which pairs share rows.
python scripts\report_sample_provenance.py

# Conversion: sweep the undersampling factor, then evaluate the one it chose.
# The second command is locked to that factor, so run the sweep first and read
# its champion rather than copying the factor below.
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
  --undersampling-factors 5 --undersampling-families t --random-state 777 `
  --report-path outputs\audit_conversion_evaluation.md `
  --validation-path outputs\tables\audit_conversion_selection.csv `
  --test-path outputs\tables\audit_conversion_test.csv `
  --contrast-path outputs\tables\audit_conversion_contrasts.csv `
  --figure-path outputs\figures\audit_conversion_policy_value.png

python scripts\analyze_uplift_calibration.py `
  --models "undersampled_t_lr_k1,undersampled_t_lr_k5" `
  --undersampling-factors "1,5" --undersampling-families t
python scripts\run_semisynthetic_benchmark.py
python scripts\design_online_experiment.py `
  --input-path outputs\tables\confirmatory_visit_test.csv `
  --policy-a s_learner --policy-b response_model `
  --budget-pct 5.0 --no-campaign-rate 0.038333
```

The protocol block at the top of every report in `outputs/` records the exact
settings that produced it, and each script's `--help` lists its arguments.

## Quality Gates

```powershell
ruff check .
pytest tests
```

Both run in CI on Python 3.11 and 3.12. No test downloads Criteo data. The lint
covers the notebooks as well, so the code in them is held to the same standard
as the code they import.

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
