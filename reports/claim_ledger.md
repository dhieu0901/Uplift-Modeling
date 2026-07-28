# Research Claim Ledger

This ledger separates observed estimates from decisions and prevents
exploratory results from being promoted to confirmatory claims.

| Claim | Evidence | Status | Approved wording |
|---|---|---|---|
| S-learner has a favorable visit point estimate at 5% | Audit difference +168.5, 95% CI [-53.0, 390.0] | Inconclusive | “The audit estimate favors the S-learner” |
| S-learner reliably beats response offline | 9/10 repeated point estimates positive; only 1/10 CIs wholly positive | Not established | Do not call it a proven winner |
| S-learner is ready for production rollout | Audit CI includes zero; no live A/B result | Rejected | “Eligible as an online challenger” |
| Modified-outcome targeting improves visits by 66.3% | The workflow reused test evidence for selection and reporting | Exploratory | “Historical exploratory estimate” |
| Rare-outcome undersampling beats response for conversion | Audit T k=5 minus response is -47.1 at 5%; significantly negative at 10–30% | Rejected | “Response remains the conversion policy” |
| Isotonic calibration improves rare-uplift magnitude | EUCE fell from 0.000629 to 0.000298 on an independent development holdout | Supported | “Calibration error improved; ranking value was not established” |
| Modified outcome best recovers the chosen semi-synthetic CATE | PEHE 0.0043 and 87.8% oracle value at 5% | Supported under this DGP | State the semi-synthetic scope |
| Cross-validated observed policy selection always finds the true best model | OOF AIPW selected CVT while exact value favored modified outcome | Rejected | “Finite-sample policy selection remains noisy” |
| The campaign is profitable | Production cost and outcome value were not available | Not evaluated | Make no ROI claim |

## Decision Gate

A production claim requires a randomized online A-B comparison with the locked
policy definitions, intention-to-treat analysis, guardrails, and validated unit
economics. Offline evidence alone cannot change this status.
