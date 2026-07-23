# Validation Against scikit-uplift

## Configuration

- Dataset: Criteo 500k sample.
- Outcome: `visit`.
- Seed: `42`.
- Local S-learner compared with `SoloModel(method="dummy")`.
- Local T-learner compared with `TwoModels(method="vanilla")`.
- Uplift and Qini curves compared with `sklift.metrics`.

## Prediction Agreement

| Model | Pearson correlation | Spearman correlation | MAE | Maximum absolute difference |
|---|---:|---:|---:|---:|
| S-learner | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| T-learner | 1.000000 | 1.000000 | 0.000000 | 0.000000 |

## Metric Agreement

| Model | Implementation | Criteo relative AUUC | scikit-uplift AUUC | Qini AUC |
|---|---|---:|---:|---:|
| S-learner | Local | 0.007201 | 0.018163 | 0.046019 |
| S-learner | scikit-uplift | 0.007201 | 0.018163 | 0.046019 |
| T-learner | Local | 0.006437 | 0.011144 | 0.028181 |
| T-learner | scikit-uplift | 0.006437 | 0.011144 | 0.028181 |

The Criteo and scikit-uplift AUUC columns use different normalizations and should not be compared with each other. Agreement is assessed within each metric.

## Result

Predictions, rankings, and all uplift/Qini curve points match, including ties. This validates the S/T-learners and curve calculations, not their business value.
