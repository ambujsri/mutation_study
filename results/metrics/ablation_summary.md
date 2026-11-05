# Feature-subset ablation

- Cross-validation: `family_group`, 840 pooled out-of-fold predictions per row
- Baseline for the delta columns: `D_full`
- Models: {'stability_ddG': 'gradient_boosting', 'functional_score': 'gradient_boosting'}

Every subset is re-fitted from scratch inside each training fold, so differences are attributable to the feature columns alone; folds, seed and model are held fixed.

A `delta_vs_baseline_` column is this subset's score minus the baseline's. For `r2` and `spearman` a negative delta is a loss; the sign is left raw rather than normalised so the column can be read straight against the level beside it.

## stability_ddG

| subset                      |   n_tabular |   n_embedding |   pooled_rmse |   pooled_spearman |   pooled_r2 |   pooled_pearson |   delta_vs_baseline_spearman |   delta_vs_baseline_r2 |
|:----------------------------|------------:|--------------:|--------------:|------------------:|------------:|-----------------:|-----------------------------:|-----------------------:|
| A_tabular_no_embeddings     |           9 |             0 |         0.483 |             0.373 |       0.152 |            0.419 |                       -0.183 |                 -0.188 |
| B_embeddings_only           |           0 |           256 |         0.551 |            -0.205 |      -0.103 |           -0.186 |                       -0.762 |                 -0.443 |
| C_tabular_plus_embeddings   |           9 |           256 |         0.476 |             0.397 |       0.177 |            0.446 |                       -0.159 |                 -0.163 |
| D_full                      |          22 |           256 |         0.426 |             0.556 |       0.341 |            0.598 |                        0.000 |                  0.000 |
| E_all_tabular_no_embeddings |          22 |             0 |         0.425 |             0.546 |       0.342 |            0.589 |                       -0.010 |                  0.001 |
| F_full_minus_deltas         |          16 |           256 |         0.428 |             0.536 |       0.335 |            0.587 |                       -0.021 |                 -0.006 |
| G_full_minus_structural     |          15 |           256 |         0.471 |             0.413 |       0.194 |            0.460 |                       -0.143 |                 -0.147 |

## functional_score

| subset                      |   n_tabular |   n_embedding |   pooled_rmse |   pooled_spearman |   pooled_r2 |   pooled_pearson |   delta_vs_baseline_spearman |   delta_vs_baseline_r2 |
|:----------------------------|------------:|--------------:|--------------:|------------------:|------------:|-----------------:|-----------------------------:|-----------------------:|
| A_tabular_no_embeddings     |           9 |             0 |         0.114 |             0.271 |       0.013 |            0.269 |                       -0.477 |                 -0.621 |
| B_embeddings_only           |           0 |           256 |         0.119 |            -0.110 |      -0.067 |           -0.129 |                       -0.857 |                 -0.701 |
| C_tabular_plus_embeddings   |           9 |           256 |         0.111 |             0.327 |       0.074 |            0.300 |                       -0.421 |                 -0.560 |
| D_full                      |          22 |           256 |         0.069 |             0.747 |       0.634 |            0.796 |                        0.000 |                  0.000 |
| E_all_tabular_no_embeddings |          22 |             0 |         0.073 |             0.723 |       0.599 |            0.777 |                       -0.025 |                 -0.035 |
| F_full_minus_deltas         |          16 |           256 |         0.074 |             0.717 |       0.590 |            0.770 |                       -0.030 |                 -0.044 |
| G_full_minus_structural     |          15 |           256 |         0.109 |             0.370 |       0.092 |            0.351 |                       -0.377 |                 -0.542 |

