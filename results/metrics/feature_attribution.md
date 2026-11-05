# Feature attribution

- Model explained: the one Phase 3 selected per target, as fitted inside each of the `family_group` folds; nothing is refit here
- Permutation importance: `spearman` lost when a column is shuffled on that fold's **validation** rows, 5 shuffles per feature, averaged over folds
- Embeddings permuted as a single block: `True`

Larger is more important for every metric: the sign is normalised so that a positive value always means the model got worse when the feature was shuffled. Small negative values are noise around zero, which is what an unused feature looks like.

`importance_std` is the spread across folds. Read it before trusting a rank: on seven families a feature whose between-fold spread exceeds its mean is not a stable finding.

## By feature group

### stability_ddG

| importance_kind   | feature_group   |   importance_mean |   importance_share |   n_features |
|:------------------|:----------------|------------------:|-------------------:|-------------:|
| impurity          | structural      |             0.343 |              0.343 |            7 |
| impurity          | deltas          |             0.245 |              0.245 |            6 |
| impurity          | physicochemical |             0.176 |              0.176 |            6 |
| impurity          | unassigned      |             0.175 |              0.175 |           16 |
| impurity          | sequence        |             0.061 |              0.061 |            3 |
| permutation       | deltas          |             0.178 |              0.378 |            6 |
| permutation       | structural      |             0.173 |              0.369 |            7 |
| permutation       | physicochemical |             0.114 |              0.251 |            6 |
| permutation       | embeddings      |            -0.002 |              0.000 |            1 |
| permutation       | sequence        |            -0.004 |              0.002 |            3 |

### functional_score

| importance_kind   | feature_group   |   importance_mean |   importance_share |   n_features |
|:------------------|:----------------|------------------:|-------------------:|-------------:|
| impurity          | structural      |             0.576 |              0.576 |            7 |
| impurity          | deltas          |             0.242 |              0.242 |            6 |
| impurity          | unassigned      |             0.121 |              0.121 |           16 |
| impurity          | sequence        |             0.032 |              0.032 |            3 |
| impurity          | physicochemical |             0.029 |              0.029 |            6 |
| permutation       | structural      |             0.383 |              0.599 |            7 |
| permutation       | deltas          |             0.248 |              0.391 |            6 |
| permutation       | sequence        |             0.003 |              0.008 |            3 |
| permutation       | embeddings      |             0.000 |              0.001 |            1 |
| permutation       | physicochemical |            -0.001 |              0.002 |            6 |

## Top 20 features

### stability_ddG — permutation

| feature                  | feature_group   |   importance_mean |   importance_std |   importance_share |   n_folds |   rank |
|:-------------------------|:----------------|------------------:|-----------------:|-------------------:|----------:|-------:|
| solvent_accessibility    | structural      |             0.108 |            0.031 |              0.228 |         5 |      1 |
| delta_hydrophobicity     | deltas          |             0.107 |            0.033 |              0.226 |         5 |      2 |
| mutant_hydrophobicity    | physicochemical |             0.092 |            0.021 |              0.195 |         5 |      3 |
| abs_delta_charge         | deltas          |             0.048 |            0.012 |              0.102 |         5 |      4 |
| conservation_score       | structural      |             0.045 |            0.013 |              0.095 |         5 |      5 |
| mutant_volume            | physicochemical |             0.026 |            0.017 |              0.056 |         5 |      6 |
| abs_delta_volume         | deltas          |             0.016 |            0.007 |              0.034 |         5 |      7 |
| ss_H                     | structural      |             0.012 |            0.006 |              0.025 |         5 |      8 |
| local_confidence_score   | structural      |             0.008 |            0.005 |              0.017 |         5 |      9 |
| abs_delta_hydrophobicity | deltas          |             0.005 |            0.002 |              0.011 |         5 |     10 |
| delta_charge             | deltas          |             0.002 |            0.005 |              0.005 |         5 |     11 |
| relative_position        | sequence        |             0.001 |            0.004 |              0.002 |         5 |     12 |
| ss_E                     | structural      |             0.001 |            0.001 |              0.002 |         5 |     13 |
| ss_C                     | structural      |             0.001 |            0.001 |              0.002 |         5 |     14 |
| wt_charge                | physicochemical |             0.000 |            0.002 |              0.000 |         5 |     15 |
| delta_volume             | deltas          |            -0.001 |            0.005 |              0.000 |         5 |     16 |
| wt_hydrophobicity        | physicochemical |            -0.001 |            0.004 |              0.000 |         5 |     17 |
| distance_to_active_site  | structural      |            -0.001 |            0.004 |              0.000 |         5 |     18 |
| mutant_charge            | physicochemical |            -0.001 |            0.002 |              0.000 |         5 |     19 |
| embeddings               | embeddings      |            -0.002 |            0.010 |              0.000 |         5 |     20 |

### stability_ddG — impurity

| feature                  | feature_group   |   importance_mean |   importance_std |   importance_share |   n_folds |   rank |
|:-------------------------|:----------------|------------------:|-----------------:|-------------------:|----------:|-------:|
| solvent_accessibility    | structural      |             0.148 |            0.006 |              0.148 |         5 |      1 |
| delta_hydrophobicity     | deltas          |             0.122 |            0.007 |              0.122 |         5 |      2 |
| mutant_hydrophobicity    | physicochemical |             0.106 |            0.012 |              0.106 |         5 |      3 |
| conservation_score       | structural      |             0.089 |            0.008 |              0.089 |         5 |      4 |
| local_confidence_score   | structural      |             0.054 |            0.007 |              0.054 |         5 |      5 |
| embedding_pc3            | unassigned      |             0.037 |            0.029 |              0.037 |         5 |      6 |
| mutant_volume            | physicochemical |             0.037 |            0.011 |              0.037 |         5 |      7 |
| abs_delta_charge         | deltas          |             0.036 |            0.010 |              0.036 |         5 |      8 |
| abs_delta_volume         | deltas          |             0.035 |            0.001 |              0.035 |         5 |      9 |
| relative_position        | sequence        |             0.031 |            0.004 |              0.031 |         5 |     10 |
| embedding_pc4            | unassigned      |             0.026 |            0.031 |              0.026 |         5 |     11 |
| distance_to_active_site  | structural      |             0.024 |            0.003 |              0.024 |         5 |     12 |
| position                 | sequence        |             0.024 |            0.002 |              0.024 |         5 |     13 |
| ss_H                     | structural      |             0.023 |            0.009 |              0.023 |         5 |     14 |
| delta_volume             | deltas          |             0.022 |            0.006 |              0.022 |         5 |     15 |
| abs_delta_hydrophobicity | deltas          |             0.021 |            0.005 |              0.021 |         5 |     16 |
| embedding_pc0            | unassigned      |             0.021 |            0.017 |              0.021 |         5 |     17 |
| wt_volume                | physicochemical |             0.016 |            0.005 |              0.016 |         5 |     18 |
| embedding_pc2            | unassigned      |             0.014 |            0.007 |              0.014 |         5 |     19 |
| wt_hydrophobicity        | physicochemical |             0.013 |            0.003 |              0.013 |         5 |     20 |

### functional_score — permutation

| feature                  | feature_group   |   importance_mean |   importance_std |   importance_share |   n_folds |   rank |
|:-------------------------|:----------------|------------------:|-----------------:|-------------------:|----------:|-------:|
| distance_to_active_site  | structural      |             0.314 |            0.059 |              0.492 |         5 |      1 |
| abs_delta_charge         | deltas          |             0.174 |            0.053 |              0.272 |         5 |      2 |
| abs_delta_hydrophobicity | deltas          |             0.067 |            0.021 |              0.104 |         5 |      3 |
| conservation_score       | structural      |             0.036 |            0.008 |              0.056 |         5 |      4 |
| local_confidence_score   | structural      |             0.020 |            0.009 |              0.032 |         5 |      5 |
| solvent_accessibility    | structural      |             0.011 |            0.004 |              0.017 |         5 |      6 |
| abs_delta_volume         | deltas          |             0.008 |            0.015 |              0.013 |         5 |      7 |
| position                 | sequence        |             0.005 |            0.004 |              0.007 |         5 |      8 |
| ss_H                     | structural      |             0.002 |            0.001 |              0.002 |         5 |      9 |
| mutant_volume            | physicochemical |             0.001 |            0.003 |              0.001 |         5 |     10 |
| delta_volume             | deltas          |             0.001 |            0.006 |              0.001 |         5 |     11 |
| mutant_hydrophobicity    | physicochemical |             0.001 |            0.004 |              0.001 |         5 |     12 |
| embeddings               | embeddings      |             0.000 |            0.012 |              0.001 |         5 |     13 |
| relative_position        | sequence        |             0.000 |            0.003 |              0.000 |         5 |     14 |
| delta_charge             | deltas          |             0.000 |            0.001 |              0.000 |         5 |     15 |
| ss_E                     | structural      |             0.000 |            0.000 |              0.000 |         5 |     16 |
| ss_C                     | structural      |            -0.000 |            0.000 |              0.000 |         5 |     17 |
| wt_volume                | physicochemical |            -0.000 |            0.003 |              0.000 |         5 |     18 |
| mutant_charge            | physicochemical |            -0.000 |            0.000 |              0.000 |         5 |     19 |
| wt_hydrophobicity        | physicochemical |            -0.001 |            0.004 |              0.000 |         5 |     20 |

### functional_score — impurity

| feature                  | feature_group   |   importance_mean |   importance_std |   importance_share |   n_folds |   rank |
|:-------------------------|:----------------|------------------:|-----------------:|-------------------:|----------:|-------:|
| distance_to_active_site  | structural      |             0.413 |            0.020 |              0.413 |         5 |      1 |
| abs_delta_charge         | deltas          |             0.117 |            0.009 |              0.117 |         5 |      2 |
| conservation_score       | structural      |             0.111 |            0.016 |              0.111 |         5 |      3 |
| abs_delta_hydrophobicity | deltas          |             0.057 |            0.005 |              0.057 |         5 |      4 |
| embedding_pc0            | unassigned      |             0.033 |            0.017 |              0.033 |         5 |      5 |
| abs_delta_volume         | deltas          |             0.033 |            0.002 |              0.033 |         5 |      6 |
| local_confidence_score   | structural      |             0.027 |            0.002 |              0.027 |         5 |      7 |
| solvent_accessibility    | structural      |             0.022 |            0.003 |              0.022 |         5 |      8 |
| delta_volume             | deltas          |             0.017 |            0.002 |              0.017 |         5 |      9 |
| embedding_pc2            | unassigned      |             0.015 |            0.001 |              0.015 |         5 |     10 |
| delta_hydrophobicity     | deltas          |             0.015 |            0.002 |              0.015 |         5 |     11 |
| position                 | sequence        |             0.014 |            0.002 |              0.014 |         5 |     12 |
| relative_position        | sequence        |             0.011 |            0.002 |              0.011 |         5 |     13 |
| embedding_pc4            | unassigned      |             0.010 |            0.006 |              0.010 |         5 |     14 |
| mutant_hydrophobicity    | physicochemical |             0.008 |            0.002 |              0.008 |         5 |     15 |
| wt_volume                | physicochemical |             0.008 |            0.001 |              0.008 |         5 |     16 |
| embedding_pc1            | unassigned      |             0.008 |            0.006 |              0.008 |         5 |     17 |
| mutant_volume            | physicochemical |             0.007 |            0.001 |              0.007 |         5 |     18 |
| protein_length           | sequence        |             0.007 |            0.003 |              0.007 |         5 |     19 |
| embedding_pc8            | unassigned      |             0.006 |            0.005 |              0.006 |         5 |     20 |

