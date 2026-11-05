# Medvolt.AI — Protein Mutation-Effect Prediction & Candidate Prioritization

## Problem statement

Given 840 experimentally characterised single-point mutations across 28 proteins, predict
two effects of a mutation and use them to prioritise 180 unlabeled candidates for
wet-lab validation:

| Target | Meaning | Direction |
|---|---|---|
| `stability_ddG` | Change in folding free energy (kcal/mol) | **Negative is better** (stabilizing) |
| `functional_score` | Retained activity, bounded | **Higher is better** |

The deliverable is a ranked Top 10 with a calibrated sense of how much each ranking can
be trusted. The engineering problem underneath it is **leakage**: mutations are nested
inside proteins, which are nested inside families, and 32% of the candidate pool comes
from two families (`PF08`, `PF09`) that appear nowhere in training. A model validated on
a random split is scored largely on proteins it has already seen, and would therefore
over-promise on exactly the candidates that matter most.

## Dataset description

| | Rows | Proteins | Families |
|---|---|---|---|
| `labeled_mutations.csv` | 840 | 28 | 7 (PF01–PF07) |
| `candidate_mutations_unlabeled.csv` | 180 | 35 (8 unseen) | 9 (PF08, PF09 unseen) |
| `protein_embeddings.csv` | 36 | 36 | 9 |

Per-mutation features cover the wild-type and mutant residue (hydrophobicity, charge,
volume), the structural microenvironment (conservation, solvent accessibility, distance
to the active site, secondary structure, local confidence) and sequence position. Each
protein also carries a 256-dimensional pretrained embedding, which is **constant within a
protein** — a fact with consequences, quantified in the ablation below.

The raw data contains no missing, infinite or out-of-range values; this is verified on
every run rather than assumed (`src/data_processing.py`, `inspect_anomalies`).

## Setup instructions

Requires **Python 3.11 or newer**. Python 3.10 will not work: the newest scipy that
supports it (1.15.x) ships macOS arm64 wheels whose compiled extensions fail to load
on macOS 26+, and the fix requires Python >= 3.11. See the note in
[requirements.txt](requirements.txt).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # runtime + lint/type/test/report tooling
```

Verify the interpreter before running anything — the pinned versions are what the published
results were produced with, and a different scikit-learn silently changes them:

```bash
python -c "import sys, sklearn, numpy; print(sys.version.split()[0], sklearn.__version__, numpy.__version__)"
# expected: 3.11.15 1.9.1 2.4.6
```

## How to run the code

Everything, end to end — this covers every assignment task except the EDA notebook:

```bash
python main.py --config config/config.yaml
```

Measured 126 s on an M-series Mac with the Homebrew/PyPI BLAS: `train` 34 s
(90 pipelines), `ablation` 32 s (70 refits), `train_torch_v2` 42 s, everything
else under 11 s. Nothing is cached between runs, so a re-run costs the same and
produces the same numbers.

### Assignment tasks, and the command for each

| # | Task | Command | Primary output |
|---|------|---------|----------------|
| 1 | Exploratory data analysis | `jupyter lab notebooks/exploratory_analysis.ipynb` | the notebook itself (outputs committed) |
| 2 | Preprocessing & feature engineering | `python -m src.data_processing --config config/config.yaml` | `data/processed/` |
| 3 | Baseline models & benchmark | `python -m src.train --config config/config.yaml` | `results/metrics/baseline_benchmark.md` |
| 4 | Neural networks | `python -m src.train_torch --config config/config.yaml`<br>`python -m src.train_torch_v2 --config config/config.yaml` | `results/metrics/torch_vs_baselines.md`<br>`results/metrics/torch_v2_vs_baselines.md` |
| 5 | Evaluation & comparative analysis | `python -m src.evaluate --config config/config.yaml` | `results/metrics/model_comparison.csv`, `evaluation_summary.{md,json}` |
| 6 | Ablation study | `python -m src.ablation --config config/config.yaml` | `results/metrics/ablation_summary.md`, `results/figures/ablation_performance_drop.png` |
| 7 | Model interpretation & attribution | `python -m src.attribution --config config/config.yaml` | `results/metrics/feature_importances.csv`, `results/figures/feature_importance.png` |
| 8 | Candidate prioritization | `python -m src.predict --config config/config.yaml` | `results/mutation_rankings/top_10_candidates.{csv,md}` |
| 9 | Per-candidate interpretation | `python -m src.interpret --config config/config.yaml` | `results/mutation_rankings/candidate_interpretation.md` |

Stages depend on the artifacts of the stages above them, so run them in that
order or just use `python main.py`. Re-executing the EDA notebook headless:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/exploratory_analysis.ipynb
```

Selected stages, always executed in dependency order:

```bash
python main.py --list                 # show the stage names
python main.py --stages train evaluate
python main.py --from evaluate        # this stage and everything after it
```

The ten stages, in dependency order, are:

```text
data_processing -> validation -> train -> train_torch -> train_torch_v2
                -> evaluate -> predict -> ablation -> attribution -> interpret
```

`validation` has no assignment task of its own: it writes the CV fold report and
the leakage audit that every later stage relies on
(`python -m src.validation --config config/config.yaml`).

Checks:

```bash
python -m pytest -q        # 786 tests (test_api.py skips without the web deps)
python -m ruff check .
python -m mypy src main.py
```

## Model architecture

Every model is a single scikit-learn `Pipeline`, built in [src/model.py](src/model.py):

```text
ColumnTransformer
  ├─ tabular   : impute → scale                 (22 engineered + raw features)
  └─ embedding : impute → scale → PCA(16)       (256-d protein embedding)
→ regressor   (ridge | random_forest | gradient_boosting)
```

Keeping imputation, scaling and PCA **inside** the pipeline is what makes leakage control
structural rather than procedural: `fit` on a training fold is the only place those
statistics are learned, and the validation fold is only ever passed to `predict`. The
tests assert this directly — fitted statistics must equal the training fold's and
*differ* from the whole table's, and corrupting validation rows must leave every fitted
object bit-identical.

Both targets are modelled independently; the model for each is chosen on its own merits.

### Neural networks

Two PyTorch architectures are trained on the **same folds** and the same
fold-local preprocessing, so any difference measures the model rather than the
plumbing around it.

**Multitask network** ([src/torch_model.py](src/torch_model.py)) — one shared
backbone, two task heads:

```text
tabular (22) + PCA embedding (16)  ->  concatenated (38)
  -> [Linear -> BatchNorm -> ReLU -> Dropout] x 3   (256 -> 128 -> 64)
       |- stability head -> Linear -> scalar        (unbounded, kcal/mol)
       `- function head  -> Linear -> Sigmoid       (bounded to 0-1)
```

Sharing a backbone is motivated by both targets depending on the same physical
question — how much the substitution perturbs its local environment — so
gradients from the better-determined target regularise the representation used
by the harder one.

**Dual-branch network** ([src/torch_model_v2.py](src/torch_model_v2.py)) —
the two feature blocks are kept apart until after each is processed:

```text
tabular (22)   -> [Linear -> BN -> ReLU -> Dropout] x 2   (128 -> 64) --,
embedding (256) -> [.. x 1] -> Linear(32) -> ReLU  (learned bottleneck) -+
                                                                        |
                        concat (96) -> fusion Linear(128)
                          -> ResidualBlock x 2  -> two heads (as above)
```

The learned bottleneck **replaces** the fold-local PCA here
(`torch_v2.embedding_pca: false`). PCA keeps the directions of greatest
variance, which in a protein-level vector is largely variance *between*
families — the component that cannot transfer to a held-out family.

Both networks early-stop on an **inner** split carved out of the training fold
and grouped the same way the outer split is. Stopping on the outer validation
fold would select the epoch that scores best on the very rows used to report
performance, so there are two leakage boundaries here, not one. The epoch loop,
seeding and both boundaries live once in
[src/torch_training.py](src/torch_training.py); each architecture supplies only
its preprocessing and its network.

## Validation approach

Three strategies are run on every model and target
([src/validation.py](src/validation.py)):

| Strategy | Group | Question it answers |
|---|---|---|
| `random` | none | Optimistic reference only |
| `protein_group` | `protein_id` | Generalisation to unseen proteins |
| `family_group` | `protein_family` | **Primary** — generalisation to unseen families |

`family_group` is primary because it is the only strategy whose held-out folds resemble
the candidate pool. Metrics are reported both per fold (mean ± std) and **pooled** over
all out-of-fold predictions; pooled is the headline, because family folds are small and
uneven (1–2 families each) so per-fold means are noisy.

The gap between the two is reported explicitly as the **optimism gap**:

| Model | Target | random | family_group | Gap |
|---|---|---|---|---|
| gradient_boosting | `stability_ddG` | 0.650 | 0.556 | **+0.094** |
| gradient_boosting | `functional_score` | 0.817 | 0.747 | **+0.070** |
| ridge | `stability_ddG` | 0.644 | 0.507 | **+0.137** |

(pooled Spearman; full table in `results/metrics/evaluation_summary.md`.)

## Main results

Selected model for both targets: **gradient boosting**, by pooled Spearman under
family-grouped CV.

| Target | MAE | RMSE | R² | Pearson | Spearman |
|---|---|---|---|---|---|
| `stability_ddG` | 0.338 | 0.426 | 0.341 | 0.598 | 0.556 |
| `functional_score` | 0.055 | 0.069 | 0.634 | 0.796 | 0.747 |

All three baselines on both targets under all three splits (18 rows, five metrics each) are in
`results/metrics/baseline_benchmark.md`. Pooled Spearman under `family_group`:

| Model | `stability_ddG` | `functional_score` |
|---|---|---|
| Ridge | 0.507 | 0.708 |
| Random forest | 0.522 | 0.721 |
| **Gradient boosting** | **0.556** | **0.747** |

**Neural networks lose to gradient boosting on both targets.** Same folds, same
preprocessing, pooled Spearman under `family_group`:

| Model | Family | `stability_ddG` | `functional_score` |
|---|---|---|---|
| **Gradient boosting** | boosted trees | **0.556** | **0.747** |
| Random forest | bagged trees | 0.522 | 0.721 |
| Ridge | linear | 0.507 | 0.708 |
| `multitask_nn` | neural net | 0.551 | 0.218 |
| `dual_branch_nn` | neural net | 0.390 | 0.282 |

The multitask network is effectively tied with the best baseline on stability
(0.551 vs 0.556) but collapses on function, where its pooled R² of **−0.677** is
worse than predicting the training mean: the unbounded stability target
dominates the shared gradients, since a 0–1 function target produces a far
smaller MSE. The dual-branch architecture was built to test whether separating
the embedding from the tabular block would help; it recovers function
(R² −0.677 → −0.002) at the cost of stability. Neither beats the trees. On 840
rows and 7 families, that is the expected outcome and is reported as such
rather than tuned until a network wins.

**Attribution agrees with the ablation, by a different mechanism.** Permutation
importance on held-out fold rows — Spearman lost when a column is shuffled,
averaged over the 5 folds:

| Feature group | `stability_ddG` | `functional_score` |
|---|---|---|
| structural | +0.173 | **+0.383** |
| deltas | **+0.178** | +0.248 |
| physicochemical | +0.114 | −0.001 |
| sequence | −0.004 | +0.003 |
| **embeddings (all 256)** | **−0.002** | **+0.000** |

The embedding block — shuffled as one unit, so correlation between its columns
cannot hide its contribution — is indistinguishable from zero, with a
between-fold spread several times its mean. The ablation removes features and
refits; this permutes them in the already-fitted production model. Two
independent routes to the same conclusion. Single strongest features:
`solvent_accessibility` and `delta_hydrophobicity` for stability,
`distance_to_active_site` (+0.314, alone twice the next) for function.

**Ablation — the headline scientific finding.** The 256-d embeddings contribute almost
nothing, and on their own are worse than useless:

| Subset | Features | `stability_ddG` | `functional_score` |
|---|---|---|---|
| `D_full` (baseline) | 22 + 256 | 0.556 | 0.747 |
| `E_all_tabular_no_embeddings` | 22 + 0 | 0.546 (−0.010) | 0.723 (−0.025) |
| `G_full_minus_structural` | 15 + 256 | 0.413 (−0.143) | 0.370 (−0.377) |
| `B_embeddings_only` | 0 + 256 | **−0.205** | **−0.110** |

Because the embedding is a per-protein constant, it cannot distinguish one mutation from
another *within* a protein — which is exactly the ranking being asked for. Structural
features carry the signal. A 22-feature model with no embeddings and no PCA matches the
278-feature model to within 0.01–0.03.

**Uncertainty — a negative result worth stating.** Ensemble spread is *lower* for
unseen-family candidates (0.0969 vs 0.1031 for stability), so the uncertainty penalty
does **not** protect against domain shift. Only the explicit `ood_flag` does.

**Honest caveat on the Top 10.** Only 5 of 180 candidates are predicted stabilizing, so
the stability objective mostly selects the *least destabilizing* mutations; and predicted
function spans only 0.798–0.905 across the Top 10, narrower than the target's own RMSE.
Single-rank ordering within the Top 10 should not be over-read.

### Scientific interpretation: five candidates in detail

Pool context for the feature values below: median conservation 0.439, median solvent
accessibility 0.551, median distance to active site 15.8 Å, median Δvolume +7.7 Å³.
Full per-candidate write-ups: `results/mutation_rankings/candidate_interpretation.md`.

**1 · CAND0162 — Y12F (PF03, coil) · rank 2 · the one prediction worth acting on**
ddG **−0.149 ± 0.038**, function 0.800 ± 0.013. This is the **only shortlisted candidate whose
stabilizing prediction is resolved** — the effect is ~4× its own ensemble spread. Y→F deletes a
single hydroxyl while keeping the aromatic ring, so it is near-isosteric (Δvolume **−3.7 Å³**,
against a pool median of +7.7). At SASA **0.940** — nearly fully solvent-exposed — removing a
polar hydroxyl raises local hydrophobicity without disturbing any packed core. Conservation is
low (0.186 vs 0.439 median), so the position is not under strong evolutionary constraint.
*Concern:* function (0.800) is the second-lowest in the shortlist; under equal weighting that
costs it rank 1. Weight the campaign toward stability and this becomes the lead candidate.

**2 · CAND0148 — S490L (PF06, helix) · rank 1 · ranked first, but the sign is not resolved**
ddG **−0.013 ± 0.085**, function 0.861 ± 0.027. The predicted effect is roughly **six times
smaller than its own uncertainty**, so the honest statement is "approximately neutral", *not*
"stabilizing". The model likely keys on low conservation (0.333) and high exposure (0.698), the
pattern it has learned to reward. *Concern:* the physicochemistry argues the other way — this is
a polar→hydrophobic swap (Δhydrophobicity **+4.6**) with a **+77.7 Å³** volume increase on an
*exposed* helix face, where burying a large hydrophobic residue is entropically unfavourable.
Mild destabilization is the more expected outcome. **This is the entry most in need of
measurement rather than trust**, and the clearest case of the model generalising a heuristic past
where it applies.

**3 · CAND0126 — Q360F (PF06, coil) · rank 5 · the least certain prediction in the list**
ddG **−0.019 ± 0.156**, function 0.798 ± 0.012. The stability uncertainty is the **largest of any
shortlisted candidate and roughly 8× the predicted effect** — the five fold models disagree
substantially. Δhydrophobicity +6.3 and Δvolume +46.1 Å³ at SASA 0.891 repeat the CAND0148
pattern. *Concern:* it shares protein **PROT022** with rank 1, so ranks 1 and 5 are not
independent evidence — a systematic error in that protein's features would move both. Treat them
as one bet, not two.

**4 · CAND0002 — M496W (PF02, β-strand) · rank 6 · highest raw score, fails the gate**
ddG **+0.235 ± 0.040** (destabilizing), function **0.878 ± 0.013**. This has the **highest
priority score in the entire pool (0.885)** yet ranks 6th, because the viability gate sorts ahead
of the score. It is a pure functional-retention pick: low conservation (0.232), 27.5 Å from the
active site. The bulky indole of Met→Trp (**+64.9 Å³**) in a β-strand is the obvious source of
the destabilization, and the tight ensemble (±0.040) means the folds agree it is destabilizing.
*Interpretation:* the model is confidently right that this will fold worse. Useful only if the
campaign tolerates a stability cost for functional retention.

**5 · CAND0091 — Y69W (PF08, coil) · rank 14 · the most instructive failure**
ddG +0.256 ± 0.074, function **0.870 ± 0.004**. The best-ranked candidate from an **unseen family**
(PF08 has zero labeled examples). Note the function uncertainty: **±0.004, the tightest of every
candidate discussed here** — on the one mutation where the model has least right to be confident.
*This is the single most important scientific concern in the whole system.* Fold disagreement
measures whether the five models agree, not whether they are jointly wrong; on a novel family they
agree precisely because they share the same blind spot. The quantitative version is in the main
results: mean spread is *lower* for unseen-family candidates (0.0969) than in-distribution ones
(0.1031). **Only the explicit `ood_flag` protects against this — never the uncertainty number.**

**What this demonstrates.** The model is not a black box here: each ranking decomposes into
named features whose values can be checked against biophysical expectation, and in two of five
cases (CAND0148, CAND0091) that check surfaces a concern the score alone would have hidden. The
defensible output is "these five clear an absolute bar, and here is what to distrust about each",
not a ranked list to be taken on faith.

Full analysis: [report/final_report.pdf](report/final_report.pdf) (6 pages), with the
extended version in [report/final_report_full.md](report/final_report_full.md).

## Repository structure

```text
config/config.yaml              Run setup and decisions: paths, seed, CV, models, ranking
data/raw/                       Original CSVs (read-only inputs)
data/processed/                 Engineered train / candidate tables + feature manifest
notebooks/
  exploratory_analysis.ipynb    EDA: completeness, duplicates, distributions, outliers,
                                mutation types, correlations, data-quality findings
src/common/                     constants.py (dataset facts, naming), config.py, io.py
src/data_processing.py          Ingestion, merge with embeddings, anomaly checks, features
src/model.py                    Model architectures: feature blocks, pipelines, registry
src/validation.py               Random vs. protein-group vs. family-group CV
src/train.py                    Fold-wise training, model selection, artifacts
src/evaluate.py                 Metrics, family/strategy comparison, optimism gap, reports
src/figures.py                  Every figure, as Figure subclasses with a shared renderer
src/api.py                      HTTP service over the trained model (FastAPI)
src/torch_model.py              Multitask network: shared backbone, two task heads
src/torch_training.py           FoldTrainer: the epoch loop and leakage rules, shared
src/train_torch.py              MultitaskTrainer: one input block, plus artifacts
src/torch_model_v2.py           Dual-branch network: separate branches, residual fusion
src/train_torch_v2.py           DualBranchTrainer: two input blocks, plus artifacts
src/predict.py                  Ensemble inference, uncertainty, viability gate, OOD flags, Top 10
src/ablation.py                 Feature-subset ablation study and attribution figure
src/attribution.py              Permutation and impurity feature importance (Task 7)
src/interpret.py                Per-candidate scientific interpretation
tests/                          Mirrors src/, one test module per source module
results/metrics/                CV metrics, OOF predictions, baseline benchmark, model
                                comparison (CSV + JSON), ablation, feature importances
results/figures/                Fit and residual plots, strategy comparison, learning
                                curves, baseline-vs-network comparison
results/mutation_rankings/      Scored candidates, Top 10, candidate interpretation
results/models/                 Fitted pipelines, network weights, provenance metadata
report/final_report.md          Source for the 6-page technical report
report/final_report.pdf         The report deliverable (built by report/build_pdf.py)
report/final_report_full.md     Extended version: full derivations, all candidate write-ups
tests/                          786 tests, mirroring src/
main.py                         End-to-end orchestration
docs/CODING_GUIDELINES.md       Standards every change must follow
docs/api/example_request.json   Ready-to-paste request body for Postman / curl
```

Every path, hyperparameter, threshold and seed above lives in
`config/config.yaml`; no run-affecting value is written into a module. Changing
which models are fitted, which CV strategies run, the viability thresholds or
the Top-K size is a config edit, not a code edit.

Contributors: read [docs/CODING_GUIDELINES.md](docs/CODING_GUIDELINES.md) first. Section 4
defines what goes in `config.yaml` versus `src/common/constants.py`.

## Research extension

Given full sequences, AlphaFold structures, MSAs, MD trajectories, larger experimental datasets
and GPUs, the following is the order I would actually work in. The ordering is not a wish list —
each step is justified by a measurement from **this** project, and the first one addresses the
bottleneck everything else is currently blocked behind.

**1 · Replace the per-protein embedding with residue-level representations.** This is the single
highest-value change, and the evidence is unusually clean: the 256-d embedding contributes
+0.010/+0.025 Spearman in ablation, scores *negative* rank correlation alone, and moves the
fitted model by −0.002/+0.000 under permutation. Two independent methods agree it carries almost
nothing. The cause is structural, not a tuning failure — **the vector is constant within a
protein**, so it cannot distinguish two mutations in the same protein, which is exactly the task.
Per-*residue* embeddings from an ESM-style model fix this by construction. The cheapest version
needs no fine-tuning at all: the wild-type-versus-mutant **log-odds at the mutated position** and
the **pseudo-log-likelihood difference** are strong zero-shot mutation-effect predictors
precisely because they vary per mutation. I would add those as features and re-run the ablation
before training anything larger — if they do not move the grouped-CV number, nothing downstream
will.

**2 · Fine-tune the foundation model, but only with grouped splits and a frozen trunk first.**
With 840 labels and 7 families, full fine-tuning of a 650M-parameter model will memorise protein
identity — the same failure the embedding already exhibits, with more capacity to do it. The
sequence is: frozen trunk + light head → LoRA/adapter tuning → full fine-tuning only if family-
grouped performance keeps improving. **Family-grouped CV is the stopping criterion**, not a
random-split validation loss, because this project measured that gap at up to +0.137 Spearman.

**3 · Structure-aware modelling: GNN, then 3D-equivariant.** The structural block dominates both
the ablation (−0.143/−0.377 when removed) and the attribution (largest permutation group for
function, with `distance_to_active_site` alone at +0.314). That is the strongest signal in the
data, and it is currently represented by four hand-computed scalars. With AlphaFold structures I
would build a **graph neural network over the residue contact graph**, mutated residue as query
node, k-nearest-neighbour edges with distance and relative-orientation features — this captures
packing and local environment that `solvent_accessibility` and `distance_to_active_site` only
approximate. **SE(3)-equivariant** layers (EGNN, or a tensor-field network) are the principled
next step, since ΔΔG is invariant to rotation and translation of the structure and an equivariant
model gets that for free instead of learning it from data it does not have. The existing
dual-branch architecture is already the right shape: **replace the embedding branch with a
structural encoder** and keep the fusion trunk.

**4 · Multimodal sequence–structure fusion, with the dual-branch lesson applied.** The natural
architecture is three branches — residue-level sequence embedding, structural graph, tabular
physicochemistry — fused with cross-attention. The lesson from this project's dual-branch
experiment is specific and worth carrying forward: **separating the branches was necessary but
not sufficient**. A controlled 2×2 showed the architecture change only helped once the loss
weighting was corrected too, because an unbounded target silently dominated the shared gradients.
Any multimodal model needs per-task loss scaling (uncertainty weighting, or GradNorm) from the
start, and each branch ablated independently — otherwise a modality that contributes nothing is
indistinguishable from one that is being drowned out.

**5 · Evolutionary information from MSAs.** Conservation is currently a single scalar and is
among the top features. An MSA gives far more: per-position **PSSM columns**, and
**coupling-derived** terms from a Potts/DCA model that capture epistasis a single-position score
cannot. Mutating a residue that co-varies with a distant contact is a different proposition from
mutating an independently varying one. Caution: MSA depth varies per protein, so depth must be
carried as a feature and a shallow alignment must not be read as "unconstrained".

**6 · MD-derived dynamic features.** Static structure gives one conformation; ΔΔG and functional
tolerance depend on the ensemble. From trajectories I would extract per-residue **RMSF**,
solvent-accessibility *variance* rather than a single value, hydrogen-bond occupancy, and
**local-unfolding frequency** at the mutated position. The scientific bet is specific: the
stability target is currently the weaker of the two (R² 0.34 vs 0.63), and rigidity/flexibility
is exactly the information a static snapshot omits. MD is expensive per protein, so I would first
validate the hypothesis on a subset before computing trajectories at scale.

**7 · Transfer and contrastive learning.** ProTherm/FireProt and ProteinGym are far larger than
840 rows; pre-train on them and fine-tune on the target assay, with the caveat that experimental
ΔΔG conventions and buffer conditions differ between sources and must be modelled as a batch
effect rather than pooled naively. **Contrastive pre-training** is attractive here for a reason
specific to this problem: training the encoder to place *mutations of the same protein* far apart
when their effects differ directly attacks the per-protein-constant failure mode, rather than
patching around it.

**8 · Calibrated uncertainty, then active learning — in that order.** This project produced a
clear negative result: ensemble spread is *lower* for out-of-distribution candidates (0.0969 vs
0.1031), so fold disagreement is **blind to domain shift**. Deep ensembles from different
initialisations, evidential regression, or **conformal prediction** (which gives coverage
guarantees rather than a bare spread) would replace it, plus a continuous distance-to-training
-data score in place of the binary `ood_flag`. Only then is active learning safe: an acquisition
function built on an uncertainty measure blind to novelty would systematically **under-explore
exactly the novel families** the campaign most needs to learn about. Getting this order wrong
produces a loop that confidently refines what it already knows.

**9 · Generative design, last and deliberately.** Inverse folding (ProteinMPNN) or a diffusion
model can propose sequences rather than score a fixed list, which removes the candidate-pool
ceiling entirely. I would not start here: a generator is only as good as the oracle scoring it,
and the current oracle cannot resolve stability differences below ~0.4 kcal/mol and is
overconfident off-distribution. Generation under that oracle would produce fluent, confident,
unverifiable proposals. Steps 1–8 build the oracle; step 9 is what it is for.

**What I would keep unchanged.** The grouped cross-validation, the leakage assertions, the
ablation harness and the config-driven reproducibility are not scaffolding to be discarded once
better models arrive — they are what makes a better model's improvement *believable*. Every
architecture above should be reported under family-grouped CV with the optimism gap stated, and
every new feature block should pass through the same ablation before it is trusted.

## HTTP API

The trained model is also callable over HTTP, for Postman, curl or any client.
This needs three dependencies the batch pipeline does not, so `src/api.py` is
the one module that will not import until they are installed — and
`tests/test_api.py` skips rather than fails in that state:

```bash
pip install fastapi==0.162.0 uvicorn==0.54.0 pydantic==2.14.2 httpx==0.28.1
# (all four are already pinned in requirements.txt / requirements-dev.txt)

python -m src.api --config config/config.yaml
```

| method | path | purpose |
|:-------|:-----|:--------|
| GET  | `/health`         | liveness + whether the model loaded |
| GET  | `/model`          | selected models, CV strategy, seed, versions |
| POST | `/predict`        | score mutations: predictions, uncertainty, OOD, viability |
| POST | `/prioritize`     | score and rank, returning the Top-K with justifications |
| GET  | `/metrics/{name}` | a published metrics table as JSON |
| GET  | `/figures/{key}`  | a generated PNG |
| GET  | `/docs`           | interactive schema |

`http://127.0.0.1:8000/openapi.json` imports into Postman as a complete
collection. A ready-made request body is in
[docs/api/example_request.json](docs/api/example_request.json).

**Training is not exposed.** A request that blocks for minutes and mutates
shared artifacts is a batch job with an HTTP trigger, not an API; two
concurrent callers would corrupt each other's run. Training stays
`python main.py`, and the service reads what it produced.

## Reproducibility

Random seed: `20261102`, set once in `config/config.yaml` and threaded through every
splitter, estimator and PCA.

Results are reproducible for a fixed environment, but **not across library versions**.
Re-running any stage with the pinned versions reproduces every reported metric, the selected
models and the Top 10 exactly; the raw out-of-fold predictions agree to floating-point tolerance
(measured maximum difference 6.7e-16) rather than byte-for-byte, because threaded BLAS does not
fix summation order. Changing
the scikit-learn pin shifts predictions slightly and reorders the Top 10 (overlap stays
9/10). Model files in `results/models/` are joblib pickles and cannot be loaded by a
different scikit-learn version, so changing that pin requires re-running training before
prediction. The exact versions, seed and feature lists used are recorded in
`results/models/training_metadata.json`.
