# Scientific interpretation of top-ranked candidates

Case studies for 4 of the Top 10 ranked mutations, placed against the distribution of all 180 scored candidates. Qualitative bands are the pool's 33%/67% quantiles, and the underlying number is always printed. For signed delta features a lower-third value means the bottom of the pool's range, not a small magnitude.

The physicochemical notes state general expectations from protein biophysics to make the inputs interpretable. They are not feature attributions: the model is a gradient-boosted tree ensemble, and no per-prediction attribution was computed.

### Rank 1 — CAND0148 (PROT022, PF06): S490L

**Substitution.** S to L at position 490, in a `H` secondary-structure element (priority score 0.850, flag `in_distribution`).

**Why the model ranks it here.**
Predicted ddG is -0.013 kcal/mol (stabilizing), with fold-ensemble spread +/-0.085. The magnitude is smaller than the ensemble spread, so the sign is not resolved: treat this as approximately neutral rather than as a confirmed stabilizing mutation. Predicted functional retention is 0.861 on the 0-1 scale (1 = wild-type-like), with fold-ensemble spread +/-0.027. The score balances these two objectives, so a candidate can rank highly on strong predicted function while being only neutral on stability.

**Driving input features.**
- `conservation_score` = +0.333 (lower third for this pool) — evolutionary conservation at the mutated position; highly conserved positions are typically under functional or structural constraint, so substitutions there carry more risk of functional loss.
- `solvent_accessibility` = +0.698 (upper third for this pool) — solvent exposure; buried positions tolerate hydrophobic-to-polar changes poorly, whereas exposed positions are generally more permissive.
- `distance_to_active_site` = +24.675 (upper third for this pool) — distance from the active site; substitutions close to the catalytic centre can disrupt function even when they are thermodynamically tolerated.
- `local_confidence_score` = +73.860 (lower third for this pool) — local structural confidence; low values mean the structural context itself is less certain, which weakens any structure-based reasoning about this position.
- `delta_hydrophobicity` = +4.600 (upper third for this pool) — change in residue hydrophobicity (mutant minus wild type); large shifts in a buried environment tend to perturb core packing.
- `delta_charge` = +0.000 (lower third for this pool) — change in formal residue charge; introducing or removing charge in a buried or active-site environment can disrupt electrostatics and salt bridges.
- `delta_volume` = +77.700 (upper third for this pool) — change in residue volume (A^3); large increases risk steric clashes in packed regions, while large decreases can leave destabilizing cavities.

**Confidence and caveats.**
- Protein `PROT022` is represented in the labeled training data, so this is an interpolation within the training distribution.
- The priority score is rank-normalized **within this 180-candidate pool**, so it ranks this mutation against the others rather than measuring absolute merit.

### Rank 2 — CAND0162 (PROT012, PF03): Y12F

**Substitution.** Y to F at position 12, in a `C` secondary-structure element (priority score 0.833, flag `in_distribution`).

**Why the model ranks it here.**
Predicted ddG is -0.149 kcal/mol (stabilizing), with fold-ensemble spread +/-0.038. Predicted functional retention is 0.800 on the 0-1 scale (1 = wild-type-like), with fold-ensemble spread +/-0.013. The score balances these two objectives, so a candidate can rank highly on strong predicted function while being only neutral on stability.

**Driving input features.**
- `conservation_score` = +0.186 (lower third for this pool) — evolutionary conservation at the mutated position; highly conserved positions are typically under functional or structural constraint, so substitutions there carry more risk of functional loss.
- `solvent_accessibility` = +0.940 (upper third for this pool) — solvent exposure; buried positions tolerate hydrophobic-to-polar changes poorly, whereas exposed positions are generally more permissive.
- `distance_to_active_site` = +16.723 (middle third for this pool) — distance from the active site; substitutions close to the catalytic centre can disrupt function even when they are thermodynamically tolerated.
- `local_confidence_score` = +67.760 (lower third for this pool) — local structural confidence; low values mean the structural context itself is less certain, which weakens any structure-based reasoning about this position.
- `delta_hydrophobicity` = +4.100 (upper third for this pool) — change in residue hydrophobicity (mutant minus wild type); large shifts in a buried environment tend to perturb core packing.
- `delta_charge` = +0.000 (lower third for this pool) — change in formal residue charge; introducing or removing charge in a buried or active-site environment can disrupt electrostatics and salt bridges.
- `delta_volume` = -3.700 (middle third for this pool) — change in residue volume (A^3); large increases risk steric clashes in packed regions, while large decreases can leave destabilizing cavities.

**Confidence and caveats.**
- Protein `PROT012` is represented in the labeled training data, so this is an interpolation within the training distribution.
- The priority score is rank-normalized **within this 180-candidate pool**, so it ranks this mutation against the others rather than measuring absolute merit.

### Rank 3 — CAND0167 (PROT012, PF03): N133S

**Substitution.** N to S at position 133, in a `C` secondary-structure element (priority score 0.750, flag `in_distribution`).

**Why the model ranks it here.**
Predicted ddG is -0.068 kcal/mol (stabilizing), with fold-ensemble spread +/-0.125. The magnitude is smaller than the ensemble spread, so the sign is not resolved: treat this as approximately neutral rather than as a confirmed stabilizing mutation. Predicted functional retention is 0.854 on the 0-1 scale (1 = wild-type-like), with fold-ensemble spread +/-0.013. The score balances these two objectives, so a candidate can rank highly on strong predicted function while being only neutral on stability.

**Driving input features.**
- `conservation_score` = +0.080 (lower third for this pool) — evolutionary conservation at the mutated position; highly conserved positions are typically under functional or structural constraint, so substitutions there carry more risk of functional loss.
- `solvent_accessibility` = +0.572 (middle third for this pool) — solvent exposure; buried positions tolerate hydrophobic-to-polar changes poorly, whereas exposed positions are generally more permissive.
- `distance_to_active_site` = +27.918 (upper third for this pool) — distance from the active site; substitutions close to the catalytic centre can disrupt function even when they are thermodynamically tolerated.
- `local_confidence_score` = +79.190 (middle third for this pool) — local structural confidence; low values mean the structural context itself is less certain, which weakens any structure-based reasoning about this position.
- `delta_hydrophobicity` = +2.700 (upper third for this pool) — change in residue hydrophobicity (mutant minus wild type); large shifts in a buried environment tend to perturb core packing.
- `delta_charge` = +0.000 (lower third for this pool) — change in formal residue charge; introducing or removing charge in a buried or active-site environment can disrupt electrostatics and salt bridges.
- `delta_volume` = -25.100 (middle third for this pool) — change in residue volume (A^3); large increases risk steric clashes in packed regions, while large decreases can leave destabilizing cavities.

**Confidence and caveats.**
- Protein `PROT012` is represented in the labeled training data, so this is an interpolation within the training distribution.
- The priority score is rank-normalized **within this 180-candidate pool**, so it ranks this mutation against the others rather than measuring absolute merit.

### Rank 4 — CAND0015 (PROT009, PF03): E353V

**Substitution.** E to V at position 353, in a `C` secondary-structure element (priority score 0.631, flag `in_distribution`).

**Why the model ranks it here.**
Predicted ddG is -0.076 kcal/mol (stabilizing), with fold-ensemble spread +/-0.089. The magnitude is smaller than the ensemble spread, so the sign is not resolved: treat this as approximately neutral rather than as a confirmed stabilizing mutation. Predicted functional retention is 0.750 on the 0-1 scale (1 = wild-type-like), with fold-ensemble spread +/-0.014. The score balances these two objectives, so a candidate can rank highly on strong predicted function while being only neutral on stability.

**Driving input features.**
- `conservation_score` = +0.080 (lower third for this pool) — evolutionary conservation at the mutated position; highly conserved positions are typically under functional or structural constraint, so substitutions there carry more risk of functional loss.
- `solvent_accessibility` = +0.980 (upper third for this pool) — solvent exposure; buried positions tolerate hydrophobic-to-polar changes poorly, whereas exposed positions are generally more permissive.
- `distance_to_active_site` = +30.185 (upper third for this pool) — distance from the active site; substitutions close to the catalytic centre can disrupt function even when they are thermodynamically tolerated.
- `local_confidence_score` = +64.530 (lower third for this pool) — local structural confidence; low values mean the structural context itself is less certain, which weakens any structure-based reasoning about this position.
- `delta_hydrophobicity` = +7.700 (upper third for this pool) — change in residue hydrophobicity (mutant minus wild type); large shifts in a buried environment tend to perturb core packing.
- `delta_charge` = +1.000 (upper third for this pool) — change in formal residue charge; introducing or removing charge in a buried or active-site environment can disrupt electrostatics and salt bridges.
- `delta_volume` = +1.600 (middle third for this pool) — change in residue volume (A^3); large increases risk steric clashes in packed regions, while large decreases can leave destabilizing cavities.

**Confidence and caveats.**
- Protein `PROT009` is represented in the labeled training data, so this is an interpolation within the training distribution.
- The priority score is rank-normalized **within this 180-candidate pool**, so it ranks this mutation against the others rather than measuring absolute merit.
