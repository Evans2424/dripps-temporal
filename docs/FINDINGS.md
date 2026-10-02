# Findings so far

Status: RQ1 answered on the pooled Portuguese data; RQ2 answered by the pooled
hierarchical model (B6): base rates differ, cue weights are not shown to. Numbers reproduce via
`make all`.

> **Revised after code review.** Earlier numbers in this file were affected by
> four measurement bugs: the interaction terms were permuted without being
> recomputed from their sources, confidence intervals were bootstrapped over CV
> repeats rather than sentences, XGBoost silently trained on library defaults,
> and macro-F1 averaged over a varying label set. Every conclusion survived; the
> numbers below are the corrected ones, transcribed from `results/tables/`. See
> `tests/test_regressions.py`.
>
> **One interval moved enough to matter.** With the bootstrap taken over
> sentence groups rather than CV folds, the connector block's ablation interval
> now includes zero ([−0.001, 0.017], previously [0.006, 0.012]). The cue's rank
> is unchanged; what changed is that it can no longer be called distinguishable
> from no effect. `docs/LINGUISTICS.md` §3.4 gives the reason.

> **Later annotation batches.** Numbers below include two batches annotated after
> the original export: 150 Portuguese sentences (50 each EP, AP, MP; `PT..V`
> IDs) and 95 EP sentences (`PTEUJ`; `docs/ANNOTATION_SCHEME.md`, "Later
> annotation batches"). The Portuguese analysis set is 1,038 clauses, up from 793.
> The conclusions and every rank in the pooled hierarchy are unchanged; the
> scores are lower. B3 falls from 0.736 on the 793 original rows to 0.705 after
> the first extra batch and 0.695 after the second, and EP's own logit from 0.694
> to 0.632. Neither batch was annotated for `SR-SC`, and the second has no
> discourse relation or connector either (every connector cell is "N/A", read as
> *no connector*). Whether the new rows are simply harder or were annotated
> differently cannot be told from the data: there is no agreement figure. The
> only overlap with an earlier annotator is two sentences, which agree on one and
> differ on the temporal relation on the other.
> Scored per batch (out-of-fold predictions of the pooled B3), macro-F1 is 0.723 on the 793
> original rows, 0.602 on the Violeta rows and 0.590 on the Abergaria rows, so the new rows are
> harder for a model trained mostly on the original data, or were annotated differently.
> The one earlier claim that did not survive: *no* non-final Portuguese APC is
> `Post` is now 2 of 80, both in the newest batch.

## Headline

**In Portuguese, the temporal reading of an adverbial perfect participial clause
is carried by the *main* clause, not by the participial clause itself.** Removing
the aspectual class of the subordinate clause — one of the two cues the prior
literature foregrounds — costs the model nothing at all (the point estimate is
slightly negative).

## RQ1 — cue hierarchy (pooled, 1,038 Portuguese clauses)

Primary measure is **refit ablation**: drop a cue block, refit, measure the fall
in out-of-fold macro-F1. Permutation importance is reported as a cross-check.
(Permutation cannot measure the interaction block, whose columns are
deterministic products of the aspect columns.)

| rank | cue block | ablation | 95% CI | permutation |
|---|---|---|---|---|
| 1 | **main-clause aspect** | 0.093 | [0.066, 0.122] | 0.155 |
| 2 | **main-clause tense** | 0.057 | [0.034, 0.081] | 0.124 |
| 3 | clause position | 0.031 | [0.013, 0.050] | 0.064 |
| 4 | connector | 0.008 | [−0.001, 0.018] | 0.020 |
| 5 | telicity/durativity interactions | −0.004 | [−0.014, 0.006] | *n/a — derived* |
| 6 | **subordinate-clause aspect** | **−0.007** | **[−0.022, 0.008]** | 0.016 |

The two measures agree on rank order for every block they share, which is the
robustness check: one holds the model fixed and corrupts its input, the other
refits without the cue entirely.

**Three of the six intervals include zero** — connector, interactions and
subordinate-clause aspect. Only the top three blocks are separated from zero by
the bootstrap, so the hierarchy that can be asserted is a three-cue hierarchy
with a tail that the data cannot distinguish from nothing. The connector case is
an *n* problem (45 clauses, 34 of them BP) compounded by collinearity: the
Portuguese connector is almost always *mesmo*, and 23 of its 44 occurrences are
non-final against a 7.7% base rate, so position absorbs its work when it is
dropped. The `aspect_sc` case is the finding itself.

Direction of effect (standardised multinomial logit coefficients):

- `mc_durative` → Simul **+0.44**, Post −0.31 — a durative main clause pulls
  toward simultaneity.
- `mc_telic` → Simul **−0.39**, Ant +0.34 — a telic main clause pushes away from it.
- `tense_present` → Ant **+0.68** — a present-tense main clause pulls toward anteriority.
- `mc_perfect` → Ant −0.40, Post +0.21.
- `pos_nonfinal` → Ant **+0.58**, Post **−0.55** — non-final clauses read as anterior.

### What this says about the prior literature

- **Supported:** durativity → simultaneity, telicity → anteriority/posteriority
  (Silvano et al. 2021). The signs come out exactly as predicted.
- **Supported:** non-final position → anteriority (Lobo 2003), though as a
  third-ranked cue rather than a defining one, and only 7.7% of Portuguese
  clauses are non-final (80/1,038; 13.6% across the full corpus, inflated by BE's
  44%). Note the support is stronger than the ranking suggests: **almost no
  non-final Portuguese APC is annotated `Post`** (67 Ant, 2 Post, 11 Simul; both
  `Post` rows are medial clauses from the newest batch).
  Fronting appears to block the posterior reading outright; it ranks third
  because it is near-deterministic but rare, and ablation weights a cue by how
  often it is available.
- **Not supported:** the claim that the relevant aspectual configuration holds
  *in both clauses*. Dropping the explicit `both_telic` / `both_durative` terms
  costs nothing (the block's ablation is −0.004, CI [−0.014, 0.006]). The two clauses are not equal partners — and the
  subordinate clause's own aspect is not merely weak but **dispensable**
  (ablation −0.007, CI [−0.022, 0.008]). The ablation is a clean one: dropping
  `aspect_sc` also drops `both_telic` and `both_durative`, since they are
  products with an `sc_` factor, so the reduced model loses *every* channel
  through which subordinate-clause aspect could reach it — and scores the same.

### Model comparison

| model | macro-F1 | 95% CI |
|---|---|---|
| B0 majority | 0.196 | [0.186, 0.206] |
| B1 one-rule | 0.442 | [0.424, 0.463] |
| **B3 multinomial logit** | **0.695** | [0.665, 0.725] |
| B4 decision tree (d=4) | 0.649 | [0.616, 0.682] |
| B5 random forest | 0.692 | [0.662, 0.722] |
| B5 XGBoost | 0.695 | [0.665, 0.724] |
| *circular (DR + SR-SC)* | *0.750* | *[0.718, 0.783] — leakage demo* |
| *circular control: B3 on the same rows* | *0.736* | *[0.701, 0.769]* |

The circular bound can only be fitted on the 793 rows annotated for `SR-SC`, so it is
compared with B3 on those same rows (last row), not with the 1,038-row figure above.

Intervals come from resampling **sentence groups**, so they reflect how much the
score would move on a different sample of sentences. Two things follow.

1. **The cues combine additively.** XGBoost (0.695) and the additive logit
   (0.695) are statistically indistinguishable — the intervals almost coincide —
   and the forest (0.692) is no better. Gradient boosting has the capacity to exploit
   cue interactions and gains nothing from it, which is the empirical
   justification for reporting coefficients as the primary result.
2. **The genuine cues get as far as the circular annotation.** The
   definitionally leaky `DR`+`SR-SC` columns reach 0.750 against the honest
   model's 0.736 on the same rows, well inside each other's intervals: the temporo-aspectual cues
   recover essentially everything the discourse-relation label encodes about
   temporal order.

The one-rule baseline picks `mc_durative`, splitting durative → `Simul`,
non-durative → `Post` (57.9% train accuracy). The depth-3 tree roots on the same
feature, then splits on position, then tense — recovering the top three ranks of
the hierarchy independently.

## RQ2 — variety differences

### Confirmatory: pooled hierarchical model (B6)

One model over EP, BP, AP, MP (n=1,038; BE excluded, it is 95% anterior), fitted as
cues (M0), plus variety intercepts (M1), plus variety × cue slopes (M2). Varieties
are one-hot coded and the slope deviations carry an L2 penalty (partial pooling),
so every variety is shrunk equally toward the pooled slope. The penalty C is chosen
by CV inside each training fold. Each p-value is a parametric bootstrap: outcomes
are redrawn from the fitted null model and the deviance drop recomputed
(`experiments/07_hierarchical.py`, `results/tables/b6_*.csv`).

| term | df | deviance drop | p (bootstrap) | p (Holm) |
|---|---|---|---|---|
| variety intercepts (base rates) | 6 | 34.2 | 0.003 | — |
| variety × cue slopes, all cues | 102 | 128.0 | 0.033 | — |
| slopes: main-clause aspect | 18 | 33.7 | 0.050 | 0.30 |
| slopes: main-clause tense | 42 | 56.6 | 0.050 | 0.30 |
| slopes: position | 6 | 12.2 | 0.070 | 0.30 |
| slopes: both-clause interactions | 12 | 17.5 | 0.21 | 0.46 |
| slopes: connector | 6 | 5.7 | 0.15 | 0.46 |
| slopes: participial-clause aspect | 18 | 15.6 | 0.68 | 0.68 |

(0.003 is the smallest value 300 draws can give. `df` counts free parameters.)
Out-of-fold macro-F1 with C chosen inside each fold: M0 0.695 [0.665, 0.726],
M1 0.704 [0.674, 0.733], M2 0.700 [0.670, 0.729]; C = 0.01 on all the data, the
second-lowest value on the grid.

**Reading.** Varieties clearly differ in how often each reading occurs. The
evidence that they weight the cues differently is weak and not localisable: the
omnibus slope test is marginal (p = 0.033), no single block survives correction
(smallest Holm p = 0.30), the slope model does not beat the intercept-only model
out of fold, and CV shrinks the slopes almost to the pooled value. The
variety-specific part of each block weight is at most 0.05 on the logit scale,
against block weights of 0.02–1.1. So the data are consistent with a shared cue
hierarchy plus different base rates, with a small residual difference they cannot
pin to a cue. This is not evidence of equal weights, at n=193–345 per variety.

**Sensitivity.** Dropping both later batches (n=793, `*_orig.csv`) gives the same
picture: intercepts p = 0.003, slopes p = 0.027, no block below Holm 0.12 (connector
and tense raw p = 0.023 and 0.020), M0/M1/M2 = 0.736/0.738/0.738.

Per-variety block weights (`b6_weights.csv`, with bootstrap intervals) are the mean
size of the block's logit contribution within the variety. They mostly track how
often a cue varies there: the connector weighs 0.60 in BP and 0.02–0.10 elsewhere
because only BP has connectors, and position weighs 0.71 in BP and 0.16–0.39 elsewhere.
Read them as descriptive, not as a variety ranking; `shift` is the variety-specific
part and is the quantity RQ2 asks about.

### First pass: per-variety refits (descriptive)

Rank of each cue within each variety by refit ablation (1 = strongest):

| cue block | EP | BP | AP | MP | *BE* |
|---|---|---|---|---|---|
| main-clause aspect | **1** | **1** | 2 | 2 | *3* |
| main-clause tense | 2 | 4 | **1** | 4 | *4* |
| position | 4 | 2 | 3 | **1** | *1* |
| connector | 6 | 3 | 5 | 5 | *2* |
| telicity/durativity interactions | 5 | 5 | 4 | 3 | *4* |
| subordinate-clause aspect | 3 | 6 | 6 | 6 | *4* |

BE's three bottom ranks are a three-way tie, all given rank 4: its importances are all ≈0 or
negative, so the ordering among them is not meaningful.

Three differences stand out in the ablation pass (per-variety permutation is not
computed, so they are not cross-checked):

- **MP barely uses tense** (ablation 0.003, rank 4; EP +0.067, AP +0.113) and
  leans on **position** instead (0.118, its rank 1). AP inverts this and ranks
  tense first.
- **BP is the only variety where the connector does work** (0.026) — it is also
  the only variety where connectors are common (17.6% vs 0.4–2.4%).
- **Subordinate-clause aspect ranks 3rd in EP and 6th elsewhere** and is non-positive in
  three of the four (BP −0.064, AP −0.019, MP −0.001; EP +0.030). The pooled result (−0.004) is not an artefact of pooling,
  but the EP figure moved from −0.016 to +0.030 when the 95 new EP rows were
  added, which says how unstable a single-variety estimate is.

Base rates differ sharply too (EP/AP/MP posteriority-dominant, BP
anteriority-dominant at 48.2%). **Base-rate and cue-weight differences are not
the same thing**, and this pass cannot separate them; B6 above does.

### Why the per-variety refits cannot be read as a hierarchy

Fitting five independent models at n=193–345 is visibly unstable. Half the blocks
(connector, interactions, subordinate-clause aspect) show
**negative** ablation importance in at least one variety — dropping the cue
*improves* held-out score — which is the signature of overfitting at this sample
size, not evidence that a cue is harmful. So these numbers establish that a
difference exists and is worth modelling; they do not license a claim about any
particular variety's hierarchy.

### British English behaves as predicted

BE macro-F1 is **0.339 against a 0.325 majority floor**, and every cue importance
is ≈0 or negative. At 95% anterior there is almost no variance to explain, which
confirms empirically that BE cue weights are unidentifiable. BE stays a
contrastive reference: the construction is near-categorically anterior in
English and genuinely ambiguous in Portuguese.

## Next

1. Neural: does the full sentence beat the 17-feature inventory (T1)? Then causal
   ablations (T3), attention as a converging view (T2).
2. Leave-one-variety-out transfer (RQ3).
