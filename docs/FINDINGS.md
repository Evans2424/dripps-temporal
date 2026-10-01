# Findings so far

Status: RQ1 answered on the pooled Portuguese data; RQ2 has a descriptive first
pass and needs the confirmatory hierarchical model. Numbers reproduce via
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

> **Second annotation batch.** Numbers below include 150 newly annotated
> Portuguese sentences (50 each EP, AP, MP), so the Portuguese analysis set is
> 943 clauses, up from 793 (`docs/ANNOTATION_SCHEME.md`, "Second annotation
> batch"). The conclusions and every rank in the pooled hierarchy are unchanged;
> the scores are lower. B3 falls from 0.736 to 0.705 macro-F1, and EP's own
> logit from 0.694 to 0.615. On the 793 original rows B3 still scores 0.736, so
> the drop appears once the new rows are included. They are 94% final-position and were not
> annotated for `SR-SC`. Whether they are simply harder or were annotated
> differently cannot be told from the data: there is no agreement figure for them.

## Headline

**In Portuguese, the temporal reading of an adverbial perfect participial clause
is carried by the *main* clause, not by the participial clause itself.** Removing
the aspectual class of the subordinate clause — one of the two cues the prior
literature foregrounds — costs the model nothing at all (the point estimate is
slightly negative).

## RQ1 — cue hierarchy (pooled, 943 Portuguese clauses)

Primary measure is **refit ablation**: drop a cue block, refit, measure the fall
in out-of-fold macro-F1. Permutation importance is reported as a cross-check.
(Permutation cannot measure the interaction block, whose columns are
deterministic products of the aspect columns.)

| rank | cue block | ablation | 95% CI | permutation |
|---|---|---|---|---|
| 1 | **main-clause aspect** | 0.074 | [0.048, 0.099] | 0.158 |
| 2 | **main-clause tense** | 0.045 | [0.018, 0.068] | 0.133 |
| 3 | clause position | 0.037 | [0.022, 0.054] | 0.067 |
| 4 | connector | 0.005 | [−0.005, 0.014] | 0.028 |
| 5 | telicity/durativity interactions | −0.004 | [−0.012, 0.003] | *n/a — derived* |
| 6 | **subordinate-clause aspect** | **−0.008** | **[−0.022, 0.008]** | 0.015 |

The two measures agree on rank order for every block they share, which is the
robustness check: one holds the model fixed and corrupts its input, the other
refits without the cue entirely.

**Three of the six intervals include zero** — connector, interactions and
subordinate-clause aspect. Only the top three blocks are separated from zero by
the bootstrap, so the hierarchy that can be asserted is a three-cue hierarchy
with a tail that the data cannot distinguish from nothing. The connector case is
an *n* problem (45 clauses, 34 of them BP) compounded by collinearity: the
Portuguese connector is almost always *mesmo*, and 23 of its 44 occurrences are
non-final against a 7.2% base rate, so position absorbs its work when it is
dropped. The `aspect_sc` case is the finding itself.

Direction of effect (standardised multinomial logit coefficients):

- `mc_durative` → Simul **+0.52**, Post −0.42 — a durative main clause pulls
  toward simultaneity.
- `mc_telic` → Simul **−0.44**, Ant +0.39 — a telic main clause pushes away from it.
- `tense_present` → Ant **+0.61** — a present-tense main clause pulls toward anteriority.
- `mc_perfect` → Ant −0.42, Post +0.20.
- `pos_nonfinal` → Ant **+0.77**, Post **−0.87** — non-final clauses read as anterior.

### What this says about the prior literature

- **Supported:** durativity → simultaneity, telicity → anteriority/posteriority
  (Silvano et al. 2021). The signs come out exactly as predicted.
- **Supported:** non-final position → anteriority (Lobo 2003), though as a
  third-ranked cue rather than a defining one, and only 7.2% of Portuguese
  clauses are non-final (68/943; 13.6% across the full corpus, inflated by BE's
  44%). Note the support is stronger than the ranking suggests: **not one
  non-final Portuguese APC is annotated `Post`** (60 Ant, 0 Post, 8 Simul).
  Fronting appears to block the posterior reading outright; it ranks third
  because it is near-deterministic but rare, and ablation weights a cue by how
  often it is available.
- **Not supported:** the claim that the relevant aspectual configuration holds
  *in both clauses*. Dropping the explicit `both_telic` / `both_durative` terms
  costs nothing (the block's ablation is −0.004, CI [−0.012, 0.003]). The two clauses are not equal partners — and the
  subordinate clause's own aspect is not merely weak but **dispensable**
  (ablation −0.008, CI [−0.022, 0.008]). The ablation is a clean one: dropping
  `aspect_sc` also drops `both_telic` and `both_durative`, since they are
  products with an `sc_` factor, so the reduced model loses *every* channel
  through which subordinate-clause aspect could reach it — and scores the same.

### Model comparison

| model | macro-F1 | 95% CI |
|---|---|---|
| B0 majority | 0.195 | [0.184, 0.207] |
| B1 one-rule | 0.451 | [0.430, 0.471] |
| **B3 multinomial logit** | **0.705** | [0.673, 0.736] |
| B4 decision tree (d=4) | 0.629 | [0.594, 0.663] |
| B5 random forest | 0.700 | [0.668, 0.730] |
| B5 XGBoost | 0.701 | [0.670, 0.732] |
| *circular (DR + SR-SC)* | *0.750* | *[0.718, 0.782] — leakage demo* |
| *circular control: B3 on the same rows* | *0.736* | *[0.700, 0.766]* |

The circular bound can only be fitted on the 793 rows annotated for `SR-SC`, so it is
compared with B3 on those same rows (last row), not with the 943-row figure above.

Intervals come from resampling **sentence groups**, so they reflect how much the
score would move on a different sample of sentences. Two things follow.

1. **The cues combine additively.** XGBoost (0.701) and the additive logit
   (0.705) are statistically indistinguishable — the intervals almost coincide —
   and the forest (0.700) is no better. Gradient boosting has the capacity to exploit
   cue interactions and gains nothing from it, which is the empirical
   justification for reporting coefficients as the primary result.
2. **The genuine cues get as far as the circular annotation.** The
   definitionally leaky `DR`+`SR-SC` columns reach 0.750 against the honest
   model's 0.736 on the same rows, well inside each other's intervals: the temporo-aspectual cues
   recover essentially everything the discourse-relation label encodes about
   temporal order.

The one-rule baseline picks `mc_durative`, splitting durative → `Simul`,
non-durative → `Post` (59.2% train accuracy). The depth-3 tree roots on the same
feature, then splits on position, then tense — recovering the top three ranks of
the hierarchy independently.

## RQ2 — variety differences (first pass, descriptive)

Rank of each cue within each variety by refit ablation (1 = strongest):

| cue block | EP | BP | AP | MP | *BE* |
|---|---|---|---|---|---|
| main-clause aspect | **1** | **1** | 2 | 2 | *3* |
| main-clause tense | 2 | 4 | **1** | 4 | *4* |
| position | 3 | 2 | 3 | **1** | *1* |
| connector | 4 | 3 | 5 | 5 | *2* |
| telicity/durativity interactions | 5 | 5 | 6 | 3 | *4* |
| subordinate-clause aspect | 6 | 6 | 4 | 6 | *4* |

BE's three bottom ranks are a three-way tie, all given rank 4: its importances are all ≈0 or
negative, so the ordering among them is not meaningful.

Three differences stand out in the ablation pass (per-variety permutation is not
computed, so they are not cross-checked):

- **MP barely uses tense** (ablation 0.003, rank 4; EP +0.031, AP +0.135) and
  leans on **position** instead (0.118, its rank 1). AP inverts this and ranks
  tense first.
- **BP is the only variety where the connector does work** (0.026) — it is also
  the only variety where connectors are common (17.6% vs 0.4–2.4%).
- **Subordinate-clause aspect ranks 4th to 6th in every variety** (4th only in
  AP, 6th in EP, BP and MP) and is non-positive in three of the four (EP −0.016,
  BP −0.064, MP −0.001; AP +0.008), so the pooled result is not an artefact of
  pooling.

Base rates differ sharply too (EP/AP/MP posteriority-dominant, BP
anteriority-dominant at 48.2%). **Base-rate and cue-weight differences are not
the same thing**, and this pass cannot separate them — that is exactly what the
pooled hierarchical model with varying intercepts *and* slopes is for.

### Caveats that block a confirmatory claim

Fitting five independent models at n=193–250 is visibly unstable. Half the blocks
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

1. Pooled hierarchical multinomial, varying intercepts + slopes by variety →
   the confirmatory RQ2 result.
2. Neural: does the full sentence beat the 17-feature inventory (T1)? Then causal
   ablations (T3), attention as a converging view (T2).
3. Leave-one-variety-out transfer (RQ3).
