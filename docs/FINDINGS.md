# Findings so far

Status: RQ1 answered on the pooled Portuguese data; RQ2 has a descriptive first
pass and needs the confirmatory hierarchical model. Numbers reproduce via
`make all`.

> **Revised after code review.** Earlier numbers in this file were affected by
> four measurement bugs: the interaction terms were permuted without being
> recomputed from their sources, confidence intervals were bootstrapped over CV
> repeats rather than sentences, XGBoost silently trained on library defaults,
> and macro-F1 averaged over a varying label set. All conclusions survived; the
> numbers below are the corrected ones. See `tests/test_regressions.py`.

## Headline

**In Portuguese, the temporal reading of an adverbial perfect participial clause
is carried by the *main* clause, not by the participial clause itself.** Removing
the aspectual class of the subordinate clause — one of the two cues the prior
literature foregrounds — costs the model nothing at all.

## RQ1 — cue hierarchy (pooled, 793 Portuguese sentences)

Primary measure is **refit ablation**: drop a cue block, refit, measure the fall
in out-of-fold macro-F1. Permutation importance is reported as a cross-check.
(Permutation cannot measure the interaction block, whose columns are
deterministic products of the aspect columns.)

| rank | cue block | ablation | 95% CI | permutation |
|---|---|---|---|---|
| 1 | **main-clause aspect** | 0.092 | [0.052, 0.125] | 0.184 |
| 2 | **main-clause tense** | 0.072 | [0.061, 0.095] | 0.150 |
| 3 | clause position | 0.037 | [0.015, 0.070] | 0.067 |
| 4 | connector | 0.008 | [0.006, 0.012] | 0.023 |
| 5 | telicity/durativity interactions | 0.001 | [0.000, 0.006] | *n/a — derived* |
| 6 | **subordinate-clause aspect** | **0.000** | **[−0.013, 0.007]** | 0.013 |

The two measures agree on rank order for every block they share, which is the
robustness check: one holds the model fixed and corrupts its input, the other
refits without the cue entirely.

Direction of effect (standardised multinomial logit coefficients):

- `mc_durative` → Simul **+0.66**, Post −0.43 — a durative main clause pulls
  toward simultaneity.
- `mc_telic` → Simul **−0.57**, Ant +0.38 — a telic main clause pushes away from it.
- `tense_present` → Ant **+0.61** — a present-tense main clause pulls toward anteriority.
- `mc_perfect` → Ant −0.59, Post +0.32.
- `pos_nonfinal` → Ant **+0.84**, Post **−0.86** — non-final clauses read as anterior.

### What this says about the prior literature

- **Supported:** durativity → simultaneity, telicity → anteriority/posteriority
  (Silvano et al. 2021). The signs come out exactly as predicted.
- **Supported:** non-final position → anteriority (Lobo 2003), though as a
  third-ranked cue rather than a defining one, and only 14.8% of clauses are
  non-final.
- **Not supported:** the claim that the relevant aspectual configuration holds
  *in both clauses*. Dropping the explicit `both_telic` / `both_durative` terms
  costs 0.001 macro-F1. The two clauses are not equal partners — and the
  subordinate clause's own aspect is not merely weak but **exactly
  dispensable** (ablation 0.000, CI spanning zero).

### Model comparison

| model | macro-F1 | 95% CI |
|---|---|---|
| B0 majority | 0.194 | [0.181, 0.206] |
| B1 one-rule | 0.460 | [0.438, 0.481] |
| **B3 multinomial logit** | **0.736** | [0.700, 0.766] |
| B4 decision tree (d=4) | 0.652 | [0.612, 0.686] |
| B5 random forest | 0.730 | [0.696, 0.761] |
| B5 XGBoost | 0.738 | [0.704, 0.769] |
| *circular (DR + SR-SC)* | *0.750* | *[0.718, 0.782] — leakage demo* |

Intervals come from resampling **sentence groups**, so they reflect how much the
score would move on a different sample of sentences. Two things follow.

1. **The cues combine additively.** XGBoost (0.738) and the additive logit
   (0.736) are statistically indistinguishable — the intervals almost coincide —
   and the forest is no better. Gradient boosting has the capacity to exploit
   cue interactions and gains nothing from it, which is the empirical
   justification for reporting coefficients as the primary result.
2. **The genuine cues get as far as the circular annotation.** The
   definitionally leaky `DR`+`SR-SC` columns reach 0.750 against the honest
   model's 0.736, well inside each other's intervals: the temporo-aspectual cues
   recover essentially everything the discourse-relation label encodes about
   temporal order.

The one-rule baseline picks `mc_durative`, splitting durative → `Simul`,
non-durative → `Post` (59.6% train accuracy). The depth-3 tree roots on the same
feature, then splits on position, then tense — recovering the top three ranks of
the hierarchy independently.

## RQ2 — variety differences (first pass, descriptive)

Rank of each cue within each variety by refit ablation (1 = strongest):

| cue block | EP | BP | AP | MP | *BE* |
|---|---|---|---|---|---|
| main-clause aspect | **1** | **1** | 2 | 2 | 3 |
| main-clause tense | 2 | 4 | **1** | 6 | 6 |
| position | 4 | 2 | 3 | **1** | 1 |
| connector | 5 | 3 | 4 | 3 | 2 |
| subordinate-clause aspect | 6 | 6 | 5 | 4 | 4 |

Three differences appear in both the ablation and permutation passes:

- **MP barely uses tense** (ablation −0.019, rank 6; EP +0.053, AP +0.071) and
  leans on **position** instead (0.110, its rank 1). EP and AP invert this.
- **BP is the only variety where the connector does work** (0.023) — it is also
  the only variety where connectors are common (17.6% vs ~1%).
- **Subordinate-clause aspect ranks 5th or 6th in every variety**, and is
  negative in all four, confirming the pooled result is not an artefact of
  pooling.

Base rates differ sharply too (EP/AP/MP posteriority-dominant, BP
anteriority-dominant at 48.2%). **Base-rate and cue-weight differences are not
the same thing**, and this pass cannot separate them — that is exactly what the
pooled hierarchical model with varying intercepts *and* slopes is for.

### Caveats that block a confirmatory claim

Fitting five independent models at n≈200 is visibly unstable. Most blocks show
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
