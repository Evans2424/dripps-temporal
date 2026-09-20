# Methods

What each experiment asks, what each model is, and why the design choices were
made. Citations are keys into `docs/references.bib`; every entry there was
checked against a primary record.

## The question

DRIPPS annotates 993 sentences across five varieties for the temporal relation
(`TR`) holding in a clause pair: `Ant` (anterior), `Post` (posterior) or `Simul`
(simultaneous). The project asks which cues carry that relation, whether they
form a hierarchy, and whether the hierarchy is stable across four Portuguese
varieties. The Portuguese analysis set is 793 rows; British English is a
contrastive reference, not a modelled variety.

Cues are grouped into six blocks, because individual dummy columns are not the
unit of a linguistic claim:

| block | columns |
|---|---|
| `position` | `pos_nonfinal` |
| `connector` | `has_connector` |
| `aspect_mc` | `mc_dynamic`, `mc_durative`, `mc_telic` |
| `aspect_sc` | `sc_dynamic`, `sc_durative`, `sc_telic` |
| `tense_mc` | `tense_present`, `tense_future`, `tense_nonfinite`, `mc_perfective`, `mc_progressive`, `mc_perfect`, `mc_irrealis` |
| `interaction` | `both_telic`, `both_durative` |

The aspect columns decompose each clause's aspectual class into the dynamic /
durative / telic primitives of [moens-steedman-1988-temporal], the tradition
running back to [vendler1957]. The `interaction` columns are deterministic
products — `both_telic = mc_telic × sc_telic` — which matters later.

## The experiments

### `01_audit.py` — what is actually in the corpus

Regenerates `docs/DATA_AUDIT.md`: label distribution per variety, cue
availability, data-quality counts, and the marginal association of each cue with
`TR`, measured as Cramér's V with the bias correction of [bergsma2013]. The
correction is not decorative — `TMC` has up to 16 levels against `Position`'s
three, and uncorrected V rewards high cardinality, which would put tense at the
top of the hierarchy for the wrong reason.

It also quantifies the leakage: `DR` and `SR-SC` encode the answer, and the
audit reports exactly how much.

### `02_baselines.py` — does the extra machinery earn its place?

Runs the ladder under repeated grouped cross-validation and reports macro-F1 with
a confidence interval. It also runs one deliberately circular model that is
allowed to see `DR` and `SR-SC`, as an upper bound showing what the leaky
features would buy. That row is a demonstration, not a result.

### `03_cue_hierarchy.py` — which cues carry the signal (RQ1)

Four views of the same question: the marginal association table (descriptive,
confounded by collinearity); the single best cue via one-rule; block importance
by refit ablation and by permutation; direction of effect from the standardised
logit coefficients; and a depth-3 tree whose splits can be read verbatim.

### `04_variety_hierarchies.py` — is the hierarchy stable (RQ2)

Refits per variety and compares rankings. Descriptive only: at n≈200 per variety
the separate fits overfit and the rankings are not strictly comparable. The
confirmatory analysis is a pooled hierarchical model that has not been built yet.

### `05_explain.py` — does a non-linear model agree?

TreeSHAP [lundberg2017; lundberg2020] on the two ensembles, as a cross-check on
the ranking the ablation produces.

## The model ladder

Each rung exists to answer "does the next rung earn its place?" All are from
scikit-learn [pedregosa2011] except XGBoost.

| rung | what it is | why it is on the ladder |
|---|---|---|
| **B0 majority** | predicts the most frequent class | the floor any result must clear |
| **B1 one-rule** | picks the single most informative feature, then predicts the majority class per value [holte1993] | the honest floor for a *hierarchy* claim: if the full model cannot beat the best single cue, there is no hierarchy worth reporting |
| **B3 logit** | L2 multinomial logistic regression on standardised features | the paper's primary inferential model — coefficients are readable and signed |
| **B4 tree** | CART, `max_depth=4`, `min_samples_leaf=20` | finds cue *combinations* an additive model cannot express, shallow enough to report verbatim |
| **B5 forest** | 1000 trees, `min_samples_leaf=5` [breiman2001] | is any nonlinearity real, or is the logit leaving nothing on the table? |
| **B5 xgboost** | gradient-boosted trees, depth 3 [chen2016] | same question, different inductive bias |

The tree reported in the viewer is fit at depth 3 — shallower than the `B4
tree(d=4)` row scored on the ladder — because it is there to be *read*, not to
compete on macro-F1.

## Design decisions that carry the results

**Leakage is blocked structurally.** `DR` and `SR-SC` encode the temporal
relation directly. `leakage.py` raises if either reaches a feature matrix, rather
than relying on anyone remembering to drop them.

**Splits are grouped by sentence.** Sentences with multiple annotated clause
pairs occupy several rows. Splitting rows independently would put the same
sentence in train and test, so every split groups on `sentence_group`.

**Intervals come from resampling sentence groups, not CV folds.** Fold-to-fold
spread measures the split, not the data: repeats differ only by fold seed on
identical rows, so quantiles over folds estimate the variance of the
randomisation and come out far too narrow — the small-sample failure
[varoquaux2018] describes. Every interval reported here is a percentile bootstrap
over sentence groups, the independent unit. Fold spread is reported separately as
`sd_fold` and is not an error bar.

**Ablation and permutation answer different questions.** Correlated cues
compensate for one another, so dropping a block and refitting tests *necessity*,
while permuting a block in the held-out fold tests how much the fitted model
*relies* on it. Both are reported; neither alone is sound here.

**The interaction block is ablation-only.** Permuting a column that is the
product of two others either fabricates impossible rows — `both_telic = 1` where
`mc_telic = 0` — or, once recomputed, does nothing. Unrestricted permutation of
correlated features forces the model to extrapolate off the data manifold
[hooker2021], which is why derived columns are recomputed after every shuffle and
why the derived block is measured by ablation instead.

**SHAP is run path-dependent for the same reason.** The interventional estimator
marginalises features independently and would score those same impossible rows;
the path-dependent estimator follows the splits the fitted trees actually took.
Shapley values still split credit among correlated features rather than assigning
it to one [aas2021], so SHAP is read here as a check on rank order, never as a
rival estimate of block importance.

**Impurity importance is never used.** Gini rewards high-cardinality features
[strobl2007], and `tense_mc` expands to seven columns against `position`'s one.

**SHAP figures are not out-of-fold.** They describe models fit on the whole
analysis set, so they characterise a fitted model, not generalisation. They are
not comparable with the macro-F1 numbers on the ladder. Forest and XGBoost values
are not comparable with *each other* either: a forest's leaves hold class
probabilities, XGBoost's hold log-odds increments.
