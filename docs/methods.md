# Methods

What each experiment asks, what each model is, and why the design choices were
made. Citations are keys into `docs/references.bib`; every entry there was
checked against a primary record.

## The question

DRIPPS annotates 1,238 clauses (993 from the original export plus a 150-clause and a 95-clause later batch) across five varieties for the temporal relation
(`TR`) holding in a clause pair: `Ant` (anterior), `Post` (posterior) or `Simul`
(simultaneous). The project asks which cues carry that relation, whether they
form a hierarchy, and whether the hierarchy is stable across four Portuguese
varieties. The Portuguese analysis set is 1,038 rows; British English is a
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

Refits per variety and compares rankings. Descriptive only: at n≈200–350 per variety
the separate fits overfit and the rankings are not strictly comparable. The
confirmatory analysis is `07_hierarchical.py` (B6).

### `07_hierarchical.py` — pooled model (RQ2, B6)

Nested multinomial logits over EP/BP/AP/MP: cues, plus variety indicators, plus
variety × cue slopes. Varieties are one-hot coded, main effects are scaled up so the
L2 penalty barely touches them, and the slope deviations keep the penalty (partial
pooling, equal for every variety). The penalty C is chosen by shuffled 5-fold CV
inside each outer training fold, so the reported macro-F1 is nested. Tests are
deviance drops with parametric-bootstrap p-values (outcomes redrawn from the fitted
null model), Holm-corrected over the six cue blocks. Block weights are the mean norm
of the block's centred logit contribution within a variety, with group-bootstrap
intervals conditional on the chosen C. `--original-only` reruns without both later
batches. Fitted with scikit-learn: the exact MLE does not converge on the sparse
design, and rows are redrawn independently in the bootstrap, which ignores the 55
multi-APC rows' clustering.

### `08_encoder.py` — does the sentence text add to the inventory? (T1, T3)

**Status.** The code is complete and tested; the tuned GPU run is pending, so no
tuned number is quoted here. A first run with fixed hyperparameters (whole-sentence
input, learning rate 2e-5, 4 epochs) scored macro-F1 0.469 [0.415, 0.528] against B3's
0.694 on the same folds. It was under-trained (Ant F1 0.25, seed spread 0.033, about 200
optimiser steps per fold) and is superseded by the procedure below.

**Question.** B3 reads 17 binary cues that annotators coded. T1 asks whether a
fine-tuned multilingual encoder, which reads the raw sentence, recovers that signal
or finds more of it. T3 asks, by perturbing the text, which parts of the sentence the
encoder uses.

**Input.** Only `sentence_norm`. No annotation column (`DR`, `SR-SC`, `ATMC`, ...) is
an input, so the leakage guard on the feature models still holds. The connector and
any temporal adverbial are in the text and are therefore available to the encoder.

**Finding the clauses.** The corpus does not annotate where the participial clause
(APC) is, so the spans are recovered:

| span | how | tool |
|---|---|---|
| APC verb group | exactly one `tendo` (the pattern in `schema.APC_AUXILIARY`), through the participle | rule |
| participle | the first word after `tendo` that is a participle, skipping only a closed list of adverbs (*já, ainda, também, ...*) and *sido/estado*, within four tokens and no punctuation | rule |
| main verb | the one finite verb (`VerbForm=Fin`) among the sentence root and its `aux`, `aux:pass` and `cop` children, outside the APC group | spaCy `pt_core_news_sm` |
| temporal adverbials | a closed lexicon (*depois, antes, dois dias depois, já, ...*), ignoring any inside the APC group | rule |

A participle is a word ending *-ado/-ido/-ído* (with gender and number) or one of a
short list of irregular stems (*feito, dito, posto, visto, ...*). In a copular APC such
as *tendo sido campeão*, *sido* is taken as the participle. If something else stands
between `tendo` and the participle (a subject, an object), or the sentence has two
`tendo`, the span is **not guessed**: it is missing and the row is left out of that
perturbation.

The **POS tagger and parser** is spaCy's `pt_core_news_sm` (statistical tagger,
morphologizer and dependency parser; the entity recogniser and lemmatiser are
switched off; spaCy 3.8 tested). It does two jobs only: finding the main verb, and
deciding whether the first word of the main clause is a proper noun, number or
acronym (`PROPN`, `X`, `NUM`, or all capitals) so that moving the APC to the front
does not also change that word's capitalisation. The participle is found by the rule
above, not by the tagger. spaCy is needed only for this step (`requirements-neural.txt`),
never by the app. On the 1,038 Portuguese rows the participle is found in 943, the main
verb in 660 (every row with a main verb also has a participle), the APC can be moved in
437 and an adverbial is present in 382. The main-verb recovery is the weak point:
the 660 rows are those where the parser found a single finite root, a selected subset.
`t3_span_sample.csv` holds 50 rows with both spans for a hand check of both columns.

**Architecture.** `xlm-roberta-base` (12 layers, 768 dimensions, all weights
fine-tuned) with a head of dropout (0.1) and one linear layer. Ten variants (the first four are the main
experiment; the rest are exploratory, see below):

- *plain*: the head sees the first-token vector.
- *marked*: the text becomes `[[ tendo X ]]` around the APC group and `<< verb >>` around
  the main verb. They are ordinary strings, not new tokens, because randomly
  initialised marker embeddings do poorly on ~800 rows. The head sees the first-token
  vector plus the vectors of the first sub-word of the participle and of the main verb,
  located through the tokenizer's character offsets. A missing span falls back to the
  first-token vector.
- *plain_feat* and *marked_feat*: the same two, with B3's annotated cues (the 17-column
  matrix of `features.build`: main-clause aspect and tense, clause position, connector,
  participial-clause aspect) concatenated to the text vector before the linear layer. The
  columns are standardised on the training fold with B3's own scaler. The layer is written as
  two whose logits add, so the cue weights (initialised at zero) get their own learning
  rate instead of the encoder's 3e-5 to 5e-5, which would barely move them in a few
  hundred steps (about 300 to 600 per refit, depending on the epochs). That rate is tuned, not fixed: too low ignores the cues and the model
  collapses to the text-only encoder, too high lets them drown the text and it collapses
  to the logit. The grid for these two variants is the text-only grid crossed with feat_lr 1e-4,
  1e-3 or 1e-2 (12 settings), so a feature variant and its text-only twin differ in the
  cues and not in tuning budget; the
  chosen rate is saved per fold in `t1_encoder_config.csv`, and rates pinned at an edge of
  the grid are reported as such. Comparing `*_feat` with the text-only variant asks whether the
  encoder gains from the cues; comparing it with B3 asks whether the text adds to them.
  They are not averaged with B3, since they already see the cues. T3 runs on the plain
  text model only.
- *plain_res*, *marked_res* (residual on B3): the text head learns a correction to B3's
  log-odds, which are added to its logits, and it starts at zero, so the model begins as B3
  and the text must earn any change (a product of experts, [clark-etal-2019-dont]). The
  offset for a training row comes from a B3 fitted without that row's sentence group
  (inner 5-fold), minus the log of the training-fold class prior so that
  it matches the class-weighted loss; for the held-out fold it is the fold's own B3. It
  needs no feature learning rate, so the grid is the text-only one. Not averaged with B3. (Inner-fit rows get offsets from B3 fits that
  include the inner-validation rows, so the inner score that picks lr and epochs is slightly optimistic;
  B3 has 17 inputs, so the effect is small.)
- *plain_txt*, *marked_txt* (cues as words): the five annotated cues are written in front
  of the sentence (`aspeto principal: ...; tempo: ...; aspeto subordinada: ...; posição:
  ...; conector: ...`, with the cue codes rewritten as the Moens and Steedman primitives and the TAM
  bundle in words, because the corpus codes such as `PIMP-Conj` are meaningless to a pretrained
  model), and the marks of the marked variant move by the prefix length.
  This lets the pretrained model read the cues in language it has seen, with no new
  weights. Not averaged with B3.
- *plain_aux*, *marked_aux* (auxiliary cue prediction): the text vector also feeds one
  small head per cue (main-clause aspect, subordinate aspect, tense, position, connector),
  trained with a cross-entropy of weight 0.5 in total, missing cues ignored. The cues
  shape the encoder during training and are not an input, so at test time the model sees
  text only and can still be fused with B3.

Any variant can run on another checkpoint (`--checkpoint`): `bertimbau` and
`bertimbau-large` (BERTimbau, Portuguese BERT), `albertina` and `albertina-900m`
(Albertina PT-PT, DeBERTa-based, trained on European Portuguese, the largest variety
here), `mdeberta`, and `xlmr-large`. A variant on a non-default checkpoint is named like
`marked@bertimbau` in the tables, and the paired differences include each one against
the same variant on xlm-roberta-base. The perturbation (T3) runs stay on the plain
xlm-roberta-base model. Pooling at the marks needs a fast tokenizer, which the script
asserts. Hyperparameters are tuned per checkpoint with the same grid, so a Portuguese
model is not handicapped by xlm-roberta's settings.

Training: AdamW (weight decay 0.01), linear schedule with 10% warm-up, batch 16,
maximum length 256, gradient clipping at 1, bf16 autocast on GPU, cross-entropy
weighted by inverse class frequency (the metric is macro-F1).

**Procedure.** Folds are B3's: `StratifiedGroupKFold(5)` on `sentence_group`, seeds
`SEED+0..4`, and B3 is rescored on those five seeds so the comparison is paired.
Hyperparameters are chosen **inside each training fold**: one sentence-grouped inner
split (about 20%), four settings (learning rate 3e-5 or 5e-5 × 6 or 12 epochs), best
inner macro-F1 wins, then a refit on the whole training fold. The held-out fold is never
used for selection, and the chosen settings are saved. Scores are pooled out-of-fold
macro-F1 per seed, averaged over seeds, with group-bootstrap intervals (2,000 draws)
and paired differences from the same resamples, also on the rows whose text is not
shared between sentences annotated for different APCs.

**Fusion.** B3 and encoder probabilities are averaged at a fixed weight of 0.5 (0.25
and 0.75 reported as post hoc sensitivity). The encoder's loss is class-weighted and
B3's is not, so its posteriors are first multiplied by the training-fold class prior
and renormalised. Fusion beating B3 with a paired interval that excludes 0 is the
evidence that text adds to the inventory.

**T3, causal ablation** (plain model, out-of-fold, so no checkpoint is kept): mask the
participle, mask the main verb, mask the temporal adverbials, and move a final APC to the
front (only when it has no internal comma and no quote or bracket is split). Each mask
has a control that masks the same number of random words outside the APC group, the
main verb and the adverbials, so "any masking hurts" is not read as an effect. The
outputs are the change in the predicted distribution and in P(true reading), with
group-bootstrap intervals, plus mask-minus-control and main-verb-minus-participle
contrasts. The tense-swap probe the design called for (PP to Pres-Ind) is not built:
it needs a Portuguese verb inflector.

**Decision rules, fixed before the run.** The encoder "recovers the inventory's signal"
only if its paired difference from B3 includes 0 or is positive. T3 is interpreted
only if the tuned plain model reaches macro-F1 0.60; below that it is exploratory.

**Exploratory variants.** The decision rules were written for `plain`, `marked`, their `_feat` twins and fusion. The `_res`, `_txt` and `_aux` variants and the other checkpoints were added afterwards, so every contrast involving them is exploratory. No multiplicity correction is applied to the roughly sixty paired intervals, and one interval that excludes 0 among them is not evidence on its own. Read them as candidates to confirm on a held-out batch or a fresh set of seeds.

**Limits.** The encoder is far smaller-data than its pre-training assumes (1,038
sentences); inner selection uses ~165 validation rows and is noisy; the markers may
shift a cased model by themselves (plain is the control); T3 is run on the plain model
only; the main-verb subset is selective; the fold cache and tables come from a GPU run
that is not bit-reproducible.

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
