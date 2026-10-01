# The linguistics of DRIPPS, and what each model does with it

Companion to `docs/methods.md`. Methods says *what* each experiment runs;
this says *what the thing being measured is*, in linguistic terms, with corpus
examples, and what each model can and cannot see about it.

Every number and every example here is from the current corpus and the current
tables in `results/tables/`, recomputed for this document, and include the 150-clause second batch
(`docs/ANNOTATION_SCHEME.md`) unless a sentence says "original". Citation keys are
entries in `docs/references.bib`.

**Contents.**
[1 The construction](#1-the-construction) ·
[2 The target](#2-the-target-tr) ·
[3 The cue inventory](#3-the-cue-inventory) ·
[4 Why a multivariate model](#4-why-none-of-this-can-be-read-off-a-crosstab) ·
[5 What each model does](#5-what-each-model-actually-does-with-the-cues) ·
[6 The three importance measures](#6-the-three-importance-measures-and-why-they-disagree) ·
[7 The encoder models](#7-the-encoder-models-not-yet-built) ·
[8 Reading the viewer](#8-reading-the-viewer-panel-by-panel) ·
[9 What you can claim](#9-what-you-can-claim-and-what-you-cannot)

---

## 1. The construction

The object of study is the **adverbial perfect participial clause** (APC): a
subordinate clause built from the auxiliary *ter* in the gerund — **`tendo`** —
plus a past participle. English uses **`having`** the same way.

> PTEU88 · `Final` · `PP` · MC `Culm` · SC `Culm` · **Post**
> A lei das plataformas deu entrada no parlamento em janeiro de 2017, **tendo
> sido aprovada** em março.

> ENBE160 · `Initial` · `Pres` · MC `St` · SC `Pro` · **Ant**
> Alex said: '**Having lived** through 104 years, you accept people as they are.'

Two properties make this construction the right probe for a cue-hierarchy study.

**It is almost never marked.** 1,088 of 1,143 clauses (95.2%) carry no discourse
connector at all. There is no *after*, no *depois de*, no *enquanto* to read the
temporal relation off. Whatever tells a reader that the participial event
precedes, follows or overlaps the main clause event has to be inferred from
tense, aspect and position. The inference is forced through exactly the channels
the corpus annotates, which is why cue weights are identifiable here and would
not be in a corpus of explicitly marked subordination.

**Its morphology points the wrong way.** The participle is *perfect*. The
descriptive tradition associates perfect morphology in these clauses with
anteriority [lobo2003]. That holds for English and fails for Portuguese, which
is the phenomenon the paper is about — see §2.

**Practical notes that shape the modelling.**

- **The APC span is not annotated.** It is recovered heuristically from the
  `tendo` / `having` token: present in all 943 Portuguese analysis rows and 199
  of 200 BE rows. The span is unambiguous for 895 of the 943 Portuguese rows
  (one row, one `tendo`). The other 48 need a rule: 29 rows share a single `tendo`
  with another row of the same sentence, and 19 rows (15 distinct sentences)
  contain more than one `tendo`, so which participial clause a row refers to
  cannot be recovered from the token alone. Any analysis that needs the span (the neural ablations, §7) has to
  report that recovery rate.
- **26 sentences are annotated twice or more, across 55 rows** — one row per
  APC in a multi-APC sentence. Every split therefore groups on
  `sentence_group`. Splitting rows independently would put the same sentence in
  train and test.

> PTEU8 / PTEU9 — one sentence, two rows, two participial clauses:
> O Santander Totta reviu a sua política de comissões, **tendo agravado** os
> custos a 50 mil contas e **subido** o preço da maioria dos cartões em 5,4 %.

---

## 2. The target: `TR`

`TR` is the temporal relation between the subordinate (participial) clause and
the main clause. Three values:

| | | |
|---|---|---|
| **`Ant`** | anteriority | the SC situation precedes the MC situation |
| **`Post`** | posteriority | the SC situation follows the MC situation |
| **`Simul`** | simultaneity | the two overlap |

One real example of each, from European Portuguese, with its annotated cues:

> **`Ant`** — PTEU81 · `Final` · `Pres-Ind` · MC `St` · SC `Culm`
> Este arguido **está** atualmente em fuga, **tendo sido condenado** a pena de
> prisão, à revelia, noutro processo.
> *The sentencing precedes the state of being at large.*

> **`Post`** — PTEU139 · `Final` · `PP` · MC `Culm` · SC `Culm`
> Os outros dois indivíduos **puseram-se** em fuga, **tendo sido** um deles
> **capturado** de seguida.
> *The capture follows the flight. `de seguida` makes the order explicit — to
> the reader, though not to the feature model, which never sees adverbials.*

> **`Simul`** — PTEU15 · `Final` · `PP` · MC `St` · SC `CP`
> **Foram consideradas** 67 freguesias, **tendo** os dados **sido recolhidos**
> entre abril e setembro deste ano.
> *The data collection and the scope of the study hold over the same interval.*

### The distribution is itself a result

| | Ant | Post | Simul | n |
|---|---|---|---|---|
| **EP** European | 69 | **96** | 85 | 250 |
| **BP** Brazilian | **93** | 38 | 62 | 193 |
| **AP** Angolan | 44 | **133** | 73 | 250 |
| **MP** Mozambican | 28 | **124** | 98 | 250 |
| *BE* British English | *190 (95%)* | *2* | *8* | *200* |

Three facts to state in this order:

1. **English is near-categorical.** 190/200 anterior. The construction is
   effectively unambiguous, which is why BE cue weights are unidentifiable and
   BE is a contrastive reference rather than a fifth variety.
2. **Portuguese is genuinely ambiguous, and leans the other way.** EP, AP and MP
   are posteriority-dominant. A *perfect* participle whose most frequent reading
   is *posterior* is the puzzle.
3. **BP inverts.** It is the one anteriority-dominant Portuguese variety (48.2%)
   — and it also patterns toward English on position, connectors and stative
   main clauses (§3). That convergence is a distributional fact; contact
   explanations for it are not testable on newswire data and should be flagged
   as speculation.

**How a perfect participle gets a posterior reading.** The corpus answer is
structural: 92.8% of Portuguese APCs are clause-final, and the posterior cases
are overwhelmingly a *perfective, telic main clause followed by a final
participial clause* — `Culm` main clause with `PP` tense goes posterior in
268/367 cases (73%). The participle marks its own event as complete, not as
prior to the main clause; in a final position after a completed main event, the
natural reading is narrative continuation. Note this is the reading the
distribution supports, and the argument the paper has to make explicitly — the
numbers establish the pattern, not the account of it.

### `DR` and `SR-SC` are not cues, they are the answer

Two annotated columns are excluded by assertion, not by judgement. `SR-SC`
(semantic role of the subordinate clause) takes the values `before` and `after`
under the `asynchrony` relation — that *is* temporal order, and in the corpus
`before` → `Ant` and `after` → `Post` without exception. `DR` == `asynchrony`
never co-occurs with `Simul`; `conjunction`, `elaboration`, `exemplification`
and `manner` are always `Simul`.

Both are part of the ISO 24617-8 DR-core inventory [iso24617-8], and both are
downstream of the temporal relation by definition. `src/dripps/leakage.py`
raises if either reaches a feature matrix. Their predictive value is reported
exactly once, as the `circular` row on the ladder (§5), which is a measurement
of annotation redundancy and not a model of anything.

---

## 3. The cue inventory

Five annotated cues become 17 binary columns in six blocks. The blocks, not the
columns, are the unit of a linguistic claim — `tense_mc` is seven columns and
`position` is one, so comparing single columns would compare a claim about tense
against a claim about one contrast within position.

| block | columns | annotated source |
|---|---|---|
| `position` | `pos_nonfinal` | `Position` |
| `connector` | `has_connector` | `CNT` |
| `aspect_mc` | `mc_dynamic`, `mc_durative`, `mc_telic` | `ATMC` |
| `aspect_sc` | `sc_dynamic`, `sc_durative`, `sc_telic` | `ATSC` |
| `tense_mc` | `tense_present`, `tense_future`, `tense_nonfinite`, `mc_perfective`, `mc_progressive`, `mc_perfect`, `mc_irrealis` | `TMC` |
| `interaction` | `both_telic`, `both_durative` | derived |

### 3.1 Aspectual class → three primitives

`ATMC` and `ATSC` annotate the **aspectual class** of each clause using the
event-nucleus taxonomy of Moens & Steedman [moens-steedman-1988-temporal], which
stands in the Vendlerian tradition [vendler1957]. Rather than five dummies, the
models use the three primitives that define the classes. Each primitive is a
real linguistic distinction with a standard diagnostic:

**`dynamic`** — does the situation involve change, or does it just hold?
States do not. *ser presidente*, *ter potencial*, *estar em fuga* are static;
*correr*, *construir*, *chegar* are dynamic.

**`durative`** — does the situation occupy an extended interval, or is it
instantaneous? The test is a *for*-adverbial: *correu **durante** uma hora* is
fine, *chegou durante uma hora* is not.

**`telic`** — does the situation have an inherent endpoint whose attainment is
entailed? The test is the imperfective paradox: *estava a correr* entails *ele
correu* (atelic), but *estava a construir uma casa* does **not** entail *ele
construiu uma casa* (telic — the house may never have been finished).

The five attested classes are each a unique combination, which is why the
primitives lose no information:

| label | class | dyn | dur | tel | corpus example (main clause) |
|---|---|:-:|:-:|:-:|---|
| `St` | State | − | + | − | PTBR86 *O Creta **tem** potencial para fazer isso mudar* |
| `Pro` | Process | + | + | − | PTAO139 *De 1993 a 1995 **orientou** a selecção francesa* |
| `Culm` | Culmination | + | − | + | PTBR109 *Em 1730, **teve início** uma nova construção* |
| `CP` | Culminated process | + | + | + | PTMZ148 ***Actuou** 60 vezes pela selecção portuguesa* |
| `Pon` | Point | + | − | − | PTMZ56 *a vítima **gritou** por socorro* |

`Pon` is attested **twice**, both in MP, and never on the subordinate clause.
It is kept rather than merged because the primitive encoding handles it without
spending a parameter on it.

**The hypothesis under test** [silvano2021apc]: anteriority and posteriority
readings track **telic** situations, simultaneity tracks **durative** ones, *in
both clauses*. The corpus, pooled over the 943 Portuguese rows:

| main clause | Ant | Post | Simul | reading |
|---|---|---|---|---|
| durative (`St`,`Pro`,`CP`) | 159 | 106 | **273** | simultaneity-leaning |
| non-durative (`Culm`,`Pon`) | 75 | **285** | 45 | posteriority-leaning |
| telic (`Culm`,`CP`) | 92 | **325** | 107 | posteriority-leaning |
| atelic (`St`,`Pro`,`Pon`) | 142 | 66 | **211** | simultaneity-leaning |

Both halves of the hypothesis hold, in the predicted direction, for the **main**
clause. Per class the pattern is sharp: `Pro` → `Simul` 98/138 (71%), `Culm` →
`Post` 283/403 (70%), `St` → `Ant` or `Simul` 244/279 with only 35 `Post`.

For the **subordinate** clause the raw pattern is weaker, and it does not
survive controlling for the main clause. `sc_telic` splits 176/330/212 against
58/61/106 raw, but §6 shows that difference is carried by the main clause. This
is the paper's main divergence from the prior literature, and §6 is about how firmly
it can be stated.

### 3.2 Tense → a TAM bundle

`TMC` annotates the main clause tense with **32 distinct labels**, and the
Portuguese and English inventories are nearly disjoint (`PP`, `PIMP-Ind`,
`PPC-Ind`, `PresPro`… against `Pst`, `PresP`, `PstCont`…). A 32-level
categorical is not estimable at n≈200–250 per variety, and it cannot transfer
between languages at all, which would make RQ3 impossible.

So each label maps to six orthogonal, language-neutral dimensions —
`tense` (past / present / future / nonfinite, with **past** as the reference
level), plus `perfective`, `progressive`, `perfect`, `finite`, `irrealis`:

| label | Portuguese name | bundle |
|---|---|---|
| `PP` | pretérito perfeito simples | past · perfective |
| `PIMP-Ind` | pretérito imperfeito | past |
| `PMP` | pretérito mais-que-perfeito | past · perfective · perfect |
| `PPC-Ind` | pretérito perfeito composto | present · perfect |
| `Pres-Ind` | presente do indicativo | present |
| `Pres-Conj` | presente do conjuntivo | present · irrealis |
| `Cond` | condicional | future · irrealis |
| `PstPro` | *estava a fazer* (second batch; coded at ingest) | past · progressive |
| `PresPro` / `PresPro-G` | *está a fazer* (EP) / *está fazendo* (BP) | present · progressive |
| `ir(Pres)+INF-S` | *vai fazer* | future |
| `INF-C` | infinitivo composto | nonfinite · perfect |
| `GS` / `Ger-S` | gerúndio simples | nonfinite · progressive |

This mapping is a publishable artifact in its own right, and it is what makes
the English and Portuguese halves of the corpus commensurable.

**`PPC-Ind` is the case worth presenting.** European Portuguese's *pretérito
perfeito composto* (*tenho feito*, *tem vindo a aumentar*) is **not** the
English present perfect. It denotes an iterative or durative eventuality
extending up to the utterance time, which is why it is coded `present ·
perfect` rather than as a past. If the durativity → simultaneity hypothesis is
right, that bundle should force a simultaneity reading.

It does, without exception. All **14** `PPC-Ind` rows in the corpus are
`Simul` — 6 EP, 3 BP, 3 AP, 2 MP:

> PTEU200 · `PPC-Ind` · MC `Pro` · SC `CP` · **Simul**
> Globalmente, o índice de preço das casas **tem vindo a aumentar** tendo sido
> vendidos cerca de 86 335 mil imóveis.

A clean theory-predicted micro-result: the prediction follows from the
grammatical description of the form, was not fitted to the data, and holds 14/14
across all four varieties. Present it as a qualitative confirmation, and say the
n out loud.

### 3.3 Position

`Position` is `Initial` / `Medial` / `Final`, modelled as **final vs non-final**
because 92.8% of Portuguese APCs are final (875/943) and the contrast of
interest is whether the minority reads differently.

> `Initial` — PTBR93 · `Pres-Ind` · MC `CP` · SC `CP` · **Ant**
> **Tendo sido criada** por Deus, a criatura só resolve o ciclo de sua
> existência no Criador.

> `Medial` — PTMZ179 · `PP` · MC `St` · SC `CP` · **Ant**
> Excelente jogadora, **tendo contribuído** com 17 pontos e três ressaltos na
> final, Alicia Devaughn foi considerada MVP.

Lobo [lobo2003] proposed that initial APCs carry anteriority and final ones
posteriority. The corpus supports the first half **categorically** and the
second half only as a tendency:

| Portuguese APCs | Ant | Post | Simul |
|---|---|---|---|
| final (n=875) | 174 | 391 | 310 |
| **non-final (n=68)** | **60** | **0** | **8** |

**Not one non-final Portuguese APC is annotated `Post`.** Zero out of 68. As a
constraint on the grammar this is the strongest single fact in the corpus —
fronting the participial clause appears to block the posterior reading outright.

And yet position ranks only **third** in the hierarchy (§5). That is not a
contradiction, and it is the question you will be asked, so have the answer
ready: **position is nearly deterministic but rare.** It applies to 7.2% of
Portuguese rows, so removing it costs the model little held-out macro-F1, while
main-clause aspect is only *graded* but applies to every row. Importance as
measured here is contribution to predictive performance, which weights a cue by
how often it is available. Categorical strength and ranked importance are
different quantities, and the paper should report both rather than let the
ranking speak for the constraint.

### 3.4 Connector

Present in only **55 of 1,143 clauses (4.8%)**, and unevenly distributed:

| | EP | BP | AP | MP | BE |
|---|---|---|---|---|---|
| connector present | 4 (1.6%) | **34 (17.6%)** | 6 (2.4%) | 1 (0.4%) | 10 (5.0%) |
| items | *mesmo* ×4 | *mesmo* ×33, *porém* ×1 | *mesmo* ×6 | *mesmo* ×1 | *despite* ×5, *after* ×4, *although* ×1 |

Modelled as presence/absence, never as a lexical item — there is no n for
lexical modelling. In practice the Portuguese connector cue *is* **`mesmo`**
('even'), 44 of 45 occurrences, and it is concessive:

> PTBR70 · `Initial` · `PP` · MC `Pro` · SC `Culm` · **Ant**
> **Mesmo tendo dado** 4 estrelas para a proteção de adultos, o Latin NCap
> criticou a estrutura do Duster.

Across the four Portuguese varieties, `mesmo` goes **37 `Ant`, 7 `Simul`, 0
`Post`** — the same asymmetry position shows. And that is not a coincidence:
23 of its 44 occurrences are non-final, against a 7.2% base rate. **`mesmo` and
fronting travel together**, so the connector's apparently large coefficient
(+0.52 → `Ant`, −0.76 → `Post`) is partly position wearing a connector's hat.
This is why the connector block's ablation interval straddles zero (§6) even
though its coefficient looks decisive: drop it and position absorbs the work.

BP is nonetheless the one variety where the connector does measurable work, and
the one variety where connectors are common (17.6% against 0.4–2.4%) — the
single place where BP's divergence has a visible mechanism rather than just a
different base rate. Any conclusion about EP, AP or MP connectors must be
phrased **"too rare to estimate here"** (4, 6 and 1 clauses) and never "connectors do not matter".

### 3.5 The interaction terms

`both_telic = mc_telic × sc_telic` and `both_durative = mc_durative ×
sc_durative` exist because "the aspectual configuration holds **in both
clauses**" is an *interaction* claim, not two main effects. An additive model
cannot express it, so if the claim were left implicit the model would be unable
to confirm or refute it and the silence would be uninformative.

Building them explicitly has a consequence that propagates through the whole
analysis: they are **deterministic functions of other columns**, so any
procedure that perturbs inputs must recompute them or it evaluates the model on
rows that cannot exist (`both_telic = 1` while `mc_telic = 0`). That is why the
`interaction` block is measured by ablation only, and why SHAP is run
path-dependent (§6).

---

## 4. Why none of this can be read off a crosstab

The marginal association of each cue with `TR`, as bias-corrected Cramér's V
[bergsma2013] (the `Descriptives` panel):

| cue | V | χ² | dof | n |
|---|---|---|---|---|
| `ATMC` main-clause aspect | **0.439** | 370.0 | 8 | 943 |
| `TMC` main-clause tense | 0.403 | 351.8 | 46 | 943 |
| `Position` | 0.288 | 160.0 | 4 | 943 |
| `CNT` connector | 0.216 | 92.1 | 4 | 943 |
| `ATSC` subordinate-clause aspect | 0.173 | 62.2 | 6 | 943 |

The bias correction matters: `TMC` has 72 cells against `Position`'s 9, and
uncorrected V rewards high cardinality, which would put tense at the top for
the wrong reason.

But even corrected, this table cannot answer RQ1, because **the cues are
collinear and the marginal view confounds them**. The cleanest demonstration is
in the corpus. Hold the main clause aspectual class fixed at `St` — stative,
durative, atelic — and vary only the tense:

| stative main clause | Ant | Post | Simul |
|---|---|---|---|
| with `Pres-Ind` (present) | **93** | 3 | 31 |
| with `PP` (past perfective) | 17 | 21 | **54** |
| with `PIMP-Ind` (past imperfective) | 6 | 9 | **16** |

Same aspectual class. Opposite reading. A stative main clause in the present
goes anterior 73% of the time; the same stative class in the simple past goes
simultaneous 59% of the time. Whatever `ATMC`'s marginal V of 0.439 is
measuring, part of it is tense.

> PTEU81 · MC `St` · `Pres-Ind` → **Ant**
> Este arguido **está** atualmente em fuga, tendo sido condenado a pena de prisão.

> PTEU15 · MC `St` · `PP` → **Simul**
> **Foram** consideradas 67 freguesias, tendo os dados sido recolhidos entre
> abril e setembro deste ano.

This is the methodological justification for the whole paper: the prior
literature's claims are qualitative and marginal, and separating the cues
requires a multivariate model with honest uncertainty. The contribution is
quantification, not discovery.

---

## 5. What each model actually does with the cues

The ladder is ordered so each rung answers "does the next one earn its place?".
Out-of-fold macro-F1 over the 943 Portuguese rows, grouped by sentence, with
intervals from resampling sentence groups:

| | model | macro-F1 | 95% CI | what it can express |
|---|---|---|---|---|
| **B0** | majority class | 0.195 | [0.184, 0.207] | the base rate, nothing else |
| **B1** | one-rule | 0.451 | [0.430, 0.471] | one cue, one lookup table |
| **B3** | multinomial logit | **0.705** | [0.673, 0.736] | every cue, additively, signed |
| **B4** | decision tree (d=4) | 0.629 | [0.594, 0.663] | conjunctions of cues, hard splits |
| **B5** | random forest | 0.700 | [0.668, 0.730] | arbitrary conjunctions, averaged |
| **B5** | XGBoost | 0.701 | [0.670, 0.732] | arbitrary conjunctions, boosted |
| — | *circular (`DR`+`SR-SC`)* | *0.750* | *[0.718, 0.782]* | *the answer, restated* |
| — | *circular control: B3 on the same 793 rows* | *0.736* | *[0.700, 0.766]* | *the fair comparison for the row above* |

**B0 majority** predicts `Post` for everything, because `Post` is 391/943. Its
macro-F1 is 0.195, not the 41% accuracy you might expect, because macro-F1
averages per-class F1 over a fixed label set and a single-class predictor scores
zero on the two classes it never emits. That is the point of using macro-F1
here: it refuses to reward a model for exploiting a base rate, which matters
enormously for BE at 95% anterior.

**B1 one-rule** [holte1993] searches all 17 features, builds a value → majority
class lookup for each, and keeps the best on training accuracy. It selects
**`mc_durative`**, with the rule *durative → `Simul`, non-durative → `Post`*
(59.2% training accuracy, macro-F1 0.451). This is the single most informative
cue in the inventory, and the rule it learns *is* Silvano et al.'s durativity
hypothesis in its crudest possible form. That the crudest form already more
than doubles the floor is a real result. What it cannot express is any
qualification — including the large one that a present-tense main clause
overrides the durativity reading entirely.

Its role on the ladder is as the honest floor for a *hierarchy* claim: a
hierarchy is only worth reporting if the full cue inventory beats the best
single cue, which it does (0.705 vs 0.451, non-overlapping intervals).

**B3 multinomial logit** is the primary inferential model. It fits one signed
weight per cue per reading on standardised features, and predictions are a
softmax over the additive sums. The output is directly readable as a statement
about direction and magnitude, which is what the `Coefficients` panel shows:

| feature | → Ant | → Post | → Simul | reading |
|---|---|---|---|---|
| `pos_nonfinal` | **+0.77** | **−0.87** | +0.10 | fronting pushes to anterior, blocks posterior |
| `mc_durative` | −0.11 | −0.42 | **+0.52** | durative main clause → simultaneity |
| `mc_telic` | +0.39 | +0.05 | **−0.44** | telic main clause → away from simultaneity |
| `tense_present` | **+0.61** | −0.52 | −0.09 | present main clause → anterior |
| `mc_perfect` | **−0.42** | +0.20 | +0.21 | perfect main clause → away from anterior |
| `has_connector` | +0.52 | −0.76 | +0.24 | (BP's concessive *mesmo*, n=45) |
| `sc_durative` | +0.33 | −0.27 | −0.06 | weak, and not necessary (§6) |
| `both_telic` | −0.06 | 0.00 | +0.05 | the "both clauses" claim, near zero |

Every sign predicted by the prior literature comes out as predicted. What the
logit *cannot* do is express a cue whose effect depends on another cue, except
for the two interactions built by hand — which is precisely why B4 and B5 are
on the ladder.

**B4 decision tree** recursively partitions on single features, so conjunctions
are native to it. At depth 3 the fitted tree is short enough to quote verbatim,
and it is worth quoting because it **independently recovers the top three ranks
of the hierarchy** — durativity, then position, then tense — without being told
the ranking:

```
mc_durative = 0  (non-durative main clause)               n=405
├── final position                                        n=381
│   ├── mc_perfective = 0                                 n= 28  →  Post  ( 14/28 )
│   └── mc_perfective = 1                                 n=353  →  Post  (271/353)
└── non-final                                             n= 24  →  Ant   ( 22/24 )
mc_durative = 1  (durative main clause)                   n=538
├── tense ≠ present                                       n=371
│   ├── final position                                    n=348  →  Simul (211/348)
│   └── non-final                                         n= 23  →  Ant   ( 19/23 )
└── tense = present                                       n=167
    ├── mc_dynamic = 0                                    n=147  →  Ant   (101/147)
    └── mc_dynamic = 1                                    n= 20  →  Simul ( 16/20 )
```

Read as a grammar: *non-durative main clause → posterior*; *durative main clause
→ simultaneous*; *unless the participial clause is fronted, or the main clause is
present tense and not dynamic, in which case anterior*. The fronting branches are
the categorical constraint from §3.3 — 22/24 and 19/23 anterior, with zero
posterior — showing up as its own leaf.

**Look at what the tree does with the participial clause: nothing.** No
subordinate-clause feature appears anywhere in it, and the one split whose two
children predict the same class (`mc_perfective`) is a main-clause split. That
is the §6 ablation result appearing in a completely different model class, and
it is among the most legible pieces of evidence for the paper's headline claim.

Note the tree scores *worse* than the logit (0.629 at depth 4). It spends its
capacity making hard partitions where the evidence is graded. It is on the
ladder to be read, not to win, and the viewer's tree is fit at depth 3 —
shallower than the scored rung — for exactly that reason.

**B5 forest** [breiman2001] and **B5 XGBoost** [chen2016] can represent
arbitrary cue interactions. Neither beats the additive logit: 0.700 and 0.701
against 0.705, with intervals that almost coincide.

**This tie is a substantive linguistic finding, not a null result.** Two models
with the capacity to exploit cue interactions, given a deliberately interaction-
friendly feature set, extract nothing the additive model missed. The cues
combine additively. That is what licenses reporting signed coefficients as the
paper's primary result instead of hedging behind a black box, and it is the
strongest available evidence that "in both clauses" is not doing work.

**The `circular` row** is allowed `DR` and `SR-SC` and reaches 0.750. Read it
two ways. As a diagnostic it confirms the leakage is definitional. As a result
it says the honest cues (0.736 on the same rows) recover essentially everything the
discourse-relation annotation encodes about temporal order — the two intervals
overlap heavily. The temporo-aspectual inventory is nearly sufficient for what
a discourse annotator concluded.

### Two rungs that do not exist yet

**B6 — pooled hierarchical multinomial** is the confirmatory instrument for
RQ2, and the distinction it draws is linguistic, not statistical. Fitting
`TR ~ cues * variety` with partial pooling separates:

- **varying intercepts** = base-rate differences. BP being anteriority-dominant
  is a fact about how often each reading occurs. Already visible in §2.
- **varying slopes** = different cue *weights*. Whether BP's speakers weight
  position or connectors differently *given the same cues* is what RQ2 actually
  asks.

Five separate per-variety models — which is what §8's `Per variety` panel shows
— cannot separate these, and at n=193–250 they visibly overfit. Base-rate
differences masquerading as cue-weight differences is the specific error the
hierarchical model exists to prevent.

**B7 — leave-one-variety-out transfer** answers RQ3. It is only meaningful
because the TAM bundle made the varieties commensurable; with raw `TMC` labels
there is no shared feature space between EP and BE at all.

---

## 6. The three importance measures, and why they disagree

Three numbers in the viewer look like they measure the same thing. They do not,
and the differences are interpretable.

**Refit ablation — necessity.** Drop a cue block, refit, measure the fall in
out-of-fold macro-F1. Because the model is refit, surviving cues are allowed to
compensate. The question is *"could the grammar do without this cue?"* This is
the primary measure.

**Block permutation — reliance.** Keep the fitted model, shuffle the block in
the held-out fold. No compensation is possible, so the numbers are larger. The
question is *"does this fitted model lean on this cue?"*

**SHAP — credit.** Per-sentence attribution within one fitted ensemble
[lundberg2017; lundberg2020]. The question is *"for this sentence, which cue
moved the prediction?"*

The pooled Portuguese result (`Cue hierarchy` panel):

| rank | cue block | ablation | 95% CI | permutation |
|---|---|---|---|---|
| 1 | **main-clause aspect** | 0.074 | [0.048, 0.099] | 0.158 |
| 2 | **main-clause tense** | 0.045 | [0.018, 0.068] | 0.133 |
| 3 | clause position | 0.037 | [0.022, 0.054] | 0.067 |
| 4 | connector | 0.005 | [−0.005, 0.014] | 0.028 |
| 5 | telicity/durativity interactions | −0.004 | [−0.012, 0.003] | *n/a — derived* |
| 6 | **subordinate-clause aspect** | −0.008 | [−0.022, 0.008] | 0.015 |

Ablation and permutation agree on rank order for every block they share. That
is the robustness check, and it is a real one: one measure holds the model fixed
and corrupts its input, the other refits without the cue entirely.

**Three intervals cross zero, and you should say so unprompted.** The connector
block [−0.005, 0.014], the interaction block [−0.012, 0.003] and the
subordinate-clause aspect block [−0.022, 0.008] are all consistent with
contributing nothing. For the connector that is an n problem (45 clauses, 34 of
them in BP). For subordinate-clause aspect it is the finding.

**Why `aspect_sc` = −0.008 is a strong claim rather than a weak measurement.**
Dropping the `aspect_sc` block also drops `both_telic` and `both_durative`,
because they are products with an `sc_` factor and would otherwise dangle as
orphans (`interpret._orphaned_derived`). So the ablated model loses *every*
channel through which subordinate-clause aspect could reach it, including the
interaction terms — and scores no worse. The subordinate clause's own aspectual
class is not merely weak; it is dispensable.

**Where SHAP disagrees, and what to say.** In the `SHAP` panel, XGBoost's
mean |SHAP| for `sc_durative` on the `Post` class is **0.283** — its third-largest
per-feature value for that class, after `tense_present` (0.409) and `mc_durative`
(0.357). Ablation says the whole `aspect_sc` block is worth −0.008. Both are correct, and the resolution is collinearity:

- `sc_durative` correlates 0.73 with `both_durative` and 0.20 with
  `mc_durative`. The fitted XGBoost genuinely routes predictions through it.
- Refit without it, and the model reconstructs the same performance from the
  correlated main-clause features.

**Used is not the same as needed.** SHAP measures the first; ablation measures
the second. Shapley values split credit among correlated features rather than
assigning it to one [aas2021], so SHAP is read here strictly as a cross-check
on *rank order* and never as a rival estimate of block importance. This is the
one place in the analysis where two panels appear to contradict each other, so
it is worth a prepared sentence.

**Three methodological choices behind these numbers**, each of which a reviewer
may probe:

- **Impurity (gini) importance is never used** [strobl2007]. It is biased toward
  high-cardinality features, and `tense_mc` expands to seven columns against
  `position`'s one.
- **SHAP runs path-dependent, not interventional.** The interventional estimator
  marginalises features independently and would score rows with `both_telic = 1`
  and `mc_telic = 0` — impossible by construction. Unrestricted permutation of
  correlated features forces extrapolation off the data manifold [hooker2021];
  the path-dependent estimator follows the splits the fitted trees took.
- **Intervals come from resampling sentence groups, not CV folds.** Repeats
  differ only by fold seed on identical rows, so quantiles over folds estimate
  the variance of the randomisation and come out far too narrow — the
  small-sample failure Varoquaux describes [varoquaux2018]. Fold spread is
  reported separately as `sd_fold` and is not an error bar.

**SHAP figures are not out-of-fold.** They describe models fit on the whole
analysis set, so they characterise a fitted model, not generalisation, and they
are not comparable with the macro-F1 numbers. Forest and XGBoost values are not
comparable with *each other* either: a forest's leaves hold class probabilities,
XGBoost's hold log-odds increments, which is why XGBoost's numbers are an order
of magnitude larger. Compare rankings within a model, never magnitudes across.

---

## 7. The encoder models (not yet built)

`src/dripps/neural/` is empty. This section is the design, so the linguistic
reasoning is on record before any training run.

**What changes when a model reads the sentence.** The feature models see 17
binary numbers. An encoder sees the sentence: subword tokens, contextual
vectors, and therefore everything the annotation throws away —

- **explicit temporal adverbials.** `de seguida`, `em março`, `entre abril e
  setembro`, `Em 1730`, `De 1993 a 1995`. Look again at PTEU139 in §2: *tendo
  sido um deles capturado **de seguida***. A human reads the posterior relation
  straight off `de seguida`. The feature model never sees it and must infer the
  relation from `Culm` + `PP` + `Final`.
- **lexical semantics of the verbs**, not just their aspectual class.
- **the rest of the sentence**, including material outside both clauses.

This makes the comparison to B3 a **test of the annotation scheme's
completeness**, not a leaderboard. That framing is the whole reason to run it.

**T1 — text vs. inventory.** Fine-tune one multilingual encoder — XLM-R base
[conneau2020xlmr] or mDeBERTa-v3-base [he2023debertav3] — on sentence → `TR`,
pooled over the four Portuguese varieties, `StratifiedGroupKFold` on
`sentence_group`, ≥5 seeds, intervals over seeds. The model must be
*multilingual*: it is what puts EP, BP, AP, MP and BE representations in one
space, and without that the leave-one-variety-out transfer in T4 has no shared
feature space to transfer through. Both outcomes are publishable:

- **encoder ≈ B3** → the 17-feature inventory captures the available signal.
  The annotation scheme is validated, and a hand-built cue inventory matched a
  pretrained encoder on the phenomenon.
- **encoder > B3** → signal exists outside the inventory. T3 locates it.

Given that B3 already sits statistically level with XGBoost, and that the
honest cues already recover what the discourse annotation encodes (§5), the
first outcome is the one to expect — but the adverbial channel is a real
advantage the encoder has and 943 clauses is a small fine-tuning set, so it
is genuinely open.

**T3 — causal ablation. The primary interpretability experiment**, because each
perturbation is a direct test of a claim made elsewhere in the paper:

| perturbation | tests | corpus prediction |
|---|---|---|
| swap main-clause tense morphology `PP` → `Pres-Ind` | the `tense_present` → `Ant` coefficient (+0.61) | predicted distribution shifts toward `Ant`; the stative crosstab in §4 says 93/127 vs 54/92 |
| mask the participle vs. mask the main-clause finite verb | **the paper's headline claim** — that the main clause carries the reading | damage should be markedly asymmetric, worse for the main verb |
| mask explicit temporal adverbials | how much of any encoder edge comes from outside the inventory | removes the channel the feature model never had |
| move the APC from final to initial | the categorical constraint of §3.3 | `Post` probability should collapse toward zero |

The second row is the one to foreground. The feature-model version of that claim
depends on the `ATSC` annotation being right; the neural version does not depend
on the annotation at all, so it is a genuinely independent test of the same
hypothesis. Agreement would be the strongest result in the paper.

All of these need the **APC span**, which is not annotated (§1). Span recovery
from `tendo` / `having` is unambiguous for 895 of 943 Portuguese rows (§1). Report that accuracy as a
limitation of T3, not as a footnote.

**T2 — attention, secondary and descriptive only.** Report attention mass from
the pooling position to cue-bearing tokens (main-clause finite verb vs.
participle, located via the `tendo` heuristic). Expected to converge with the
rest: attention concentrating on the main-clause verb rather than the
participle, echoing `ATMC` ≫ `ATSC`. Frame as *converging* evidence and cite the
critique directly — attention weights are not faithful explanations
[jain2019attention], the rebuttal narrows but does not remove the objection
[wiegreffe2019attention], and saliency methods are the better tool for the job
[bastings2020elephant]. The causal ablations carry the interpretive weight; T2
is a picture.

**T4 — neural leave-one-variety-out**, complementing B7.

**One confound to control before interpreting anything.** The encoder sees
adverbials the feature model never gets. Any encoder advantage must be
decomposed before it is read as "cues outside the inventory" — the
adverbial-masking ablation exists for exactly this, and it must be run before
T1's result is interpreted, not after.

---

## 8. Reading the viewer, panel by panel

`results/viewer.html` (`make viewer`). Eight sections. For each: what it shows,
the sentence to say, and the trap.

### `Cue hierarchy` — RQ1
Paired bars, ablation against permutation, with bootstrap whiskers.
**Say:** main-clause aspect, then main-clause tense, then position; the two
measures agree on order; subordinate-clause aspect is dispensable.
**Trap:** three intervals cross zero (connector, interactions, `aspect_sc`). Volunteer it. And
the interaction block has no permutation value *by construction*, not because
it failed to compute — that blank is a design decision.

### `Model ladder` — B0–B5
Out-of-fold macro-F1 per rung with intervals.
**Say:** the additive logit ties the two ensembles, so the cues combine
additively, so coefficients are the right primary result.
**Trap:** the `circular` row is a leakage demonstration, not a competitor. If
it gets read as "our best model", the framing of the whole results section
collapses. Say what it is before anyone asks.

### `Decision tree` — d=3
The fitted depth-3 tree as a diagram; clicking a leaf lists the sentences that
land in it.
**Say:** it recovers the top three cues independently, and reads as three
grammatical rules (§5).
**Trap:** this is depth 3; the ladder's tree is depth 4 and scores 0.629. Two
different fits, deliberately. And the tree is *not* the paper's model — it is
the readable illustration of the logit's story.

### `SHAP` — B5
Mean |SHAP| per feature per class, plus per-sentence attribution.
**Say:** a non-linear model's per-sentence credit assignment agrees on the
ranking of the top blocks.
**Trap:** the `sc_durative` disagreement (§6) — used ≠ needed. Also: not
out-of-fold, and forest/XGBoost magnitudes are not comparable with each other.

### `Per variety` — RQ2
Per-variety ablation importances and a rank matrix.

| cue block | EP | BP | AP | MP | *BE* |
|---|---|---|---|---|---|
| main-clause aspect | **1** | **1** | 2 | 2 | *3* |
| main-clause tense | 2 | 4 | **1** | 4 | *4* |
| position | 3 | 2 | 3 | **1** | *1* |
| connector | 4 | 3 | 5 | 5 | *2* |
| interaction | 5 | 5 | 6 | 3 | *4* |
| subordinate-clause aspect | 6 | 6 | 4 | 6 | *4* |

**Say:** three differences stand out in the ablation pass (per-variety permutation
is not computed, so they are not cross-checked) — MP barely uses tense (importance
0.003, rank 4) and leans on position (rank 1, importance 0.118); BP is the only
variety where the connector does work (0.026), and the only one where connectors
are common; subordinate-clause aspect ranks 4th to 6th everywhere (4th only in AP)
and is non-positive in three of four varieties (AP is +0.008), so the pooled result
is not an artefact of pooling.
**Trap, and it is a big one:** these are five independent fits at n=193–250 and
they are visibly unstable — half the blocks show *negative* importance in at least
one variety, which is the signature of overfitting, not of a harmful cue. These
numbers establish that a difference exists and is worth modelling. They do not
license a claim about any one variety's hierarchy. That is B6's job (§5).
**Second trap:** the BE column is present because the script iterates all five
varieties. BE macro-F1 is 0.339 against a 0.325 majority floor and every
importance is ≈0 or negative — at 95% anterior there is no variance to explain.
BE is reference, not a fifth peer.

### `Coefficients` — B3
Signed standardised logit coefficients, feature × class.
**Say:** the direction table in §5; every sign matches the prior literature's
prediction.
**Trap:** these are standardised, so magnitudes are comparable across features
but are not probabilities or odds ratios. And single columns are not blocks —
`tense_mc`'s claim lives across seven rows here.

### `Descriptives` — 01
Marginal Cramér's V per cue, and the one-rule rule.
**Say:** this is the confounded view, shown deliberately, and §4 is why it
cannot answer RQ1.
**Trap:** the V ordering happens to match the multivariate ordering for the top
cue. Do not let that be read as the marginal view being adequate — the stative
crosstab in §4 is the counterexample to have ready.

### `Methods`
`docs/methods.md` rendered, with the reference list from `docs/references.bib`.

---

## 9. What you can claim, and what you cannot

**Claim, with the evidence to hand.**

1. **In Portuguese, the temporal reading of an APC is carried by the main
   clause, not the participial clause.** Main-clause aspect is the top cue
   (ablation 0.074 [0.048, 0.099]); subordinate-clause aspect is dispensable
   (−0.008 [−0.022, 0.008], with the interaction terms dropped alongside it).
2. **The direction of every effect matches the prior literature's prediction.**
   Durative main clause → simultaneity (+0.52); telic main clause → away from
   simultaneity (−0.44); non-final position → anterior (+0.77) and never
   posterior (0/68).
3. **`PPC-Ind` forces simultaneity, 14/14**, as the grammatical description of
   EP's *pretérito perfeito composto* predicts.
4. **The cues combine additively.** Two ensembles with interaction capacity tie
   the additive logit.
5. **The honest cues recover what the discourse annotation encodes about
   temporal order** — 0.736 against the circular model's 0.750 on the same
   793 rows, intervals overlapping.
6. **The four Portuguese varieties differ**, and the difference is worth
   modelling.

**Do not claim.**

- *Any specific variety's hierarchy.* Five fits at n=193–250, unstable, negative
  importances. Wait for B6.
- *That base rates and cue weights differ.* Only B6 separates them.
- *That connectors do not matter in EP, AP or MP.* n = 4, 6, 1. Say "too rare
  to estimate here".
- *Anything about English cue weights.* Unidentifiable at 95% one class.
- *That the SHAP ranking is a second estimate of block importance.* It is a
  cross-check on rank order.
- *Generalisation beyond newswire.* Single genre, written, no spoken data.
- *Reliability of the target.* DRIPPS reports Cohen's κ **for the discourse
  relation only** — EP .959, BP .900, BE .881, MP .676, **AP .551**
  (Table 2 of [silvano2023dripps]). The paper states that agreement on the other
  annotated variables was not computed, on the grounds that their classification
  is clear-cut. **So there is no inter-annotator agreement figure for `TR`, the
  dependent variable of this study.** A reviewer will ask; the strongest
  mitigation is a second annotator re-coding a sample for `TR`. The 150-clause second batch has no agreement figure at all. AP's moderate
  `DR` agreement is also a live confound for RQ2, because noisier annotation can
  masquerade as different cue weights.

**The framing to lead with.** Silvano et al. [silvano2021apc] already claim
qualitatively that position and aspectual class drive the temporal
interpretation, that telicity tracks anteriority/posteriority and durativity
tracks simultaneity, that English is near-categorically anterior, and that
BP diverges from EP/AP/MP. The contribution here is **quantification, not
discovery**: a multivariate, uncertainty-aware hierarchy; the finding that the
two clauses are not equal partners; and a neural counterpart testing whether
the hand-built inventory is complete. State that in the Introduction or it reads
as a re-run of prior work.
