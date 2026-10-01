# DRIPPS × COLING 2027 — Project Plan

## Context

A COLING 2027 **long paper (8 pages)** using the DRIPPS corpus (993 annotated adverbial perfect participial clauses, five varieties) to answer:

- **RQ1** — Which linguistic cues drive the temporal interpretation (anteriority / posteriority / simultaneity) of adverbial perfect participial clauses (APC), and what is their *relative* contribution? → cue hierarchies.
- **RQ2** — Do the four Portuguese varieties (EP, BP, AP, MP) weight the *same* cue inventory differently? → shared inventory, different hierarchies.
- **RQ3 (coda)** — Does a model learned on one variety transfer to the others?
- **RQ4** — Deferred to a follow-up paper.

**Venue.** ARR deadline **12 October 2026 — 22 days out**, and the *last* cycle eligible for COLING 2027. Long papers: 8 pages body, mandatory Limitations section, unlimited references and appendix. The COLING 2027 special theme (**"NLP for Linguistics"** — linguistic diversity, under-researched languages and phenomena) fits precisely, but is selected *at commitment, after meta-review*; the ARR submission picks a standard track. Best fit: *Discourse and Pragmatics*, *Linguistic Theories and Language Modeling for Linguistic Insights*, or *Language Diversity and NLP for Low-resourced Languages*.

**The scientific gap.** Silvano et al. (2021), summarised in the DRIPPS paper, already claims *qualitatively*: in Portuguese the key factors are clause position + aspectual classes; anteriority/posteriority track **telic** situations, simultaneity tracks **durative** situations; English is near-categorically anterior; and AP/MP pattern with EP while BP diverges (contra the Afro-Brazilian continuum hypothesis, Petter 2009). **Our contribution is quantification, not discovery**: a multivariate, uncertainty-aware cue hierarchy; a formal test of whether cue *weights* differ across varieties independently of base rates; and a neural counterpart showing whether the hand-built cue inventory is complete. State this framing explicitly in the Introduction, or reviewers will read it as a re-run of prior work.

---

## The data (audited, full corpus)

`export@20260920-10h47.csv` — 993 rows, `;`-delimited. Schema: `ID; DR; SR-SC; CNT; Position; TR; TMC; ATMC; ATSC; Sentence`. Variety is the ID prefix: `PTEU` 200, `PTAO` 200, `PTMZ` 200, `ENBE` 200, `PTBR` 193. IDs are clean and gapless. (`DRIPPS/data.csv` is a stale EP-only export — delete or archive it to avoid ambiguity.)

### Target distribution — already a headline result

| | Ant | Post | Simul |
|---|---|---|---|
| ENBE | **190 (95%)** | 2 | 8 |
| PTAO | 36 | **112** | 52 |
| PTMZ | 23 | **97** | 80 |
| PTEU | 58 | **78** | 64 |
| PTBR | **93 (48%)** | 38 | 62 |

EP/AP/MP are posteriority-dominant; **BP inverts to anteriority-dominant**; English is near-categorically anterior. This confirms both prior qualitative claims quantitatively.

### Cue availability varies by variety — this *is* RQ2 evidence

| | EP | BP | AP | MP | BE |
|---|---|---|---|---|---|
| non-final position | 6.5% | **17.1%** | 3.5% | 3.0% | **44.0%** |
| connector present | 1.0% | **17.6%** | 1.0% | 0.5% | 5.0% |
| stative main clause | 30% | **48%** | 22% | 21% | **60%** |

**BP patterns toward English on four independent dimensions** (more anterior, more initial, more connectors, more stative main clauses) while AP/MP sit at or beyond the EP end. That convergence deserves its own paragraph — described as a distributional fact, with contact-influence explanations flagged as speculation we cannot test on newswire data.

### Facts that constrain the design

1. **Leakage is definitional, not incidental.** `SR-SC` → TR is Cramér's V = 0.770 (`after` → Post 59/59; `before` → Ant 18/18 in EP). `DR` → TR is V = 0.559 (`asynchrony` never yields Simul; `conjunction`/`elaboration`/`exemplification`/`manner` are always Simul). **`DR` and `SR-SC` can never be features for `TR`.** Report once as a "circular upper bound", then exclude by assertion in code.
2. **Tense and aspect are collinear.** In EP, 32 of 37 `Pres-Ind` rows have a stative main clause; `Culm`+`PP` alone is 74/200. Relative contribution therefore **cannot** be read off marginal χ²/Cramér's V — it needs a multivariate model. This is the methodological justification for the paper. 4 of 12 ATMC×TMC cells have n<5, so full interactions are not estimable: main effects plus pre-registered interactions only.
3. **11 sentences appear more than once** (multi-APC sentences duplicated by annotators). **All CV splits must be `GroupKFold` on sentence text**, neural models included, or train/test leak.
4. **EN and PT tense vocabularies are disjoint** (`Pst/Pres/PresP/PresCont` vs `PP/Pres-Ind/PIMP-Ind/PPC-Ind/PMP/PresPro/…`; 11–16 values per variety). TAM harmonization is mandatory infrastructure for RQ3, not a nicety.
5. **`Pon` (Point) is attested twice, MP only.** Merge into the nearest class or drop; document the choice. Note the label is `Pon`, not `Point`.
6. **BE is near-single-class** (95% Ant), so a majority baseline already scores 95 and cue weights are unidentifiable. **BE is a contrastive reference throughout, not a fifth data point in RQ2.**

### Signal in EP (indicative; recompute per variety)

`ATMC` V=0.478 — strongest genuine cue. `TMC` V=0.506, inflated by 12 levels with singletons. `ATSC` V=0.154, χ²=9.5, df=6 → **not significant**. Coherent patterns: MC=`Pro` → Simul 78%; MC=`Culm` → Post 67%; MC=`Pres-Ind` → Ant 78%; MC=`PPC-Ind` → **Simul 6/6** (EP's *pretérito perfeito composto* is iterative/durative, unlike the English present perfect, so it should force simultaneity — a clean theory-predicted micro-result).

Headline refinement of prior work: **main-clause aspect and tense carry the temporal interpretation; the participial clause's own aspectual class contributes little.** Silvano et al. weight aspectual class "in both clauses"; the data says the clauses are not equal partners. Verify this holds across all four PT varieties before promoting it to a claim.

---

## Method

### Feature engineering (the linguistic core)

**Aspect → Moens & Steedman (1988) primitives**, turning 4–5 dummies into 3 interpretable binaries and directly operationalising the telicity/durativity hypothesis:

| Class | dynamic | durative | telic |
|---|---|---|---|
| State (`St`) | − | + | − |
| Process (`Pro`) | + | + | − |
| Culmination (`Culm`) | + | − | + |
| Culminated process (`CP`) | + | + | + |
| Point (`Pon`) | + | − | − |

Applied to both clauses, plus `telic_SC × telic_MC` and `durative_SC × durative_MC` — "in both clauses" *is* an interaction claim and must be tested as one.

**Tense → TAM bundles.** Collapse the 11–16 per-variety `TMC` values by Portuguese/English TAM morphology into `{tense: past/present/future, aspect: perfective/imperfective, mood: indicative/subjunctive/non-finite, periphrasis: progressive/prospective}`. E.g. `PP`→past·perfective·ind, `PIMP-Ind`→past·imperfective·ind, `PPC-Ind`→past-to-present·iterative·ind, `PresPro`→present·progressive, `ir(Pres)+INF-S`→prospective; `Pst`→past·perfective·ind, `Pres`→present·imperfective·ind. This mapping is a publishable artifact and is what makes EN↔PT transfer possible at all.

**Position** → `final` / `non-final`. **Connector** → presence/absence (now live: BP 17.6%, BE 5%).

### Model ladder

Decided on the evidence (you delegated this): **regularised multinomial logistic regression is the primary inferential model; trees are secondary, for interaction discovery and rule extraction.** With 993 rows, 3 classes and ~12 dummy parameters, a penalised logit is well-estimated and yields calibrated coefficient intervals; XGBoost will not beat it and its importances would be unstable at n≈200 per variety. Trees still earn their place — they find cue *combinations* a main-effects model misses, and they answer the question you actually asked about how splits organise.

| | Model | Purpose |
|---|---|---|
| B0 | Majority class per variety | Floor |
| B1 | OneR (single best cue) | What one cue buys you |
| B2 | Multinomial logit, raw categorical cues | Baseline |
| **B3** | **Multinomial logit, M&S primitives + TAM bundles + 2 pre-registered interactions, L2** | **Primary: RQ1** |
| B4 | Depth-limited decision tree | Readable rules |
| B5 | Random Forest / XGBoost | Interaction discovery, nonlinearity check |
| **B6** | **Hierarchical multinomial, varying intercepts *and* slopes by variety** | **Primary: RQ2** |
| B7 | Leave-one-variety-out transfer | RQ3 |

**RQ1 cue hierarchy** = ranked standardised effect sizes from B3 with bootstrap CIs, cross-checked against permutation importance from B5. Agreement is a robustness result; disagreement is a finding about collinearity.

**RQ2 is a variety × cue interaction test**, not five separate models — and it must separate two things that are easy to conflate:
- **varying intercepts** = base-rate differences (BP is anteriority-dominant), already visible in the table above;
- **varying slopes** = genuinely different cue *weights*, which is what RQ2 actually asks.

Without that separation, base-rate differences masquerade as cue-weight differences. Fit pooled `TR ~ (position + telic_SC + telic_MC + dur_SC + dur_MC + TAM + connector) * variety` with partial pooling; at n≈200 per variety, five independent models would overfit and their rankings would not be comparable. Statsmodels MNLogit + LRT is the fast path; `bambi`/`numpyro` for the Bayesian version with posterior intervals on each variety's cue weights.

### Neural component

Purpose is **not** to beat the feature model — it is to test whether the hand-built cue inventory is *complete*.

- **T1 — Text vs. inventory.** Fine-tune a single multilingual encoder (XLM-R base or mDeBERTa-v3-base, so representations are comparable across varieties) on sentence → TR, pooled over the four PT varieties, `GroupKFold` by sentence, ≥5 seeds, CIs over seeds. Compare macro-F1 to B3. *BERT ≈ B3* → the four-cue inventory captures the available signal, validating the annotation scheme. *BERT > B3* → cues exist outside the inventory, and T3 finds them. **Both outcomes are publishable**, which is what makes this experiment worth running.
- **T3 — Causal ablation (primary interpretability).** Perturb inputs and measure Δ in the predicted TR distribution: swap main-clause tense morphology (`PP`→`Pres-Ind`) and test whether predictions shift toward Ant as the corpus statistics predict; mask the participle vs. the main-clause finite verb and compare damage; mask explicit temporal adverbials (*dois dias depois*) to isolate how much of any BERT edge comes from cues absent from the annotation scheme. This is causal, maps directly onto RQ1's "relative contribution", and is far stronger than attention.
- **T2 — Attention (secondary, descriptive).** Report attention mass from the pooling position to cue-bearing tokens (main-clause finite verb vs. participle, located via `tendo`/`having` + following verb). Frame strictly as *converging* with the ablation and feature results — e.g. "attention concentrates on the main-clause verb rather than the participle", echoing ATMC ≫ ATSC. Cite Jain & Wallace (2019), Wiegreffe & Pinter (2019) and Bastings & Filippova (2020) directly; do not claim attention is explanation.
- **T4 — Neural leave-one-variety-out**, complementing B7.

Caveat to control for: BERT sees clause *content*, including explicit temporal adverbials the feature model never gets. Any BERT advantage must be decomposed before it is interpreted — that is exactly what the adverbial-masking ablation is for.

### Non-negotiable controls

- `GroupKFold` by sentence everywhere (11 duplicated sentences), neural runs included.
- `assert` that `DR`/`SR-SC` never enter a `TR` feature matrix.
- **Permutation importance, never gini** — gini is biased toward high-cardinality features and `TMC` has up to 16 levels vs `Position`'s 3. Use native categorical handling or aggregate importance across a feature's dummies.
- Repeated stratified CV (10×5) with bootstrap CIs on every reported metric; ≥5 seeds for neural runs. A single CV run at this n is noise.
- Macro-F1 and per-class F1, never bare accuracy (BE would score 95% on the majority class alone).

---

## Repository structure

```
dripps-temporal/
├── README.md
├── pyproject.toml
├── Makefile                      # data | features | models | neural | figures | paper
├── data/
│   ├── raw/export@20260920-10h47.csv   # verbatim, never edited
│   ├── interim/
│   └── processed/dripps.parquet
├── src/dripps/
│   ├── schema.py                 # canonical names, allowed values, ID-prefix → variety
│   ├── io.py                     # load + validate, fail loudly on unknown labels
│   ├── features.py               # M&S primitives, TAM bundles, position/connector collapse
│   ├── leakage.py                # guards excluding DR / SR-SC
│   ├── models.py                 # B0–B7
│   ├── evaluate.py               # GroupKFold, macro-F1, bootstrap CIs
│   ├── interpret.py              # permutation importance, SHAP, surrogate rules
│   ├── stats.py                  # χ², Cramér's V, LRT, hierarchical fit
│   └── neural/                   # finetune.py, ablate.py, attention.py
├── experiments/
│   ├── 01_audit.py  02_baselines.py  03_cue_hierarchy.py
│   ├── 04_variety_interaction.py     05_transfer_lovo.py
│   └── 06_neural.py  07_ablation.py
├── docs/
│   ├── SCIENTIFIC_PLAN.md   # RQs, hypotheses H1–H5, predictions, falsifiers
│   ├── ANNOTATION_SCHEME.md # every field/value, linguistic definitions, M&S + TAM tables
│   ├── DATA_AUDIT.md        # the audit above, regenerated per variety
│   ├── EXPERIMENTS.md       # model ladder, protocol, pre-registered decisions
│   └── RELATED_WORK.md      # Silvano 2021, Lobo 2003, Leal 2011, Móia & Viotti 2004,
│                            # ISO 24617-8, Moens & Steedman 1988, attention-critique refs
├── results/{tables,figures,models}/
└── paper/                        # ARR LaTeX template
```

Deterministic seeds; `make all` reproduces every number in the paper.

---

## Execution order

1. Scaffold repo; `schema.py` + `io.py` with strict validation. Archive the stale EP-only `data.csv`.
2. Write `ANNOTATION_SCHEME.md` and `DATA_AUDIT.md`; regenerate the audit per variety.
3. `features.py` + leakage guards. Unit tests on the M&S and TAM mappings.
4. B0–B3 → RQ1 cue hierarchy. **First checkpoint:** does ATMC ≫ ATSC hold in all four PT varieties?
5. B4–B5, permutation importance + SHAP interactions, extract readable rules.
6. **B6 pooled hierarchical model → RQ2.** The paper's centrepiece; separate varying intercepts from varying slopes.
7. T1 neural, then T3 ablation, then T2 attention. **Second checkpoint:** does BERT beat B3? The answer determines the framing of Section 6.
8. B7 + T4 transfer → RQ3.
9. Draft: Intro / Related Work / Data / Method / Results (RQ1, RQ2) / Neural analysis / RQ3 / Limitations. Figures: cue-hierarchy forest plot with CIs; per-variety cue-weight heatmap; ablation Δ-probability chart.

Feasible in 22 days because the data is hand-annotated and the modelling is light. The real risk is the neural component eating time — T1/T3 are small (993 sentences, base-size encoder, minutes per run on one GPU), but keep them behind a checkpoint so they can be cut to an appendix if step 6 slips.

## Verification

- `make all` from a clean checkout reproduces every table and figure.
- Unit tests: M&S mapping round-trips all five classes including `Pon`; TAM mapping covers every attested `TMC` value in all five varieties (fail on unseen); leakage guard raises when `DR`/`SR-SC` reaches a `TR` model; `GroupKFold` shares no sentence across folds.
- Sanity checks that must hold: B3 macro-F1 > B0 and > B1 with non-overlapping bootstrap CIs; the deliberately circular `SR-SC` model scores near ceiling (confirming the leakage diagnosis rather than hiding it); EP `PPC-Ind` → Simul reproduces 6/6; BE majority baseline reproduces ≈95%.
- Per-variety confusion matrices, not just aggregates.
- Neural: report seed variance explicitly; a result that does not survive 5 seeds is not a result.

## Limitations (mandatory section — draft these early)

- **No inter-annotator agreement exists for `TR`, our dependent variable.** DRIPPS reports Cohen's κ *for DRel only*: EP .96, BP .90, BE .88, MP .68, **AP .55 ("moderate")**. Reviewers will ask. Strongest mitigation: have a second annotator re-code ~100 sentences for `TR` and report κ — worth raising with the authors (Cordeiro is at INESC TEC).
- **AP's low DRel agreement is a live confound for RQ2**: noisier annotation can masquerade as different cue weights. Acknowledge, and probe if a reliability sample becomes available.
- n≈200 per variety; per-variety estimates depend on partial pooling. Newswire only, single genre — no claims about spoken or other registers.
- EP/AP/MP connectors are ~1% attested: conclusions are "not attested here", not "connectors do not matter".
- BE is 95% anterior, so its cue weights are unidentifiable; it is contrastive context only.
- The APC span is **not** annotated; it is heuristically recoverable via `tendo`/`having` (198/200 in EP; 8 EP sentences ambiguous with >1 `tendo`). This affects the neural ablations, which need the span — report span-recovery accuracy.
- Attention analysis is descriptive and contested; the causal ablations carry the interpretive weight.

## Open question (not blocking)

Whether to approach the DRIPPS authors as collaborators. It could unlock the unannotated pool (4222 PT + 2635 EN crawled sentences) and a `TR` reliability study — both materially strengthen the paper — but it changes authorship. Not on the critical path for 12 October.
