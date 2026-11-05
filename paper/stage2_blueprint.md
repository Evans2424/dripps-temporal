# Stage 2 Paper Blueprint (plan mode): outline, storyline, result selection

**Status:** proposal for agreement; no section is drafted yet. **Date:** 2026-11-05.

## Paper Configuration Record

| item | value |
|---|---|
| Type | ACL long paper: 8 pages of content. Limitations, Ethics, references and appendix don't count |
| Venue / cycle | ACL Rolling Review, January cycle; anonymous review version (`\usepackage[review]{acl}`) |
| Template | `NLP_for_linguistics/acl_latex.tex` (official ACL style) |
| Citation style | ACL natbib (`acl_natbib.bst`), author–year |
| Language | English |
| Bibliography | `docs/references.bib` + `paper/stage1_new_refs.bib` → merged into `NLP_for_linguistics/custom.bib` at drafting |
| Emphasis | computational methods and results; linguistic background and conclusions brief |
| Pending experiments | fully described, with placeholder tables marked **[PENDING: …]** |

## Working title

**Which Cues Matter? Computational Evidence on Temporal Interpretation in Portuguese Perfect Participial Clauses**
(Alternatives to decide at the end: "*…from Annotated Cues, Encoders and Zero-Shot LLMs*".)

## Storyline (the argument in six moves)

1. **Puzzle.** A Portuguese adverbial perfect participial clause (*tendo* + participle) is morphologically a perfect,
   yet in our corpus it reads as **posterior 42%** of the time and anterior only 25%. Its temporal reading is not in
   its form, so where is it?
2. **Hierarchy from annotated cues (RQ1).** An additive logit on 17 expert-coded cues reaches macro-F1 0.695.
   Interventions (refit ablation, permutation) rank **main-clause aspect > main-clause tense > position**, and
   **participial-clause aspect is dispensable**. Non-linear models agree and gain nothing, so the cues combine
   additively.
3. **Does the text add anything a model can learn? Encoders say no.** Four fine-tuned encoders, including two
   Portuguese ones, at best *match* the cues once the two verbs are marked, and never add to them.
4. **…but zero-shot LLMs do.** With option-order debiasing and calibration, open (Qwen 27B) and closed (Jev) LLMs
   reach the cue model's level, and **averaged with it they add +0.04 macro-F1**, the only systems that do. Small LLMs
   collapse to "Before", a label prior.
5. **What do they use?** LLMs rely on the cues differently from the corpus: *higher strength* for clause order,
   *lower* for main-clause aspect. They separate clauses that the cue inventory makes identical. Candidate sources:
   verb meaning and the timeline. **[PENDING: robustness tests]**
6. **Varieties (RQ2).** EP, BP, AP and MP differ in base rates, not in demonstrable cue weights. **[PENDING:]** do
   variety-matched models (PT-BR- against PT-PT-trained) do better on their own variety, and how do the African
   varieties fare?

## Contributions (for the introduction)

1. An interventional cue hierarchy for an underdetermined temporal construction, converging across classical
   models, fine-tuned encoders and zero-shot LLMs.
2. Evidence that zero-shot LLMs, with controls for option-order bias and calibration, carry temporal information
   beyond an expert cue inventory, which supervised encoders trained on the same data do not recover. Includes a
   matched test of decision tuning (OpenJev against its base model) **[PENDING]**.
3. A variety-aware evaluation: pooled hierarchical tests across four varieties, and a training-variety × test-variety
   design for Portuguese LLMs **[PENDING]**.

## Outline and page budget (8 pp)

| § | Section | pp | Content | Results used |
|---|---|---|---|---|
| — | Abstract | — | about 180 words; the six moves | — |
| 1 | Introduction | 0.9 | puzzle + one contrasting example pair; why it is a computational question; RQ1, RQ2; contributions | label distribution |
| 2 | Related work | 0.6 | temporal relations (TimeBankPT, aspect as a feature, MATRES, TRAM); LLMs as annotators and informants, option-order bias, calibration; probing against interventions; Portuguese LLMs and variety bias (P3B3, ALBA, AMALIA, African-variety adaptation) | — |
| 3 | Data and task | 0.6 | DRIPPS: 1,038 clauses, 4 varieties (n per variety), 3 readings, 17 cues in 6 blocks; leakage control (DR and SR-SC excluded; 0.750 against 0.736 in one sentence); sentence-grouped 5-fold CV; group bootstrap | Table 1 (corpus) |
| 4 | Models and probes | 1.6 | 4.1 cue models (B0–B5) + refit ablation / permutation; 4.2 encoders (plain against marked; four checkpoints); 4.3 zero-shot decisions: single forward pass, 3 cyclic orders + geometric mean, Dirichlet calibration on training folds, fusion w = 0.5 fixed in advance; open models (Qwen 4B–27B), Jev, OpenJev **[PENDING]**; 4.4 probes: masking against random-word controls, fronting, cue-strength refit, identical-cue test; 4.5 varieties: B6 nested model; variety-matched LLM design **[PENDING]** | Figure 1 (pipeline schematic) |
| 5.1 | Cue hierarchy | 0.7 | ranks and intervals; additivity (logit ≈ XGBoost); participial aspect dispensable, with the artefact checks (it varies; r ≤ 0.21 with main aspect) | Table 2 |
| 5.2 | Encoders | 0.4 | plain below B3 (3 of 4 checkpoints), marked ≈ B3, no fusion gain | Table 3 (compact) |
| 5.3 | Zero-shot LLMs | 0.8 | raw collapse of small models → scale; calibration; fusion gains; Jev raw lead, calibrated parity; OpenJev against base **[PENDING]** | Table 4 |
| 5.4 | Beyond the inventory | 0.6 | cue strength against validity; fronting shift; masking contrasts; identical-cue test **[PENDING intervals]** | Figure 2 (cue strength against validity) |
| 5.5 | Varieties | 0.6 | B6: base rates differ (p = .003), slopes not localisable; variety-matched LLMs **[PENDING]** | Table 5 (B6 + placeholder model × variety grid) |
| 6 | Discussion | 0.4 | what is learnable from 1k examples against what pretraining brings; cue validity ≠ cue strength; implications for annotation schemes | — |
| — | Limitations | (free) | no annotator agreement for later batches; n per variety; closed model; perturbation label validity; newswire | — |
| — | Ethics | (free) | data, licences (CC BY-NC OpenJev), API data handling, AI-assistance disclosure | — |
| — | Appendix | (free) | encoder variants (_feat/_res/_txt/_aux); SHAP; calibration details; prompts; hyperparameters; per-variety refits; BE contrast | — |

## Result selection

**In the main text:**
- B3 and the model ladder (as a row group);
- the cue-block ablation with intervals, and permutation;
- the four encoder checkpoints, plain and marked against B3;
- all T5 models (raw, calibrated, B3 + model);
- the order-agreement and cue-strength diagnostics;
- the fronting and masking contrasts;
- the B6 tests.

**Appendix only:**
- the round-3 encoder variants (`_res`, `_txt`, `_aux`, `_feat`), which are exploratory;
- the SHAP tables;
- the post hoc fusion weights (0.25, 0.75);
- the per-variety refits (descriptive, unstable);
- the leakage-demo numbers;
- the Qwen3.6 against 3.8 release comparison (one sentence in the main text);
- British English.

**Excluded:**
- the 0.8B smoke-test numbers;
- the first under-trained encoder run (0.469).

## Placeholders (to fill when the runs finish)

| id | where | content |
|---|---|---|
| P1 | Table 4 + §5.3 | OpenJev 27B (shim) and Qwen3.8-27B (shim): H1–H3 of `docs/OPENJEV_PLAN.md` |
| P2 | §5.4 | identical-cue AUC with a permutation null; incremental validity (B3 + LLM log-probs) |
| P3 | §5.4 | valid LLM masking (deletion / *algo*); relabelled fronting sample |
| P4 | Table 5 + §5.5 | variety × model grid: macro-F1 per variety for variety-paired and variety-targeted models (below) |
| P5 | §3 / Limitations | inter-annotator agreement on a double-annotated batch |

## Variety experiment design (P4), for agreement

**Models.** Same zero-shot decision pipeline; generative models only, plus the paired encoders:
- **Variety-paired:** Gervásio PT-PT against Gervásio PT-BR (same architecture and size), and Albertina PT-PT against
  PT-BR (encoders, fine-tuned).
- **Variety-targeted:** AMALIA (PT-PT), and Tucano or Bode (PT-BR).
- **Multilingual controls:** Qwen3.8-27B (done), EuroLLM-9B, Salamandra-7B.

**Hypothesis H-var.** Macro-F1 is higher when training variety = test variety: PT-PT models on EP, PT-BR models on BP.
- On AP and MP, PT-PT models do better, following Alves 2024.
- Tested as a model × variety interaction (pooled logit on correctness, group bootstrap), since per-variety n is
  only 193–345.

## Questions to settle before drafting (see the chat)

1. Jev's place: main text next to Qwen, or a secondary comparison?
2. The variety model set above: add or remove anything? (Sabiá is access-restricted.)
3. How central "beyond the inventory" (move 5) should be, given that its confirmatory tests are pending.
