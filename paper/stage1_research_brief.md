# Stage 1 research brief: computational framing for "Which Cues Matter?"

**Pipeline:** academic-pipeline, Stage 1 (deep-research, quick mode). **Date:** 2026-11-05.

**Target:** ACL long paper (8 pages), January ARR cycle.

**Inputs:**
- Project docs: `docs/FINDINGS.md`, `methods.md`, `CALIBRATION.md`, `T5_ROBUSTNESS_PLAN.md`, `OPENJEV_PLAN.md`.
- Results: `results/tables/`.
- Existing bibliography: `docs/references.bib`.

**New verified references:** `paper/stage1_new_refs.bib`. 23 entries; 22 are verified and DialectBench is still to be confirmed.

## Research question brief

**RQ1 (cue hierarchy).** Which annotated cues carry the temporal reading (Ant, Post, Simul) of Portuguese adverbial
perfect participial clauses? The candidates are main-clause aspect and tense, position, connector, and participial
aspect. Do models with access to the sentence text find the same hierarchy, or information beyond it?

**RQ2 (varieties).** Do the cue–reading associations differ across EP, BP, AP and MP once the base rates of each
variety are accounted for? Does model performance depend on the match between a model's training variety and the
test variety?

**Scope.**
- *In scope:* predictive modelling, interpretability probes, zero-shot LLM decisions, and variety-conditioned
  evaluation.
- *Out of scope:* a full linguistic literature review (kept minimal on purpose), and British English beyond a
  contrastive note.

## Synthesis by theme

**1. Temporal relations in NLP.**
- Temporal relation classification grew out of TimeML resources and TempEval, and the MATRES scheme
  [ning-etal-2018-multi] raised inter-annotator agreement by annotating start points only.
- For Portuguese, TimeBankPT [costa-branco-2012-timebankpt] adapted the TempEval data, and
  [costa-branco-2012-aspectual] already showed that **aspectual type is a useful feature for temporal relation
  classification**. That is the closest computational precedent for our RQ1.
- Recent LLM benchmarks such as TRAM [wang-zhao-2024-tram] test temporal *reasoning* broadly (order, duration,
  arithmetic), and find GPT-4 still well below humans.
- **None of this work targets a single construction whose reading is underdetermined by its morphology**, which is
  our setting: a perfect form that is read as posterior 42% of the time.

**2. LLMs as classifiers, annotators and informants.**
- LLMs are increasingly used as zero-shot annotators [gilardi2023chatgpt] and as objects of metalinguistic
  analysis [begus2025large].
- Two known threats apply directly to our design:
  - *option-order or selection bias* in multiple-choice settings [zheng2024selection_verified; pezeshkpour2024order];
  - *miscalibration* [guo2017calibration; zhao2021calibrate; kull2019dirichlet] (the last three are already in
    `docs/references.bib`).
- Our pipeline handles both explicitly: cyclic option permutation with a geometric mean, and Dirichlet calibration
  on the training folds. That is a methodological contribution in its own right for LLM-as-informant studies.

**3. Probing and perturbation.**
- Probing classifiers show what is *decodable*, not what is *used* [belinkov-2022-probing]. Counterfactual or
  amnesic interventions are needed to make behavioural claims [elazar-etal-2021-amnesic].
- Our design is in that interventional tradition: masking against random-word controls, displacement, refit
  ablation, and refitting B3 on model answers (cue *strength* against corpus cue *validity*).

**4. Portuguese language models by variety.**
- **Encoders** (in the current bib): BERTimbau is PT-BR; Albertina comes as PT-PT and PT-BR variants; XLM-R and
  mDeBERTa are multilingual.
- **Generative models:**
  - *European:* Gervásio PT* (in PT-PT and PT-BR variants) [santos2024gervasio], AMALIA (PT-PT, trained with .br
    domains filtered out) [simplicio2026amalia].
  - *Brazilian:* Sabiá [pires2023sabia; access restricted], Tucano (trained from scratch, at most 2.4B)
    [correa2024tucano], Bode [garcia2024bode].
  - *Multilingual European:* EuroLLM [martins2024eurollm], Salamandra [gonzalezagirre2025salamandra].
  - *Our current runs:* the Qwen 4B–27B models and the decision models.
- **Variety-paired models exist** (Gervásio PT-PT/PT-BR; Albertina PT-PT/PT-BR). They allow a clean
  training-variety × test-variety design, which is what the planned experiment needs.

**5. Varieties and model bias.**
- Evidence that training variety matters:
  - pt-BR dominates training data, and most LLMs are biased towards it [ferreira2026p3b3].
  - European Portuguese benchmarks exist because the existing ones skew Brazilian [vieira2026alba].
  - AMALIA does better on pt-PT-specific evaluations [simplicio2026amalia].
- **For African Portuguese the evidence is thin.** One perplexity study finds Albertina PT-PT closer to African
  varieties than PT-BR, though less so in writing from Guinea-Bissau, Mozambique and São Tomé [alves-etal-2024-african].
- Cross-dialect robustness work in English [ziems-etal-2023-multi] documents systematic drops on non-standard
  varieties.
- **No work tests whether a model's training variety affects a *semantic* judgement**, such as temporal
  interpretation, across EP, BP, AP and MP.

**6. Decision models.**
- Jev [typesafe2026jev] and OpenJev [openjev2026] return one probability per option in a single pass.
- OpenJev is a decision-tuned version of Qwen3.8-27B, so the effect of decision tuning can be isolated.
- These are documentation, not peer-reviewed work: cite as such, and keep the claims to what we measure.

## The gap this paper fills

1. **A cue hierarchy for a construction with an underdetermined reading**, measured with interventions, not only
   correlations, and converging across four model families.
2. **The first test of whether zero-shot LLMs (open and closed) carry temporal information beyond an expert cue
   inventory**, with explicit controls for option-order bias and calibration.
3. **A variety-aware design.** Do cue weights, and model accuracy, depend on the variety, and on the match between a
   model's training variety and the test variety (e.g. PT-BR models on BP, PT-PT models on EP, African varieties
   with neither)?

## Implications for the design (feeds Stage 2)

- **Planned experiment E-var** (placeholder in the draft): per-variety macro-F1 and the cue-reliance refit for:
  - variety-paired models: Gervásio PT-PT/PT-BR, Albertina PT-PT/PT-BR;
  - a variety-targeted generative model per side: AMALIA (EP) and Tucano or Bode (BP);
  - multilingual controls: Qwen, EuroLLM, Salamandra.
- **Hypothesis H-var:** accuracy is higher when the training variety matches the test variety (EP or BP). For AP and
  MP, the PT-PT models are expected to transfer better, following [alves-etal-2024-african]. This is a
  model × variety interaction, tested with a group bootstrap by variety.
- **Caveat to state in the paper:** per-variety n is 193–345, so the per-variety intervals will be wide. The
  confirmatory test is the interaction in a pooled model, like B6.

## Sources consulted

- **ACL Anthology:** L12-1096, E12-1027, P18-1122, 2024.findings-acl.382, 2022.cl-1.7, 2023.acl-long.44,
  2024.propor-1.58.
- **arXiv abstract pages:** 2309.03882, 2402.18766, 2304.07880, 2411.07854, 2401.02909, 2603.26511, 2603.26516,
  2606.16753, 2409.16235, 2502.08489, 2305.00948, 2303.15056.
- **Product documentation:** docs.typesafe.ai; huggingface.co/openjev.

*AI disclosure:* the searches, verification and synthesis were done with an AI assistant (Claude). Every reference
was checked against a primary record in this session; unconfirmed items are marked in the .bib.
