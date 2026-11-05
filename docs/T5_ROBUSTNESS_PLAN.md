# T5 robustness plan: tests that make the Jev conclusions defensible

**Status:** design, not run. **Freeze this file (commit hash below) before running any confirmatory test.**
Written 2026-11-04, after the panel review (`docs/reviews/2026-11-04-full-panel-review.md`).

Frozen at commit: `__________` · date: `__________`

## 1. The claims under test

| id | claim (as currently stated) | current evidence |
|---|---|---|
| C1 | Zero-shot Jev reaches macro-F1 0.628 without any labels | `t5_decision.csv` |
| C2 | Marked, calibrated Jev (0.660) cannot be told apart from B3 (0.695) | paired −0.035 [−0.077, +0.004] |
| C3 | Jev carries information that the 17 annotated cues do not | B3 + Jev 0.738, +0.043 [0.016, 0.067] |
| C4 | Jev separates readings among clauses whose annotated cues are identical | +0.27, positive in 91% of pairs (descriptive) |
| C5 | Jev relies on different cues from the corpus: more on position, less on main-clause aspect, strongly on fronting | refit shares; fronting shift +0.58 Ant |
| C6 | Jev's raw probabilities are close to calibrated | ECE 0.084 |
| C7 | What Jev adds is lexical verb meaning and the timeline, not explicit adverbials | masking contrasts and examples (descriptive) |

## 2. Threats each claim faces

| threat | affects | test |
|---|---|---|
| **Post hoc selection:** Jev was added after the headline was fixed, and many rows were tried | C1–C3 | §3 (pre-registration), A4, A5 |
| **Contamination:** DRIPPS is published; Jev or Qwen may have seen the labels | C1–C4 | A6 |
| **Prompt fragility:** one wording, one set of option names | C1, C2 | A2 |
| **Run or version instability:** API answers may vary between calls or releases | all | A1, A3 |
| **"Any text helps":** the inventory is lossy, and any text model would add something | C3, C4 | B3 |
| **Annotation noise:** Jev matches annotator idiosyncrasy, not language | C3, C4 | B4 |
| **Invalid perturbations:** `<mask>` is out of distribution; fronting keeps a label it may invalidate | C5, C7 | C1, C4 |
| **Correlational masking:** masking effects don't isolate a causal factor | C7 | C2, C3 |
| **Informal statistics:** no intervals or null for C4, C5 or C6 | C4–C6 | B2, C5, D1 |

## 3. Pre-registration and multiplicity

- **Confirmatory family (Holm-corrected, α = 0.05):**
  - **H1:** incremental validity (B1).
  - **H2:** within-cell discrimination (B2).
  - **H3:** replication on held-out data (A5), with C1 and C3 re-tested on a batch no one has modelled.
- **Everything else is secondary:** reported with 95% group-bootstrap intervals and labelled exploratory.
- **The T5 headline stays as pre-specified:** `qwen3.8-27b-fp8`, plain text, template 1, raw. Jev is reported as
  exploratory in the current results and becomes confirmatory only through H1–H3.
- **Fixed in this file before running:** the prompt variants (A2), the deletion lists (C1, C3), the cells (B2), the
  number of permutations and bootstrap draws, and the decision rules.

## 4. The tests

### A. Is the result real and stable?

**A1 · Test–retest determinism** *(API, ~600 requests, < $0.02)*
- **Procedure:** rescore 200 random clauses × 3 orders on `jev-1.14.0` at a different time of day.
- **Statistic:** the maximum and mean absolute difference in probabilities, and agreement on the top answer.
- **Decision:** if the top answer agrees on less than 98% of clauses, score every clause 3 times and average before
  any other test.

**A2 · Prompt sensitivity** *(API, ~3,100 requests, < $0.15)*
- **Procedure:** four pre-declared variants, sent as **separate questions in one request** over the same state
  (TypeSafe skill: "compare wordings… in one request", so the state is paid for once):
  1. the current English wording, with option names `before` / `after` / `same_time`;
  2. the same question, with option names `anterior` / `posterior` / `simultaneous`;
  3. the Portuguese wording (template 2);
  4. a plain paraphrase without the "the sentence below" phrasing, written to the skill's "write plainly" rule.
- **Statistic:** macro-F1 per variant, with intervals; the spread is the largest minus the smallest.
- **Decision:** C1 is reported as the *minimum* over the variants. C1 is robust if every variant is at least 0.55
  and the spread is under 0.05.

**A3 · Version sensitivity** *(API, ~3,100 requests, < $0.15)*
- **Procedure:** the plain text, scored on `jev-1.13.0`.
- **Statistic:** agreement with 1.14 and the paired difference in macro-F1.
- **Decision:** report both versions. A difference above 0.03 means version-specific claims need a caveat.

**A4 · Fresh-seed confirmation** *(offline)*
- **Procedure:** recompute every T5 row on 10 new fold seeds (`SEED+5 … SEED+14`), which are unseen. No new API calls.
- **Decision:** the B3 + Jev gain must keep a paired interval above 0.

**A5 · Held-out replication (H3)** *(new data + API, < $0.10)*
- **Procedure:** use the fresh double-annotated batch from the annotator-agreement task (P0-1, about 150 clauses).
  Score Jev on it, train B3 on all existing rows, and evaluate on the new batch only. No refitting, tuning or choice
  of variant on these rows.
- **H3:** Jev's raw macro-F1 is above B1 (one rule), and B3 + Jev is above B3 (paired bootstrap).
- **Decision:** C1 and C3 count as confirmed only if H3 holds after Holm correction.

**A6 · Contamination check** *(offline + API, < $0.05)*
- **Procedure 1:** compare Jev's advantage on the **published** rows (the 793 rows of the original release) with
  the **later, unpublished** batches. To separate that from the later batches being harder, use the ratio of
  Jev's score to B3's on each subset.
- **Procedure 2 (memorisation probe):** for 100 published clauses, ask Jev a Choice question about which of four
  continuations follows the sentence: the real one and three plausible distractors. Near-chance means the sentence
  is not memorised.
- **Decision:** if Jev/B3 is clearly higher on published rows (the interval of the ratio difference excludes 0),
  report contamination as a live explanation, and rest C1–C3 on A5.

### B. Does Jev carry information beyond the cues?

**B1 · Incremental validity (H1, primary)** *(offline)*
- **Models:**
  - **M0:** B3 on the 17 cues.
  - **M1:** B3 on the 17 cues plus Jev's two log-ratios, `log(P_Ant/P_Simul)` and `log(P_Post/P_Simul)`, from the
    debiased raw probabilities.
  - Same L2 logit, standardised in the training fold, same folds and seeds.
- **Why it's leak-free:** Jev's probabilities use no labels, so they can enter as features directly. No nested
  out-of-fold step is needed, unlike stacking an encoder.
- **Statistic:** the paired difference in out-of-fold log loss (primary) and in macro-F1 (secondary), from a group
  bootstrap with 5,000 draws.
- **H1:** the log-loss difference is below 0, with an interval that excludes 0. This replaces the fixed-weight
  average as the evidence for C3.

**B2 · Within-cell discrimination (H2)** *(offline)*
- **Cells:** clauses with identical 17-cue vectors. Only cells that contain at least two readings and at least 5
  clauses are used.
- **Statistic:** for each cell and reading, the AUC of Jev's P(reading), comparing clauses that have it with those
  that don't. Average the AUCs, weighting by the number of pairs. AUC is used instead of the mean probability
  difference because it doesn't depend on calibration.
- **Null:** within each cell, shuffle Jev's probability vectors among the clauses; this keeps each cell's make-up
  and Jev's marginals. Use 10,000 permutations for a p-value, and a group bootstrap for the interval.
- **H2:** the weighted AUC is above 0.5 (permutation p < α after Holm).
- **Secondary:** the same statistic for an encoder's out-of-fold probabilities, as a comparison.

**B3 · "Any text helps" control** *(offline)*
- **Procedure:** repeat B1 and B2 with two label-trained text models, each fitted in-fold:
  - (a) TF-IDF bag-of-words + logit on `sentence_norm`;
  - (b) the marked encoder's out-of-fold probabilities.
- **Caveat:** in B1, the training rows of (a) and (b) need nested out-of-fold predictions; generate those in-fold.
- **Statistic:** the paired difference between Jev's B1 gain and each text model's B1 gain.
- **Reading:**
  - Jev's gain is clearly larger: it brings knowledge that this corpus can't teach.
  - The gains are equal: Jev adds "text information" that a supervised model could also learn.

**B4 · Annotation-noise alternative** *(needs the annotator-agreement batch)*
- **Procedure:** split the double-annotated clauses into *agreed* and *disagreed* readings. Compute B2's AUC and
  Jev's error rate in each group.
- **Reading:**
  - Jev's gain concentrated on disagreed items: it is matching one annotator's holistic reading.
  - Jev's gain holding on agreed items: genuine information.
- **Also:** report Jev's agreement with each annotator separately, as a human-like rater.

### C. What information does Jev use?

**C1 · Valid masking for LLMs** *(API, ~11,000 requests, < $0.30)*
- **Procedure:** re-run the three masks (participle, main verb, adverbials) and their controls in two forms:
  - **deletion:** the span is removed and the whitespace tidied;
  - **placeholder:** the span is replaced by the word *algo* ("something").
- **Keep:** the random-word controls.
- **Decision:** a span effect is claimed only if both forms agree in sign and both intervals exclude 0.

**C2 · Lexical verb minimal pairs** *(human + API, ~60 sentences, < $0.01)*
- **Procedure:** a linguist rewrites the main verb of 60 sentences (20 per reading) with a verb of the **same
  annotated aspectual class and tense** but a different lexical meaning. The 17 cues are identical, so B3's
  prediction can't change. Score Jev on the original and the rewrite.
- **Statistic:** the share of pairs where Jev's top reading changes, and the mean |ΔP|. A second annotator labels
  the rewrites; the statistic is computed against those labels.
- **Reading:** Jev's changes track the rewrite's gold label more often than chance (sign test). That is direct
  evidence for C7 ("lexical meaning beyond the aspect class").

**C3 · Timeline ablation** *(API, ~3,000 requests, < $0.10)*
- **Procedure:** delete all explicit time expressions together: the adverbial lexicon, plus pre-declared
  patterns for years, months, weekdays and *em/desde/até* + date. Recompute B2 (within-cell AUC) and B1.
- **Reading:**
  - B2's AUC falls a lot: the timeline expressions carry Jev's extra information.
  - It barely moves: the information is elsewhere (verb meaning, world knowledge).

**C4 · Relabelling after fronting** *(human, 120 sentences)*
- **Procedure:** two native speakers label 60 fronted rewrites and 60 originals, mixed and blind to the condition.
- **Statistic:** the share of moved sentences whose reading changes. Then compare Jev's shift with the relabelled
  gold, not the original gold.
- **Decision:** the claim "Jev over-reacts to fronting" is only allowed if Jev's shift towards Ant exceeds the
  human shift.

**C5 · Cue reliance with intervals** *(offline)*
- **Procedure:** bootstrap over sentence groups (5,000 draws) the B3 refit on Jev's labels against the refit on gold
  labels.
- **Statistic:** the paired difference in each block's share of |coefficients|, especially position and main-clause
  aspect, plus the Spearman ρ between coefficient vectors, with intervals.
- **Framing:** cue *validity* (gold refit) against cue *strength* (Jev refit), in the Competition Model sense.

### D. Calibration

**D1 · Calibration with intervals** *(offline)*
- **Procedure:** reliability diagrams with bootstrap bands, and ECE computed both with 10 equal-width and 15
  equal-mass bins, plus a debiased estimator. Paired differences between raw Jev, calibrated Jev and B3 in log loss
  and Brier.
- **Decision:** "Jev is close to calibrated" requires that raw Jev's ECE interval overlaps B3's, or the claim is
  rewritten as "less miscalibrated than stock LLMs".

### E. Comparators

- **E1 · The pre-specified headline:** Qwen 27B (queued), and one open decision model as in the grant plan. Same
  tables, same tests B1 and B2.
- **E2 · Human baseline:** 3 native speakers per variety, without the guidelines, label 100 clauses. This gives a
  human macro-F1 to place Jev's 0.628 against, and an untrained-agreement figure.

## 5. Budget, data sent and order of work

**API.** About 21,000 requests in total; under **$1** at $0.042 per million input tokens.
- **Data sent:** corpus sentences and their edited variants only, from public newswire. No annotation and no
  personal data beyond names already in the news text.
- **Key:** `TYPESAFE_API_KEY` from `.env`.
- **Logging:** every request payload is logged (`results/jev_payloads/*.jsonl`), so any answer can be rerun.
- **Approval:** ask before each API batch.

**Human tasks.** These can run in parallel:
- the annotator-agreement batch (shared with P0-1);
- the fronting relabel (C4), 120 sentences;
- the verb minimal pairs (C2), 60 sentences;
- the human baseline (E2), 100 clauses × 3 speakers × 4 varieties.

**Order of work:**

| week | work | needs |
|---|---|---|
| 1 | B1, B2, B3, C5, D1, A4 | offline, existing caches; new script `experiments/11_beyond_cues.py` |
| 2 | A1, A2, A3, A6, C1, C3 | API, one batch per approval; new script `experiments/12_jev_robustness.py` |
| 3–5 | C2, C4, E2 | human labels |
| 5–6 | A5, B4 | the double-annotated batch |
| 6 | Holm over H1–H3, then the write-up | |

## 6. How the claims will be worded after the tests

| claim | wording if the tests pass | fallback wording |
|---|---|---|
| C1 | "zero-shot, Jev reaches X (minimum over four prompts), replicated on held-out data" | "on this corpus, with this prompt" |
| C3 | "Jev's probabilities improve held-out log loss beyond the 17 cues (H1), more than supervised text models do (B3)" | "Jev's errors differ from B3's" |
| C4 | "within identical cue configurations, Jev ranks the true reading above chance (H2)" | drop the claim |
| C5 | "Jev's cue strengths differ from corpus cue validities: position ↑, main-clause aspect ↓ (C5 intervals)" | "descriptively" |
| C7 | "minimal pairs show sensitivity to lexical verb meaning (C2); timeline expressions explain Y% (C3)" | "candidate explanation" |
