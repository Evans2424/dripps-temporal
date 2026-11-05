# Full panel review — DRIPPS temporal cue hierarchy (pre-manuscript)

- **Mode:** `academic-paper-reviewer` full, round 1.
- **Date:** 2026-11-04.
- **Material reviewed:** there is no manuscript yet, so the panel reviewed the claims the paper would make, as stated in
  `docs/FINDINGS.md`, `docs/methods.md`, `docs/CALIBRATION.md`, the app's *Conclusions so far* page and
  `results/tables/` (including T5 / Jev). Section references point to those files.
- **Read-only:** no project file was changed. This report is a separate document.

---

## Phase 0 — Field analysis and reviewer configuration

| item | assessment |
|---|---|
| Primary discipline | Computational linguistics: corpus-based tense and aspect semantics |
| Secondary | NLP evaluation (encoders, LLMs as zero-shot classifiers), Portuguese linguistics, variationist linguistics |
| Paradigm | Quantitative corpus study with predictive models as instruments; hypothesis-testing about cue hierarchies |
| Target venue | COLING (main track), or *Computational Linguistics* / *Language Resources and Evaluation* for a longer version |
| Maturity | Results largely complete for RQ1 and RQ2; T1, T3 and T5 partly run (27B decision models and two encoders pending); no draft |

| reviewer | configured identity | focus |
|---|---|---|
| **EIC** | Area chair, COLING semantics and discourse track | Fit, originality, coherence of the contribution |
| **R1, methodology** | Statistician working on ML evaluation in small-n NLP | Validity of CV, bootstrap, multiplicity, perturbation design, calibration |
| **R2, domain** | Romance linguist working on Portuguese non-finite clauses and tense and aspect | Theory, annotation validity, literature |
| **R3, perspective** | Psycholinguist and cognitive scientist working on cue integration and LLM-as-informant studies | Cross-disciplinary framing, human baselines, reproducibility |
| **Devil's Advocate** | Sceptical senior reviewer | The strongest case against the two headline claims |

*Adjustable: tell me if you want different reviewer profiles and I'll rerun Phase 1.*

---

## Phase 1 — Reviewer reports

### EIC report

**Recommendation:** Major Revision · **Confidence:** 4

**Summary.** The work asks which cues determine the temporal reading (Ant, Post or Simul) of Portuguese adverbial
perfect participial clauses (*tendo* + PP), whether they form a hierarchy, and whether the hierarchy differs across
EP, BP, AP and MP. On 1,038 annotated clauses, an additive logit on 17 annotated cues (B3, macro-F1 0.695) is the
reference. Main-clause aspect and tense dominate, and participial-clause aspect is dispensable. Fine-tuned encoders
only match B3. A commercial zero-shot decision model (Jev) reaches 0.628 and, averaged with B3, 0.738. The question
is well posed, the statistical hygiene is far above the venue average (grouped CV, a group bootstrap, structural
leakage guards, decision rules written down before the runs), and the "main clause carries the reading" result is a
clear, citable claim. The main risk is that the paper tries to be two papers: a linguistic hierarchy study and a
six-family model comparison. The second currently overshadows the first, with a closed model as its most striking
result.

**Strengths.**
- **S1. A sharp, falsifiable linguistic claim.** The participial-clause aspect ablation is −0.007 [−0.022, 0.008],
  and it is clean, since dropping it also removes the interaction terms (FINDINGS, RQ1).
- **S2. Unusually careful evaluation.** Bootstrap over sentence groups, not folds (methods, "Intervals…"), a leakage
  guard in code, and decision rules fixed before the runs (methods, `08_encoder.py`).
- **S3. Several converging methods** (ablation, permutation, trees, SHAP, encoders, LLM perturbations) agree on
  main-clause dominance (Conclusions, "Cue importance").

**Weaknesses.**
- **W1. The contribution isn't focused (Major).** The paper has six model families, about 60 encoder contrasts and
  about 20 T5 rows. *Suggestion:* make RQ1 and RQ2 the paper. Keep the model families as converging evidence in one
  section and the Jev result as a clearly labelled exploratory finding. Move the encoder variants to an appendix.
- **W2. The most striking result rests on a closed, recent commercial system (Major).** Venue reviewers will ask
  for an open replication. *Suggestion:* pair Jev with at least one open model that does comparably, or frame Jev
  as a probe, not a method.
- **W3. Annotation quality is unknown (Major; see R1 and the DA).**

---

### Reviewer 1 — Methodology

**Recommendation:** Major Revision · **Confidence:** 5

**Summary.** The core design is sound and, in places, exemplary. The problems are in the newer layers: how the
headline is chosen, multiplicity, the validity of the perturbations for LLMs, and an informal "beyond the cues"
analysis.

**Strengths.**
- **S1. The unit of independence is right everywhere.** `StratifiedGroupKFold` on `sentence_group`, a percentile
  bootstrap over groups, and paired differences from shared resamples.
- **S2. Ablation and permutation are reported together**, with the derived interaction block handled correctly
  (ablation only, recomputed products; methods, "The interaction block…").
- **S3. Calibration is done correctly:** Dirichlet calibration fitted on training folds only, raw and calibrated
  rows kept apart, and ECE reported alongside log loss and Brier (CALIBRATION, §4–6).

**Weaknesses.**
- **W1. The headline was switched after the fact (Major).**
  - *Problem:* `10_decision.py` names `qwen3.8-27b-fp8`, plain, raw as the T5 headline, and that model hasn't been
    run. The results narrative (Conclusions, headline 4) is built on Jev, which was added later.
  - *Why it matters:* this is the garden of forking paths in its plainest form.
  - *Suggestion:* report the pre-specified headline once it exists. Label every Jev row exploratory, as `methods.md`
    already does for the later encoder variants. Confirm the B3 + Jev gain on fresh seeds or a held-out batch.
- **W2. Multiplicity across T1 and T5 (Major).**
  - *Problem:* about 80 paired intervals with no correction. "The best of several" (B3 + Jev marked calibrated) is
    presented with its unadjusted interval.
  - *Suggestion:* use Holm or max-T correction over the pre-declared family, or a held-out confirmation set. At
    minimum, report how many combinations were tried.
- **W3. The perturbations aren't designed for LLMs (Major).**
  - *Problem 1:* masking writes a literal `<mask>` string, the XLM-R mask token, which is out of distribution for
    Qwen and Jev.
  - *Problem 2:* `move_apc_initial` keeps the original gold label, although fronting is a coded cue that changes the
    expected reading (67 of 80 fronted clauses are Ant). Jev's −0.25 in P(gold) may be *correct* behaviour.
  - *Suggestion:* for LLMs, delete the span or substitute a neutral word. For fronting, report the shift towards Ant
    as the result, not the drop in P(gold), or have native speakers relabel a sample of moved sentences.
- **W4. The "beyond the cues" evidence is informal (Major).**
  - *Problem:* the identical-cues result (+0.27, 91%) has no interval and no null distribution. Its ceiling is the
    in-sample cell majority.
  - *Suggestion:* add a within-cell permutation null (shuffle Jev's probabilities within each cue cell), a group
    bootstrap interval, and the incremental-validity test (B3 cues plus Jev's log-probabilities as features,
    out-of-fold).
- **W5. Small reproducibility gaps (Minor).**
  - B3 is 0.6943 on Deucalion and 0.6950 locally; record the environments.
  - Hyperparameter selection uses about 165 inner-validation rows.
  - The main-verb subset (660) is selective.
  - The encoder run isn't bit-reproducible; methods says so.

---

### Reviewer 2 — Domain

**Recommendation:** Major Revision · **Confidence:** 4

**Summary.** Linguistically, the most interesting results are that the subordinate clause's aspect is dispensable
and that a *perfect* form (*tendo* + PP) frequently has a **posterior** reading (42%). Both deserve more theory than
they get.

**Strengths.**
- **S1. Aspect is decomposed into Moens and Steedman primitives**, not raw Vendler classes, which makes the cue
  blocks interpretable.
- **S2. The directions match the predictions in the literature** (durative → Simul, telic → Ant or Post; Silvano
  et al. 2021), and fronting → Ant (Lobo 2003).
- **S3. Using British English as a contrastive reference** (95% Ant) explains the cross-linguistic puzzle well.

**Weaknesses.**
- **W1. Could the dispensable result come from the annotation (Major)?**
  - *Problem:* in *tendo* + PP the participle carries perfect morphology. If annotators coded the participial
    clause's aspectual class with little variance, or partly from the perfect form itself, the ablation would find
    nothing to remove.
  - *Suggestion:* report the distribution of participial-clause aspect values and their agreement. Show that the
    block varies, and that its marginal association with TR is non-trivial before conditioning on the main clause.
- **W2. Thin theoretical engagement with the Post reading (Major).**
  - *Problem:* a perfect participle reading as *posterior* needs an account. Narrative progression, or the APC as a
    continuation, is mentioned only in the project summary.
  - *Suggestion:* engage with the semantics of absolute and free-adjunct constructions (Stump 1985; Kortmann 1991),
    with default narration and order of mention in discourse (Dowty 1986; Lascarides & Asher 1993), and with
    Portuguese work on *tendo*-clauses beyond Lobo (2003).
- **W3. The variety claim is too strong (Minor).**
  - *Problem:* "one grammar, different usage" rests on a test with no single significant slope, at n = 193–345, on
    newswire.
  - *Suggestion:* say "no evidence of different cue weights at this sample size" (FINDINGS already says this; the
    Conclusions page drops the hedge).
- **W4. The "lexical semantics of verbs" interpretation goes beyond the evidence (Minor).** It is inferred from
  masking effects and examples.
  - *Suggestion:* annotate a lexical verb class on a sample and test it directly. This is next step 4 on the
    Conclusions page.

---

### Reviewer 3 — Perspective

**Recommendation:** Minor Revision · **Confidence:** 3

**Summary.** The study is close to a classic question in cognitive science: how comprehenders combine cues with
different validities. It could speak to that literature with little extra work.

**Strengths.**
- **S1. Refitting B3 on a model's own answers is a neat behavioural probe** (`t5_decision_diag.csv`). It measures
  how strongly the model relies on each cue, separately from how well each cue predicts.
- **S2. Option orders and calibration are handled rigorously**, which is rare in LLM-as-annotator work.

**Weaknesses.**
- **W1. Frame the cue results in terms of cue validity versus cue strength (Major).**
  - *Problem:* B3 estimates corpus cue *validity*. Jev's refit estimates the cues a reader relies on. Their mismatch
    is a result in its own right: Jev uses position more and main-clause aspect less.
  - *Suggestion:* frame this with the Competition Model (MacWhinney & Bates 1989), in which cue validity and cue
    strength can come apart.
- **W2. There's no human baseline (Major).**
  - *Problem:* the paper can't tell whether 0.63 zero-shot is high, because nobody knows how often untrained native
    speakers agree with the annotation.
  - *Suggestion:* have 3–5 native speakers per variety label 100 clauses without the annotation guidelines. That
    gives a human ceiling and an implicit agreement figure.
- **W3. Reproducibility and data ethics for the closed model (Minor).** Record the payload format, the version pin,
  the date and the data-processing terms (TypeSafe's DPA). Release the raw Jev probabilities so others can reanalyse
  them without the API.

---

### Devil's Advocate report

**Strongest counter-argument.** The two headline claims may both follow from **how the labels were made**, not from
how Portuguese works.

1. **"The main clause carries the reading."** Annotators decided TR by reading the whole sentence, then coded the
   aspect of each clause. If the participial clause's aspect is near-constant (it is formally a perfect) or was
   coded with the reading in mind, its ablation will be null whatever the linguistic truth. Being dispensable *given
   the main-clause cues* is a statement about redundancy in this annotation, not proof that the participial clause
   plays no role.
2. **"Jev finds information beyond the cues."** Any model that reads the full text has access to what the
   annotators used and the 17-cue inventory left out, including their idiosyncrasies. A gain from combining models
   is then a measure of how lossy the inventory is, plus annotation noise that Jev happens to share. It isn't
   evidence of new linguistic factors.
3. **No annotator agreement.** Without it, neither alternative can be ruled out. The later batches score 0.60
   against 0.72, which is consistent with annotation drift.

**Issues.**

| # | severity | issue | where |
|---|---|---|---|
| DA1 | **CRITICAL** | No inter-annotator agreement at all, while the central claims are about what the labels encode, and the later batches behave differently | FINDINGS, "Later annotation batches" |
| DA2 | MAJOR | Participial-clause dispensability is not checked against the cue's variance or annotation bias (with R2 W1) | FINDINGS, RQ1 |
| DA3 | MAJOR | The T5 headline was changed after the fact; Jev is presented as the result (with R1 W1) | `10_decision.py` docstring; Conclusions |
| DA4 | MAJOR | The fronting perturbation keeps a label it may invalidate, so "Jev's strongest reaction" is uninterpretable (with R1 W3) | Conclusions, linguistics tab |
| DA5 | MINOR | "Lexical semantics of verbs" is inferred from correlational masking and hand-picked examples | Conclusions, linguistics tab |

**Alternative explanations not considered:**
- Jev shares a prior with the annotators (both read the whole text), not new information.
- Qwen's "always Before" reflects instruction-tuned priors, not missing knowledge; the calibrated 0.58 suggests the
  knowledge is there.
- The variety differences in base rates reflect newspaper genre and topic, not dialect.

**Missing stakeholders:** speakers of AP and MP (the varieties with the least annotation), and the annotators
themselves, whose guideline interpretation drives the results.

**Observations (not defects):**
- The decision rules written down before the encoder run are a model for the field.
- The calibration write-up is clear and correct.

---

## Phase 2 — Editorial decision

**Decision: Major Revision.** The Devil's Advocate raised a CRITICAL issue (DA1), so Accept is not possible.

**Consensus (all five reviewers):**
- The core RQ1 design and the statistical hygiene are strong.
- Main-clause dominance is the paper's clearest contribution.
- The Jev results are interesting but must be labelled exploratory and backed by formal tests.

**Disagreements:**
- *Scope.* The EIC wants the model comparison cut back. R3 sees the cue-validity versus cue-strength contrast as a
  selling point. **Ruling:** keep Jev, but as a probe of cue reliance, framed through the Competition Model, not as
  a leaderboard entry.
- *Severity of the annotation gap.* R3 rates it Minor; the DA rates it Critical. **Ruling:** Critical. It bears
  directly on both headline claims.

## Revision roadmap (prioritised)

**P0 — required before submission**
1. **Inter-annotator agreement:** double-annotate a fresh sample (about 150 clauses, across varieties and batches).
   Report κ or α for TR and for each cue block, especially participial-clause aspect. (DA1, DA2, R2 W1)
2. **Participial-clause aspect checks:** its distribution, its marginal association with TR, and agreement. Show the
   ablation isn't a variance artefact. (R2 W1, DA2)
3. **Headline and multiplicity:** run and report the pre-specified Qwen 27B headline. Label all Jev rows exploratory.
   Correct over a declared family, or confirm B3 + Jev on fresh seeds or a held-out batch. (R1 W1–W2, DA3)
4. **Formalise "beyond the cues"** (`11_beyond_cues.py`): the within-cell permutation null with intervals, plus the
   incremental-validity test (B3 cues plus Jev's log-probabilities, out-of-fold). (R1 W4)
5. **Make the perturbations valid for LLMs:** delete or substitute a neutral word instead of `<mask>`. Report
   fronting as a shift in the distribution, and relabel a sample of moved sentences. (R1 W3, DA4)

**P1 — strongly recommended**
6. A human baseline: untrained native speakers on 100 clauses. (R3 W2)
7. The theory section: absolutes and free adjuncts, narration and order of mention, Portuguese *tendo*-clause
   literature, and the Competition Model. (R2 W2, R3 W1)
8. An open replication of the decision result, or a clear "probe" framing; release Jev's raw probabilities and the
   request format. (EIC W2, R3 W3)

**P2 — polish**
9. Narrow the paper to RQ1 and RQ2, with converging model evidence; move the encoder variants to an appendix. (EIC W1)
10. Hedge the variety claim and the lexical-semantics interpretation. (R2 W3–W4, DA5)
11. Document the environments (the 0.6943 versus 0.6950 B3 gap) and the selective main-verb subset. (R1 W5)

*Phase 2.5 (revision coaching) is available: ask for it, or say "just fix it" to skip.*

**Reference note:** the literature the reviewers suggest (Stump 1985; Kortmann 1991; Dowty 1986; Lascarides & Asher
1993; MacWhinney & Bates 1989) was cited from memory and not checked in this session. Verify each one before it
enters `docs/references.bib`.
