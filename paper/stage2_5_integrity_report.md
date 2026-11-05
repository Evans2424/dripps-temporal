# Stage 2.5 integrity verification (pre-review)

**Paper:** `NLP_for_linguistics/acl_latex.tex`, draft 1. **Date:** 2026-11-05. **Verdict: PASS**, after 3 corrections
applied in this stage.

## Phase A: references (38 cited)

| status | n | keys |
|---|---|---|
| Verified in Stage 1 (primary record this session) | 21 | e.g. costa-branco-2012-*, ning-etal-2018-multi, wang-zhao-2024-tram, santos2024gervasio, simplicio2026amalia, ferreira2026p3b3 |
| Verified in this stage (added from memory earlier) | 6 | guo2017calibration (PMLR 70:1321–1330), kull2019dirichlet (NeurIPS 32), pezeshkpour2024order (Findings NAACL 2024, 2006–2017; DOI added), zhao2021calibrate (PMLR 139:12697–12706), souza2020bertimbau (BRACIS 2020, 403–417; DOI added), rodrigues2023albertina (EPIA 2023, 441–453; DOI added) |
| Original `docs/references.bib`, checked against primary records when that file was built | 11 | breiman2001, chen2016, conneau2020xlmr, he2023debertav3, holte1993, lobo2003, lundberg2020, silvano2021apc, silvano2023dripps, varoquaux2018, zheng2024selection |
| Product documentation (cited as misc) | 2 of the 21 | typesafe2026jev, openjev2026 |
| Missing or unverifiable | **0** | DialectBench (unconfirmed) is **not cited** |

## Phase B: citation context

Two attributions were stronger than their sources and are **corrected**:
- **Costa & Branco (2012):** "shown to improve" became "used as a feature".
- **Jev:** "trained for calibrated decisions" became "that its provider describes as trained for calibrated decisions".

All other attributions match the verified abstracts.

## Phase C: numbers (every figure traced to a table or a script)

**Traced:**
- Label shares, n per variety and 1,017 sentences: corpus (`io.analysis_frame`).
- Baselines, circular-model check (0.750 against 0.736): `baselines.csv`.
- Table 2: `cue_importance.csv`.
- Coefficients, fronting counts (67/2/11 of 80): `docs/FINDINGS.md` and `logit_coefficients.csv`.
- Participial-aspect checks (466, 797, r ≤ 0.21): computed from the corpus in this session.
- Table 3: `t1_encoder.csv`, including the paired rows.
- Table 4: `t5_decision.csv` and `t5_decision_diag.csv`.
- Table 5: `t5_decision_diag.csv`, `t5_decision_perturb.csv`, `t3_ablation.csv`.
- Identical-cue numbers: **`experiments/12_identical_cues.py` → `t7_identical_cues.csv`** (new; replaces a one-off
  scratchpad analysis so every number is reproducible).
- B6: `b6_tests.csv`, `b6_performance.csv`.
- Table 6 and its appendix: **`experiments/11_variety_breakdown.py` → `t6_variety_by_model.csv`** (new).
- Appendix A: `t1_encoder.csv` (lab tables).
- Limitations (0.60 against 0.72): `docs/FINDINGS.md`, `b3_by_batch.csv`.

**Corrected during drafting (before this stage):**
- the fronting range 0.58–0.62 became 0.48–0.62;
- "LLMs reach the cue model" now says only the open 27B model is indistinguishable (calibrated plain Jev is below B3);
- BERTimbau marked is indistinguishable from B3;
- the Post/Simul share is attributed per model.

## Phase D: originality

The text was written for this paper from the project's own documents. No passages are copied from cited works.

## Phase E: claims

The central claims carry their evidence and hedges:
- **The gain over B3** (weight fixed in advance, paired intervals) is stated as evidence of complementary information,
  not proof of new cues.
- **The "beyond the inventory" mechanism** is flagged as descriptive, with the confirmatory tests marked
  `\pending{}`.
- **Variety contrasts** are labelled descriptive.

## AI research failure-mode checklist (7 modes)

| mode | status | note |
|---|---|---|
| 1 Implementation bugs | CLEAR | pipeline code under test (93 tests); code review findings fixed; B3 recovery is exact algebra (2·fused − model) |
| 2 Hallucinated results | CLEAR | every number is traced (Phase C); two new scripts make the last two analyses reproducible |
| 3 Shortcut reliance | CLEAR | leakage layers excluded by construction; sentence-grouped folds; LLM raw rows use no labels |
| 4 Bug presented as insight | CLEAR | small models' "always Ant" is checked across 3 option orders (a label prior, not a slot artefact) |
| 5 Methodology fabrication | CLEAR | methods match `docs/methods.md`, `CALIBRATION.md` and the code; pending work is marked as pending |
| 6 Frame lock | WATCH | the LLM-complementarity story could absorb the alternative "annotation noise" explanation; it is named in Limitations and in robustness test B4 |
| 7 Pipeline-level | CLEAR | headline set in advance (Qwen3.8-27B), Jev secondary; exploratory status stated |

No mode is SUSPECTED, so the pipeline is not blocked.

## For the authors (not blocking)

- The Ethics statement says "We will release code, tables and the LLM probability caches". Confirm this is intended.
- `\pending{}` markers (red) must all be resolved before submission; the build should fail if any remain.
