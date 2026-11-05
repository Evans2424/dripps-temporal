# Paper creation process record: "Which Cues Matter?" (draft 1)

**Pipeline:** academic-pipeline. **Date:** 2026-11-05.

**Outcome:** an ACL long-paper draft:
- `NLP_for_linguistics/acl_latex.tex`, with a copy of the PDF at `paper/draft1_which_cues_matter.pdf`;
- about 5.7 pages of main text out of 8, plus Limitations, Ethics, references and 3 appendix sections;
- 14 marked placeholders for the experiments still running or planned.

## Journey

| stage | what happened | user decisions |
|---|---|---|
| Intake | Found the ACL template with a title skeleton and RQ1/RQ2 | Long paper; pending work with placeholders; title "Which Cues Matter?"; section-by-section |
| 1 Research (quick) | 23 new references, 22 verified against primary records (DialectBench left out); synthesis of six themes; gap statement | continue |
| 2 Write (plan → full) | Blueprint agreed; §1 written after you approved the argument; then the full draft at your request | Jev secondary; full variety-model set; "beyond the inventory" central but hedged |
| 2 (analyses added) | Two new reproducible scripts: per-variety model breakdown (`11_variety_breakdown.py`) and identical-cue analysis (`12_identical_cues.py`) | — |
| 2.5 Integrity | 6 memory-added references verified (4 DOIs/pages added); 2 attributions softened; all numbers traced | — |
| 3 Review | Minor Revision; 10-item roadmap; no CRITICAL issues | — |
| 4 Revise | 9 items addressed; 1 declined with a reason (figure) | — |
| 3' / 4.5 | Every roadmap item verified; final integrity PASS for the draft stage | — |
| 5 Finalize | PDF compiled from LaTeX with the official ACL style (review mode) | — |

## Collaboration quality (AI self-assessment, 1–100)

| dimension | score | evidence |
|---|---|---|
| Accuracy of claims | 82 | Six over-statements were caught before review (fronting range, "reach the cue model", BERTimbau, Post/Simul attribution, Costa & Branco, Jev). Each was caught by checking against the tables, but **they were written in the first place**. |
| Reference integrity | 88 | 38 of 38 verified. Six entries were initially written from memory and verified later, a weakness in process order. |
| Reproducibility | 85 | Every number traced. Two analyses existed only as a scratchpad script until the integrity stage. |
| Scope discipline | 80 | The variety breakdown was added during drafting: in scope (you asked for some variety work now) but not in the blueprint. |
| Responsiveness to user decisions | 90 | Your choices (format, placeholders, Jev's role, framing) are reflected throughout. |
| Review independence | 65 | The Stage 3 panel was simulated by the same model that wrote the draft. Its critiques are real but likely miss blind spots a human reviewer would find. |

## AI self-reflection

- **Weakest point:** self-review. The same system wrote and reviewed the paper. A human co-author's read, and ideally
  an external colleague's, is the real Stage 3.
- **Recurring error pattern:** generalising a result from one model to "LLMs" (e.g. a Jev-only statistic). Watch for
  it in future edits.
- **Failure-mode audit:** no mode SUSPECTED. Frame lock is on WATCH: the annotation-noise explanation of the LLM gain
  remains open until annotator agreement and test B4 are run.

## Before submission (blocking)

1. Resolve all 14 `\pending{}` markers. They cover:
   - OpenJev and the shim baseline;
   - the robustness tests (identical-cue null, incremental validity, valid masking, fronting relabel);
   - the variety-matched model grid;
   - the native-speaker baseline;
   - the robustness-plan reference.
2. Freeze `docs/T5_ROBUSTNESS_PLAN.md` and `docs/OPENJEV_PLAN.md` (fill in the commit hash) **before** the
   confirmatory runs.
3. Decide on the code and data release statement (Ethics).
4. Check the anonymity of the review version (no repository URLs or author-identifying details).
5. Add a figure once the pending tables land (revisit RV9).
