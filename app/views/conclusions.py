import streamlit as st

# A meeting snapshot, written by hand from results/tables on 2026-11-05 (all launched runs in). The live numbers are
# on the Results pages; re-check every figure here after any re-run or new model (e.g. the OpenJev runs).

HEADLINE = """
1. **The main clause carries the reading.** Main-clause aspect and tense are the top two cues for every method.
   The participial clause's own aspect is dispensable: dropping it costs nothing.
2. **The cues combine additively.** A linear logit (B3, macro-F1 0.695) matches random forests and XGBoost.
   There is no evidence for a configuration that needs both clauses at once.
3. **Fine-tuned encoders only match B3.** Text, with or without the cues, never beats the 17 annotated features.
4. **Large zero-shot LLMs carry information the cues don't.** The pre-set headline, open Qwen3.8-27B, reaches
   0.560 raw and 0.668 calibrated (≈ B3). Averaged with B3, it gives **0.738 (+0.043 [0.018, 0.068])**; Jev on the
   same plain text gives +0.037 [0.014, 0.059] (Jev is better raw, at 0.628). Smaller Qwen models answer "Before" for
   almost everything.
5. **Varieties share the cue grammar** and differ in base rates. Different cue weights are not shown.
"""

SCOREBOARD = """
| family | best | worst | takeaway |
|---|---|---|---|
| **Classical** (annotated cues) | B3 logit / XGBoost **0.695** | majority 0.196, one rule 0.442 | additive is enough; the tree (0.649) finds the same top three cues |
| **Encoders** (fine-tuned on the text) | xlm-r residual on B3 **0.702**, marked + cues 0.702, Albertina marked 0.693, mDeBERTa marked 0.698 (≈ B3) | xlm-r plain **0.621**, unstable across seeds; Albertina plain 0.625, BERTimbau plain 0.652, mDeBERTa plain 0.664 | marking the two verbs closes most of the gap; Portuguese pretraining, European or Brazilian, doesn't |
| **Stock LLMs** (Qwen, zero-shot) | **Qwen3.8-27B calibrated 0.668** (≈ B3: −0.027 [−0.064, +0.012]); raw 0.560; Qwen3.6-27B raw 0.491, calibrated 0.660 | 4B / 9B raw **0.16**: "Before" for 98% of clauses | scale removes the label bias; the newer 27B release is better raw (+0.069) but not once calibrated (−0.008, n.s.) |
| **Decision model** (Jev, zero-shot) | marked + calibrated **0.660** (≈ B3: −0.035 [−0.077, +0.004]) | raw 0.626–0.628 (still good) | close to B3 without annotation; little position bias (3 orders agree 90%) |
| **B3 + model** (w = 0.5, pre-set) | **B3 + Qwen 27B 0.738** (+0.043); B3 + Jev 0.732 (+0.037; marked 0.738, +0.043); B3 + Qwen3.6-27B 0.726 (+0.031); B3 + Qwen 9B 0.722 | B3 + encoders at w = 0.5: no gain | the zero-shot LLMs add information, open and closed alike; encoders trained on the same labels relearn the cues |
"""

CUES = """
Ranks 1 (strongest) to 6. Ablation and permutation refit or corrupt B3; SHAP explains the tree models; the last two
columns perturb the sentence text.

| cue | logit ablation | permutation | SHAP forest | SHAP XGBoost | encoder (masking) | Jev |
|---|---|---|---|---|---|---|
| main-clause aspect | **1** (0.093) | **1** | **1** | 2 | main verb masked: −0.033 | main verb ≈ participle |
| main-clause tense | **2** (0.057) | **2** | **2** | **1** | — | — |
| clause position | 3 (0.031) | 3 | 4 | 4 | fronting: Ant **+0.36** | fronting: Ant **+0.58** |
| connector | 4 (≈ 0) | 4 | 6 | 6 | — | — |
| both-clause interactions | 5 (≈ 0) | n/a | 3 | 5 | — | — |
| participial-clause aspect | **6 (−0.007)** | 5 | 5 | 3 | participle masked: −0.009 | weights it more than the corpus |

**Where the methods agree:**
- The **main clause is first** for every method on the annotated cues, and for the encoder and Qwen 9B on the text:
  masking the main verb hurts more than masking the participle.
- **Position matters** for every method. Moving the participial clause to the front pushes **every** text model
  towards Ant.

**Where they disagree:**
- **XGBoost's SHAP ranks participial-clause aspect 3rd.** SHAP shares credit between correlated features, while the
  refit shows the model doesn't need it, so the SHAP rank is redundancy, not use.
- **Large LLMs weight the cues differently from the corpus**, Qwen 27B and Jev alike. Refit on their answers,
  main-clause aspect gets 12% and 8% of the weight (gold: 23%), and position 20% and 19% (gold: 11%). Both shift
  strongly to Ant under fronting (+0.62, +0.58). So high cue *strength* for position, against moderate cue
  *validity*, is a shared LLM pattern, not a quirk of Jev.
"""

LINGUISTICS = """
- **The participial clause is temporally anaphoric on the main clause.** Its reading follows the main event's
  aspectual class and tense, not its own. This qualifies accounts that require the same aspectual configuration
  in both clauses.
- **The directions match the literature:** durative main clause → Simul; telic main clause → Ant or Post;
  present-tense main clause → Ant.
- **Fronting all but blocks the posterior reading** (2 of 80 fronted clauses are Post). This looks like an
  order-of-mention (iconicity) constraint, and every text model has learned it, Jev most strongly. It ranks only 3rd
  because fronted clauses are rare (7.7%).
- **Post versus Simul is the hard boundary.** Of the clauses Jev gets right and B3 misses, 107 of 186 are Post/Simul
  confusions. Separating them seems to need the **lexical meaning of the verbs and the timeline** (sequences of
  years, "*tendo cessado atividade em 2009*"), which the aspect classes don't encode.
- **Large LLMs find information beyond the inventory.** On plain text, B3 + Qwen 27B gains +0.043 and B3 + Jev
  +0.037 (both intervals above 0). Among
  clauses with identical annotated cues but different readings, Jev gives the true reading **+0.27 more
  probability** (in 91% of cases), which no cue-only model can do.
  Explicit adverbials and dates are only slightly more frequent where it wins, so the lexical semantics of the
  verbs is the stronger candidate.
- **One grammar, different usage.** The four varieties differ in how often each reading and context occurs (BP
  leans Ant, with more fronting and *mesmo*), not in how the cues are weighted.
"""

CAVEATS = """
- **Annotation:** there is no annotator agreement for the later batches. They also score lower (B3: 0.72 on the
  original rows, about 0.60 on the new ones), so either they're harder or they were annotated differently.
- **Exploratory rows:** apart from the pre-set headline rows, everything is exploratory, with no correction for
  multiple comparisons. B3 + Jev is the best of several combinations.
- **Jev is closed** (pinned to `jev-1.14.0`; only TypeSafe can rerun it). The open Qwen 27B now replicates the
  B3 + LLM gain, so that claim no longer depends on Jev.
- **The "beyond the cues" analysis is descriptive** (a one-off script, no intervals yet). Masking is correlational.
- **Scope:** British English is excluded from the pooled models (95% Ant).
"""

NEXT = """
1. **All launched runs are in** (Qwen 4B, 9B, 3.6-27B, 3.8-27B; Jev; encoders xlm-r, BERTimbau, Albertina,
   mDeBERTa). No encoder beats B3. Every large LLM adds to it.
2. **Run the robustness tests on Qwen 27B as well as Jev** (`docs/T5_ROBUSTNESS_PLAN.md`): it is the
   pre-registered headline, and it is open.
3. **Turn the "beyond the cues" analysis into an experiment** (`11_beyond_cues.py`): the identical-cues test and the
   masking contrasts, with paired bootstrap intervals.
4. **Test the candidate cues directly.** Annotate verb lexical class and explicit temporal expressions on a sample,
   add them to B3, and see whether that closes the gap to B3 + Jev.
5. **Annotator agreement** on a fresh double-annotated batch, before the paper relies on the later batches.
6. **Robustness of the T5 results:** Jev with the Portuguese wording (template 2), and a second open decision
   model, as in the grant plan.
7. **RQ3:** leave-one-variety-out transfer.
8. **Paper outline:** RQ1 hierarchy → model families → beyond the inventory (Jev) → varieties.
"""


def render():
    st.title("Conclusions so far")
    st.caption("Meeting overview, as of 5 November 2026. Live tables are on the Results pages; numbers are macro-F1 "
               "on 1,038 Portuguese clauses, out-of-fold, with 95% sentence-bootstrap intervals.")
    st.markdown(HEADLINE)
    t1, t2, t3, t4 = st.tabs(["Methods: best and worst", "Cue importance: do methods agree?",
                              "Linguistic ideas", "Caveats and next steps"])
    with t1:
        st.markdown(SCOREBOARD)
    with t2:
        st.markdown(CUES)
    with t3:
        st.markdown(LINGUISTICS)
    with t4:
        st.markdown("#### Caveats")
        st.markdown(CAVEATS)
        st.markdown("#### Next steps")
        st.markdown(NEXT)
