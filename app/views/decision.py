import pandas as pd
import streamlit as st

from data import table
from views.encoder import _t1_chart

from dripps import decision

SAMPLE = "Orlando Janeiro desfilou pelas artérias da cidade, tendo apostado no contacto interpessoal."

MODELS = pd.DataFrame([
    ("Qwen3.8-27B-FP8", "27B", "headline", "Largest and newest open model in the ladder; FP8 weights (~27 GB, ~54 GB in bf16)."),
    ("Qwen3.6-27B-FP8", "27B", "same size, older release", "Same size, one release earlier: shows whether a newer model makes a difference."),
    ("Qwen3.5-9B", "9B", "scale ladder", "Middle rung: does the effect hold below 27B?"),
    ("Qwen3.5-4B", "4B", "scale ladder", "Small rung: where does the task stop working?"),
    ("Qwen3.5-0.8B", "0.8B", "smoke test only", "Pipeline check; expected to be close to chance."),
    ("Jev 1.14.0 (TypeSafe)", "closed", "trained decision model", "A model trained for calibrated decisions, through its API "
     "(version pinned): what does decision training add over a stock LLM read the same way?"),
], columns=["model", "size", "role", "why it is here"])


def _how():
    st.markdown("""
A **decision** model gets a closed set of options and returns one probability per option. It does not
write an answer. Here the options are the three readings of the participial clause, and nothing is
trained: the model sees the task for the first time (zero-shot).

1. **One prompt per sentence.** The question, the sentence and the three readings as options A, B and C.
   The prompt ends at `Answer:`, so the model's next token has to be a letter.
2. **One forward pass.** No text is generated. The logits of the tokens ` A`, ` B` and ` C` are read and
   soft-maxed into three probabilities. Thinking mode is off, so the answer is the model's first impulse,
   not a chain of reasoning.
3. **Three option orders.** LLMs prefer some slots (often A or the last option). Each sentence is
   scored three times with the options rotated, so every reading sits in every slot once. The geometric
   mean of the three cancels a pure slot preference exactly.
4. **Optional calibration.** Raw probabilities are scores, not odds. The *calibrated* rows fit a
   12-parameter logit on the training fold only. This is the only place where gold labels touch the model.
""")
    st.markdown("**The user turn the Qwen models see** (template 1, option order A = Ant, B = Post, C = Simul):")
    st.code(decision.render_user(decision.build_prompt(SAMPLE)), language=None)
    st.caption("The model's chat template wraps this as a user turn, with thinking off, and the assistant turn starts "
               "with `Answer:`, so the next token is a letter. Jev gets the same question as its `instructions`, the "
               "sentence as its `state`, and the three readings as `criteria` named before / after / same_time. "
               "The *marked* text adds [[ ]] around the participial verb group and << >> around the main verb, with "
               "one sentence saying so. Template 2 asks the same thing in Portuguese (sensitivity check only).")


def _why():
    st.markdown("""
- **It answers a different question from the other families.** B3 learns from the annotated cues, and the
  encoder learns from the labelled text. A zero-shot LLM brings only what it already knows about Portuguese
  and time. If it still uses telicity, duration and the main clause the way the corpus does, the cue
  hierarchy is not just a property of this annotation.
- **Probabilities, not free text.** Three numbers per sentence go into the same macro-F1, log loss, Brier
  score and calibration error as B3 and the encoder, on **the same folds**. Nothing has to be parsed, and
  no answer can come back unusable.
- **Same perturbations as the encoder (T3).** Masking the participle, the main verb or the adverbials, or
  moving the clause to the front, shows which parts of the sentence the model relies on. The `ctrl_`
  rows mask random words of the same length as a baseline.
- **Order bias is measured, not hoped away.** The diagnostics report accuracy per option order and how
  often the three orders agree. A model that agrees with itself only by chance is visible.
- **One model family, several sizes.** The open models are all Qwen, so the size comparison is not
  confounded by different training data or tokenisers, and each runs from a pinned commit, so every run can
  be reproduced.
- **A trained decision model as the comparison.** Jev, TypeSafe's closed model, is trained to give
  calibrated decisions. Run on the same texts, folds and option orders, it shows what that training adds
  over a stock LLM read the same way. It is pinned to one version, but only TypeSafe can rerun it.
""")


def _results():
    t5 = table("t5_decision.csv")
    if t5 is None:
        st.info("No T5 tables in this checkout yet. The full runs are on Deucalion (`make decision`); copy "
                "`results/tables/t5_*` back when they finish. The 100-row smoke test with Qwen3.5-0.8B (a pipeline "
                "check, not a result) scored macro-F1 0.18 to 0.22 against 0.62 for B3 on the same rows, and its three "
                "option orders agreed on only 35% of sentences.")
        return
    _t1_chart(t5)  # same chart as the encoder page: no paired or text-unambiguous rows
    st.dataframe(t5, hide_index=True)
    diag = table("t5_decision_diag.csv")
    if diag is not None:
        st.subheader("Option-order diagnostics")
        st.dataframe(diag, hide_index=True)


def render():
    st.title("T5 · LLM decisions")
    st.caption("A stock LLM picks one of the three readings, scored on B3's folds and perturbed like the encoder. "
               "Nothing is trained.")
    t1, t2, t3, t4 = st.tabs(["How it works", "Models", "Why it makes sense", "Results"])
    with t1:
        _how()
    with t2:
        st.dataframe(MODELS, hide_index=True)
        st.caption("Each model is scored on all 1,038 clauses × 9 texts (plain, marked, 7 perturbations) × 3 option orders. "
                   "The Qwen models run on one A100 each; Jev runs through TypeSafe's API.")
    with t3:
        _why()
    with t4:
        _results()
