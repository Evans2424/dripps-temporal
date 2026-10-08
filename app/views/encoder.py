import re

import pandas as pd
import streamlit as st

from data import ROOT, results, table

from dripps import ablate

SAMPLE = "Orlando Janeiro desfilou pelas artérias da cidade, tendo apostado no contacto interpessoal."


def _methods_section() -> str:
    """The encoder section of docs/methods.md, so the page and the doc cannot drift apart."""
    md = (ROOT / "docs/methods.md").read_text(encoding="utf-8")
    m = re.search(r"### `08_encoder\.py`.*?(?=\n### )", md, re.S)
    return re.sub(r"\[([a-z][\w\-]*(?:; ?[a-z][\w\-]*)*)\]", r"[`\1`]", m.group(0)) if m else ""


def _status():
    tuned = (results.TABLES / "t1_encoder_config.csv").exists()
    if tuned:
        st.success("The tuned GPU run is in this checkout; its tables are below.")
        return True
    st.warning("**Implemented, results pending.** The code (fine-tuning, verb markers, B3 fusion, text "
               "perturbations) is complete and tested, but the tuned run on the GPU machine has not been "
               "copied into this checkout (`make encoder`, then commit `results/tables/t1_*` and `t3_*`).")
    st.markdown(
        "**First run, superseded** (fixed settings: lr 2e-5, 4 epochs, whole sentence): macro-F1 "
        "**0.469** [0.415, 0.528] against B3's **0.694** on the same folds; paired difference "
        "-0.226 [-0.288, -0.162]. Ant F1 0.25 and a seed spread of 0.033 (B3: 0.002) show the model was "
        "under-trained, not that the text carries less signal. The tuned run changes the optimisation "
        "(settings chosen inside each training fold, class-weighted loss), marks the two verbs, and adds a "
        "B3 + encoder fusion test.")
    return False


def _results():
    t1 = table("t1_encoder.csv")
    if t1 is not None:
        st.subheader("T1 · encoder against B3 (same folds, 5 seeds)")
        full = t1[~t1["model"].str.contains("text-unambiguous")]
        st.dataframe(full.drop(columns="n").round(3), hide_index=True, width="stretch")
        st.caption("Macro-F1 with 95% intervals from resampling sentence groups; the paired rows are differences from "
                   "the same resamples. 'w' is the weight of the encoder in the B3 + encoder average (0.5 was fixed in "
                   "advance; the others are post hoc).")
    cfg = table("t1_encoder_config.csv")
    if cfg is not None:
        st.subheader("Settings chosen inside the training folds")
        st.dataframe(cfg.groupby(["variant", "lr", "epochs"]).size().rename("folds").reset_index(),
                     hide_index=True, width="stretch")
        st.caption("If one corner of the grid wins every fold, the grid was too narrow.")
    t3 = table("t3_ablation.csv")
    if t3 is not None:
        st.subheader("T3 · change in the predicted reading when the text is perturbed")
        st.dataframe(t3.round(4), hide_index=True, width="stretch")
        st.caption("d_P_* is the mean change in predicted probability (plain model, out-of-fold). 'ctrl_*' masks the "
                   "same number of random words; read a perturbation net of its control. Interpreted only if the "
                   "tuned plain model reaches macro-F1 0.60.")
    span = results.TABLES / "t3_span_sample.csv"
    if span.exists():
        s = pd.read_csv(span)
        if s["participle_ok"].notna().any():
            st.caption(f"Hand check of span recovery on {len(s)} rows: participle correct in "
                       f"{int((s['participle_ok'] == 'y').sum())}, main verb correct in {int((s['main_verb_ok'] == 'y').sum())}.")


def _demo():
    st.markdown("Type a sentence with *tendo* to see what the rule-based span recovery does with it. The main verb "
                "needs spaCy's parser and is not shown here; the other spans use only the rules.")
    text = st.text_area("Sentence", SAMPLE)
    apc = ablate.apc_span(text)
    if apc is None:
        st.info("No unambiguous APC: the sentence needs exactly one *tendo* followed, within four words, by a "
                "participle (only adverbs or *sido/estado* may stand between). The pipeline leaves such rows out "
                "of the perturbations instead of guessing.")
        return
    (gs, ge), (ps, pe) = apc
    st.markdown(f"**APC verb group:** `{text[gs:ge]}` · **participle:** `{text[ps:pe]}`")
    marked, _ = ablate.mark(text, [((gs, ge), "[[ ", " ]]")])
    adv = ablate.adverbial_spans(text)
    adv = [a for a in adv if not gs <= a[0] < ge]
    moved = ablate.move_apc_initial(text)
    rows = [("marked (APC only; the pipeline also wraps the main verb in << >>)", marked),
            ("participle masked", ablate.mask_spans(text, [(ps, pe)], "<mask>")),
            ("temporal adverbials masked", ablate.mask_spans(text, adv, "<mask>") if adv else "(none found)"),
            ("APC moved to the front", moved or "(not applicable: internal comma, split quote or bracket)")]
    st.dataframe(pd.DataFrame(rows, columns=["variant", "text"]), hide_index=True, width="stretch")


def render():
    st.title("Encoder models (T1, T3)")
    st.caption("A fine-tuned xlm-roberta-base reads the sentence itself, to test whether the 17 annotated cues "
               "miss anything the text carries (T1) and which parts of the text it uses (T3).")
    t1, t2, t3 = st.tabs(["Status and results", "Architecture and procedure", "Span recovery demo"])
    with t1:
        if _status():
            _results()
    with t2:
        st.markdown(_methods_section())
    with t3:
        _demo()
