import re

import altair as alt
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


def _tuned() -> bool:
    return (results.TABLES / "t1_encoder_config.csv").exists()


def _status():
    if _tuned():
        st.success("Tuned run: settings chosen inside each training fold, class-weighted loss, plain and "
                   "verb-marked variants, B3 + encoder fusion.")
        return
    st.warning("**First run only (fixed settings), shown below.** The tuned run (settings chosen inside each "
               "training fold, verb markers, B3 + encoder fusion) is implemented and tested, but its tables are "
               "not in this checkout yet (`make encoder` on the GPU machine, then commit `results/tables/t1_*` "
               "and `t3_*`).")
    st.markdown("The first run used lr 2e-5 and 4 epochs on the whole sentence. Ant F1 is 0.25 and the seed spread "
                "is 0.033 (B3: 0.002), which points to an under-trained model, not to less signal in the text. "
                "Read it as a lower bound, and the T3 shifts as exploratory for the same reason.")


def _t1_chart(t1):
    d = t1[~t1["model"].str.contains("paired") & ~t1["model"].str.contains("text-unambiguous")]
    base = alt.Chart(d).encode(y=alt.Y("model:N", title=None, sort=None))
    st.altair_chart(base.mark_bar().encode(x=alt.X("macro_f1:Q", title="out-of-fold macro-F1", scale=alt.Scale(domain=[0, 0.8])))
                    + base.mark_rule().encode(x="ci_lo:Q", x2="ci_hi:Q"), width="stretch")


def _t3_chart(t3):
    d = t3.dropna(subset=["d_P_true"])
    base = alt.Chart(d).encode(y=alt.Y("perturbation:N", title=None, sort=None))
    st.altair_chart(base.mark_bar().encode(x=alt.X("d_P_true:Q", title="change in P(true reading)"),
                                           color=alt.condition("indexof(datum.perturbation, 'ctrl_') >= 0",
                                                               alt.value("#9AA5AB"), alt.value("#4C4B9E")))
                    + base.mark_rule().encode(x="d_P_true_lo:Q", x2="d_P_true_hi:Q"), width="stretch")


def _results():
    sfx = "" if _tuned() else "_first_run"
    t1 = table(f"t1_encoder{sfx}.csv")
    if t1 is not None:
        st.subheader("T1 · encoder against B3 (same folds, 5 seeds)")
        _t1_chart(t1)
        full = t1[~t1["model"].str.contains("text-unambiguous")]
        st.dataframe(full.drop(columns="n").round(3), hide_index=True, width="stretch")
        st.caption("Macro-F1 with 95% intervals from resampling sentence groups; paired rows are differences from the "
                   "same resamples. In the tuned table, 'w' is the weight of the encoder in the B3 + encoder average "
                   "(0.5 was fixed in advance; the others are post hoc).")
    cfg = table("t1_encoder_config.csv") if _tuned() else None
    if cfg is not None:
        st.subheader("Settings chosen inside the training folds")
        st.dataframe(cfg.groupby(["variant", "lr", "epochs"]).size().rename("folds").reset_index(),
                     hide_index=True, width="stretch")
        st.caption("If one corner of the grid wins every fold, the grid was too narrow.")
    t3 = table(f"t3_ablation{sfx}.csv")
    if t3 is not None:
        st.subheader("T3 · masking experiments: change in the predicted reading")
        _t3_chart(t3)
        st.dataframe(t3.round(4), hide_index=True, width="stretch")
        st.caption("d_P_* is the mean change in predicted probability when that part of the sentence is masked (or the "
                   "participial clause moved to the front), out-of-fold, with 95% intervals over sentence groups. "
                   "'ctrl_*' masks the same number of random words, so read a perturbation net of its control "
                   "(grey bars). Interpreted only if the tuned plain model reaches macro-F1 0.60.")
    span = results.TABLES / "t3_span_sample.csv"
    if span.exists():
        s = pd.read_csv(span)
        if "participle_ok" in s and s["participle_ok"].notna().any():
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
        _status()
        _results()
    with t2:
        st.markdown(_methods_section())
    with t3:
        _demo()
