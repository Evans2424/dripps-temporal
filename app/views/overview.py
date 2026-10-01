import streamlit as st

from data import BLOCK_NAMES, corpus, table


def render():
    st.title("What carries the temporal reading of a participial clause?")
    st.caption("DRIPPS cue hierarchy · adverbial perfect participial clauses (*tendo* / *having* + participle)")

    df = corpus()
    pt = df[df["is_portuguese"]]
    c = st.columns(4)
    c[0].metric("Portuguese clauses", len(pt), help="EP, BP, AP, MP. British English is a reference only.")
    c[1].metric("sentences", pt["sentence_group"].nunique())
    c[2].metric("with no connector", f"{(df['CNT'].str.strip() == '').mean():.1%}",
                help="Across all 993 clauses: the reading must be inferred from other cues.")
    base = table("baselines.csv")
    if base is not None:
        b3 = base.set_index("model").loc["B3 logit"]
        c[3].metric("B3 macro-F1", f"{b3.macro_f1:.3f}",
                    help=f"95% interval [{b3.ci_lo:.3f}, {b3.ci_hi:.3f}], resampling sentences")

    st.subheader("Research questions")
    q = st.columns(3)
    q[0].markdown("**RQ1 · Which cues, in what order?**  \nRanking with intervals, four Portuguese varieties pooled. *Answered.*")
    q[1].markdown("**RQ2 · Do the varieties differ?**  \nSame cues, different weights, once base rates are separated. *Descriptive only; needs B6.*")
    q[2].markdown("**RQ3 · Does a model transfer?**  \nTrain on three varieties, test on the fourth, then English. *Not started.*")

    st.subheader("Main findings so far")
    imp = table("cue_importance.csv")
    if imp is not None:
        imp = imp.sort_values("importance_ablation", ascending=False)
        lines = [f"{i + 1}. **{BLOCK_NAMES[r.block]}**: {r.importance_ablation:.3f} "
                 f"[{r.ci_lo_ablation:.3f}, {r.ci_hi_ablation:.3f}]"
                 for i, r in enumerate(imp.itertuples())]
        left, right = st.columns([1, 1])
        left.markdown("**Cue hierarchy** (drop in macro-F1 when the block is removed and the model refit)  \n"
                      + "  \n".join(lines))
        right.markdown(
            "- **Main-clause aspect** sets the default reading: telic → posterior, durative → simultaneous.\n"
            "- **Tense is not a stand-in for aspect**: removing it still costs 0.071 with aspect in the model.\n"
            "- **The participial clause** matters through its position; its own aspect adds 0.001.\n"
            "- **No sign of interactions** at this size: forest and XGBoost do not beat the logit.\n"
            "- **Varieties** differ in base rates; whether they weight cues differently is open (B6)."
        )
    st.info("Use the sidebar to follow the same path as the team briefing: data → evaluation → models → results.",
            icon=":material/route:")
